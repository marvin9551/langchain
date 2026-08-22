"""Agent factory for creating agents with middleware support."""

from __future__ import annotations

import functools
import importlib
import itertools
import re
from dataclasses import dataclass, field, fields
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Generic,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph._internal._runnable import RunnableCallable
from langgraph.constants import END, START
from langgraph.graph.state import StateGraph
from langgraph.prebuilt import ToolCallTransformer
from langgraph.prebuilt.tool_node import ToolNode
from langgraph.types import Command, Send
from langsmith import traceable
from typing_extensions import NotRequired, Required, TypedDict, overload

from langchain.agents._subagent_transformer import SubagentTransformer
from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ExtendedModelResponse,
    InputAgentState,
    JumpTo,
    ModelRequest,
    ModelResponse,
    OmitFromSchema,
    OutputAgentState,
    ResponseT,
    StateT_co,
    ToolCallRequest,
)
from langchain.agents.structured_output import (
    AutoStrategy,
    MultipleStructuredOutputsError,
    OutputToolBinding,
    ProviderStrategy,
    ProviderStrategyBinding,
    ResponseFormat,
    StructuredOutputError,
    StructuredOutputValidationError,
    ToolStrategy,
)
from langchain.chat_models import init_chat_model


@dataclass
class _ComposedExtendedModelResponse(Generic[ResponseT]):
    """Internal result from composed `wrap_model_call` middleware.

    Unlike `ExtendedModelResponse` (user-facing, single command), this holds the
    full list of commands accumulated across all middleware layers during
    composition.
    """

    model_response: ModelResponse[ResponseT]
    """The underlying model response."""

    commands: list[Command[Any]] = field(default_factory=list)
    """Commands accumulated from all middleware layers (inner-first, then outer)."""


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from langchain_core.runnables import Runnable, RunnableConfig
    from langgraph.cache.base import BaseCache
    from langgraph.graph.state import CompiledStateGraph
    from langgraph.runtime import Runtime
    from langgraph.store.base import BaseStore
    from langgraph.stream._mux import TransformerFactory
    from langgraph.types import Checkpointer

    from langchain.agents.middleware.types import ToolCallWrapper

    _ModelCallHandler = Callable[
        [ModelRequest[ContextT], Callable[[ModelRequest[ContextT]], ModelResponse]],
        ModelResponse | AIMessage | ExtendedModelResponse,
    ]

    _ComposedModelCallHandler = Callable[
        [ModelRequest[ContextT], Callable[[ModelRequest[ContextT]], ModelResponse]],
        _ComposedExtendedModelResponse,
    ]

    _AsyncModelCallHandler = Callable[
        [ModelRequest[ContextT], Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse]]],
        Awaitable[ModelResponse | AIMessage | ExtendedModelResponse],
    ]

    _ComposedAsyncModelCallHandler = Callable[
        [ModelRequest[ContextT], Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse]]],
        Awaitable[_ComposedExtendedModelResponse],
    ]


STRUCTURED_OUTPUT_ERROR_TEMPLATE = "Error: {error}\n Please fix your mistakes."

DYNAMIC_TOOL_ERROR_TEMPLATE = """
Middleware added tools that the agent doesn't know how to execute.

Unknown tools: {unknown_tool_names}
Registered tools: {available_tool_names}

This happens when middleware modifies `request.tools` in `wrap_model_call` to include
tools that weren't passed to `create_agent()`.

How to fix this:

Option 1: Register tools at agent creation (recommended for most cases)
    Pass the tools to `create_agent(tools=[...])` or set them on `middleware.tools`.
    This makes tools available for every agent invocation.

Option 2: Handle dynamic tools in middleware (for tools created at runtime)
    Implement `wrap_tool_call` to execute tools that are added dynamically:

    class MyMiddleware(AgentMiddleware):
        def wrap_tool_call(self, request, handler):
            if request.tool_call["name"] == "dynamic_tool":
                # Execute the dynamic tool yourself or override with tool instance
                return handler(request.override(tool=my_dynamic_tool))
            return handler(request)
""".strip()


def _scrub_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    """Remove `runtime` and `handler` from trace inputs before sending to LangSmith."""
    filtered = inputs.copy()
    filtered.pop("handler", None)
    req = filtered.get("request")
    if isinstance(req, (ModelRequest, ToolCallRequest)):
        filtered["request"] = {
            f.name: getattr(req, f.name) for f in fields(req) if f.name != "runtime"
        }
    return filtered


FALLBACK_MODELS_WITH_STRUCTURED_OUTPUT = [
    # If model profile data are not available, model names matching these patterns
    # are assumed to support provider-native structured output. These are regexes
    # so matches stay bounded to model-name segments instead of arbitrary substrings.
    r"(^|[/:.])gpt-4\.1($|[-/:])",
    r"(^|[/:.])gpt-4o($|[-/:])",
    r"(^|[/:.])gpt-5($|[-/:])",
    r"(^|[/:.])gpt-5\.1($|[-/:])",
    r"(^|[/:.])gpt-5\.2(-\d{4}-\d{2}-\d{2})?($|[/:])",
    r"(^|[/:.])gpt-5\.2-(chat|codex)($|[-/:])",
    r"(^|[/:.])gpt-5\.3($|[-/:])",
    r"(^|[/:.])gpt-5\.4(-\d{4}-\d{2}-\d{2})?($|[/:])",
    r"(^|[/:.])gpt-5\.4-(mini|nano)($|[-/:])",
    r"(^|[/:.])gpt-5\.5($|[-/:])",
    r"(^|[/:.])claude-(fable|mythos)-5(?:-\d{8})?(?:-v\d(?::\d)?)?($|[/:])",
    r"(^|[/:.])claude-haiku-4-5(?:-\d{8})?(?:-v\d(?::\d)?)?($|[/:])",
    r"(^|[/:.])claude-opus-4-(5|6|7|8)(?:-\d{8})?(?:-v\d(?::\d)?)?($|[/:])",
    r"(^|[/:.])claude-sonnet-4-(5|6)(?:-\d{8})?(?:-v\d(?::\d)?)?($|[/:])",
    r"(^|[/:.])grok-4($|[-.:/])",
    r"(^|[/:.])grok-build($|[-/:])",
]


def _normalize_to_model_response(
    result: ModelResponse | AIMessage | ExtendedModelResponse,
) -> ModelResponse:
    """Normalize middleware return value to ModelResponse.

    At inner composition boundaries, `ExtendedModelResponse` is unwrapped to its
    underlying `ModelResponse` so that inner middleware always sees `ModelResponse`
    from the handler.
    """
    if isinstance(result, AIMessage):
        return ModelResponse(result=[result], structured_response=None)
    if isinstance(result, ExtendedModelResponse):
        return result.model_response
    return result


def _build_commands(
    model_response: ModelResponse,
    middleware_commands: list[Command[Any]] | None = None,
) -> list[Command[Any]]:
    """Build a list of Commands from a model response and middleware commands.

    The first Command contains the model response state (messages and optional
    structured_response). Middleware commands are appended as-is.

    Args:
        model_response: The model response containing messages and optional
            structured output.
        middleware_commands: Commands accumulated from middleware layers during
            composition (inner-first ordering).

    Returns:
        List of `Command` objects ready to be returned from a model node.
    """
    state: dict[str, Any] = {"messages": model_response.result}

    if model_response.structured_response is not None:
        state["structured_response"] = model_response.structured_response

    for cmd in middleware_commands or []:
        if cmd.goto:
            msg = (
                "Command goto is not yet supported in wrap_model_call middleware. "
                "Use the jump_to state field with before_model/after_model hooks instead."
            )
            raise NotImplementedError(msg)
        if cmd.resume:
            msg = "Command resume is not yet supported in wrap_model_call middleware."
            raise NotImplementedError(msg)
        if cmd.graph:
            msg = "Command graph is not yet supported in wrap_model_call middleware."
            raise NotImplementedError(msg)

    commands: list[Command[Any]] = [Command(update=state)]
    commands.extend(middleware_commands or [])
    return commands


def _chain_model_call_handlers(
    handlers: Sequence[_ModelCallHandler[ContextT]],
) -> _ComposedModelCallHandler[ContextT] | None:
    """Compose multiple `wrap_model_call` handlers into single middleware stack.

    Composes handlers so first in list becomes outermost layer. Each handler receives a
    handler callback to execute inner layers. Commands from each layer are accumulated
    into a list (inner-first, then outer) without merging.

    Args:
        handlers: List of handlers.

            First handler wraps all others.

    Returns:
        Composed handler returning `_ComposedExtendedModelResponse`,
        or `None` if handlers empty.
    """
    if not handlers:
        return None

    def _to_composed_result(
        result: ModelResponse | AIMessage | ExtendedModelResponse | _ComposedExtendedModelResponse,
        extra_commands: list[Command[Any]] | None = None,
    ) -> _ComposedExtendedModelResponse:
        """Normalize any handler result to _ComposedExtendedModelResponse."""
        commands: list[Command[Any]] = list(extra_commands or [])
        if isinstance(result, _ComposedExtendedModelResponse):
            commands.extend(result.commands)
            model_response = result.model_response
        elif isinstance(result, ExtendedModelResponse):
            model_response = result.model_response
            if result.command is not None:
                commands.append(result.command)
        else:
            model_response = _normalize_to_model_response(result)

        return _ComposedExtendedModelResponse(model_response=model_response, commands=commands)

    if len(handlers) == 1:
        single_handler = handlers[0]

        def normalized_single(
            request: ModelRequest[ContextT],
            handler: Callable[[ModelRequest[ContextT]], ModelResponse],
        ) -> _ComposedExtendedModelResponse:
            return _to_composed_result(single_handler(request, handler))

        return normalized_single

    def compose_two(
        outer: _ModelCallHandler[ContextT] | _ComposedModelCallHandler[ContextT],
        inner: _ModelCallHandler[ContextT] | _ComposedModelCallHandler[ContextT],
    ) -> _ComposedModelCallHandler[ContextT]:
        """Compose two handlers where outer wraps inner."""

        def composed(
            request: ModelRequest[ContextT],
            handler: Callable[[ModelRequest[ContextT]], ModelResponse],
        ) -> _ComposedExtendedModelResponse:
            # Closure variable to capture inner's commands before normalizing
            accumulated_commands: list[Command[Any]] = []

            def inner_handler(req: ModelRequest[ContextT]) -> ModelResponse:
                # Clear on each call for retry safety
                accumulated_commands.clear()
                inner_result = inner(req, handler)
                if isinstance(inner_result, _ComposedExtendedModelResponse):
                    accumulated_commands.extend(inner_result.commands)
                    return inner_result.model_response
                if isinstance(inner_result, ExtendedModelResponse):
                    if inner_result.command is not None:
                        accumulated_commands.append(inner_result.command)
                    return inner_result.model_response
                return _normalize_to_model_response(inner_result)

            outer_result = outer(request, inner_handler)
            return _to_composed_result(
                outer_result,
                extra_commands=accumulated_commands or None,
            )

        return composed

    # Compose right-to-left: outer(inner(innermost(handler)))
    composed_handler = compose_two(handlers[-2], handlers[-1])
    for h in reversed(handlers[:-2]):
        composed_handler = compose_two(h, composed_handler)

    return composed_handler


def _chain_async_model_call_handlers(
    handlers: Sequence[_AsyncModelCallHandler[ContextT]],
) -> _ComposedAsyncModelCallHandler[ContextT] | None:
    """Compose multiple async `wrap_model_call` handlers into single middleware stack.

    Commands from each layer are accumulated into a list (inner-first, then outer)
    without merging.

    Args:
        handlers: List of async handlers.

            First handler wraps all others.

    Returns:
        Composed async handler returning `_ComposedExtendedModelResponse`,
        or `None` if handlers empty.
    """
    if not handlers:
        return None

    def _to_composed_result(
        result: ModelResponse | AIMessage | ExtendedModelResponse | _ComposedExtendedModelResponse,
        extra_commands: list[Command[Any]] | None = None,
    ) -> _ComposedExtendedModelResponse:
        """Normalize any handler result to _ComposedExtendedModelResponse."""
        commands: list[Command[Any]] = list(extra_commands or [])
        if isinstance(result, _ComposedExtendedModelResponse):
            commands.extend(result.commands)
            model_response = result.model_response
        elif isinstance(result, ExtendedModelResponse):
            model_response = result.model_response
            if result.command is not None:
                commands.append(result.command)
        else:
            model_response = _normalize_to_model_response(result)

        return _ComposedExtendedModelResponse(model_response=model_response, commands=commands)

    if len(handlers) == 1:
        single_handler = handlers[0]

        async def normalized_single(
            request: ModelRequest[ContextT],
            handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse]],
        ) -> _ComposedExtendedModelResponse:
            return _to_composed_result(await single_handler(request, handler))

        return normalized_single

    def compose_two(
        outer: _AsyncModelCallHandler[ContextT] | _ComposedAsyncModelCallHandler[ContextT],
        inner: _AsyncModelCallHandler[ContextT] | _ComposedAsyncModelCallHandler[ContextT],
    ) -> _ComposedAsyncModelCallHandler[ContextT]:
        """Compose two async handlers where outer wraps inner."""

        async def composed(
            request: ModelRequest[ContextT],
            handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse]],
        ) -> _ComposedExtendedModelResponse:
            # Closure variable to capture inner's commands before normalizing
            accumulated_commands: list[Command[Any]] = []

            async def inner_handler(req: ModelRequest[ContextT]) -> ModelResponse:
                # Clear on each call for retry safety
                accumulated_commands.clear()
                inner_result = await inner(req, handler)
                if isinstance(inner_result, _ComposedExtendedModelResponse):
                    accumulated_commands.extend(inner_result.commands)
                    return inner_result.model_response
                if isinstance(inner_result, ExtendedModelResponse):
                    if inner_result.command is not None:
                        accumulated_commands.append(inner_result.command)
                    return inner_result.model_response
                return _normalize_to_model_response(inner_result)

            outer_result = await outer(request, inner_handler)
            return _to_composed_result(
                outer_result,
                extra_commands=accumulated_commands or None,
            )

        return composed

    # Compose right-to-left: outer(inner(innermost(handler)))
    composed_handler = compose_two(handlers[-2], handlers[-1])
    for h in reversed(handlers[:-2]):
        composed_handler = compose_two(h, composed_handler)

    return composed_handler


@functools.lru_cache(maxsize=100)
def _get_schema_type_hints(schema: type) -> dict[str, Any]:
    """Return cached type hints for a schema."""
    return get_type_hints(schema, include_extras=True)


def _resolve_schemas(schemas: list[type]) -> tuple[type, type, type]:
    """Resolve state, input, and output schemas for the given schemas.

    Schemas are merged in list order; later entries override earlier ones when the
    same field is declared by multiple schemas.  Duplicates are harmless — a type
    that appears more than once is processed at its last position.
    """
    schema_hints = {schema: _get_schema_type_hints(schema) for schema in schemas}
    return (
        _resolve_schema(schema_hints, "StateSchema", None),
        _resolve_schema(schema_hints, "InputSchema", "input"),
        _resolve_schema(schema_hints, "OutputSchema", "output"),
    )


def _resolve_schema(
    schema_hints: dict[type, dict[str, Any]],
    schema_name: str,
    omit_flag: str | None = None,
) -> type:
    """Resolve schema by merging schemas and optionally respecting `OmitFromSchema` annotations.

    Args:
        schema_hints: Resolved schema annotations to merge
        schema_name: Name for the generated `TypedDict`
        omit_flag: If specified, omit fields with this flag set (`'input'` or
            `'output'`)

    Returns:
        Merged schema as `TypedDict`
    """
    all_annotations = {}

    for hints in schema_hints.values():
        for field_name, field_type in hints.items():
            should_omit = False

            if omit_flag:
                metadata = _extract_metadata(field_type)
                for meta in metadata:
                    if isinstance(meta, OmitFromSchema) and getattr(meta, omit_flag) is True:
                        should_omit = True
                        break

            if not should_omit:
                all_annotations[field_name] = field_type

    # `TypedDict` dynamically creates a class, but type checkers don't infer that
    # the runtime result satisfies this function's `type` return contract.
    return cast("type", TypedDict(schema_name, all_annotations))  # type: ignore[operator]


def _extract_metadata(type_: type) -> list[Any]:
    """Extract metadata from a field type, handling `Required`/`NotRequired` and `Annotated` wrappers."""  # noqa: E501
    # Handle Required[Annotated[...]] or NotRequired[Annotated[...]]
    if get_origin(type_) in {Required, NotRequired}:
        inner_type = get_args(type_)[0]
        if get_origin(inner_type) is Annotated:
            return list(get_args(inner_type)[1:])

    # Handle direct Annotated[...]
    elif get_origin(type_) is Annotated:
        return list(get_args(type_)[1:])

    return []


def _get_can_jump_to(middleware: AgentMiddleware[Any, Any], hook_name: str) -> list[JumpTo]:
    """Get the `can_jump_to` list from either sync or async hook methods.

    Args:
        middleware: The middleware instance to inspect.
        hook_name: The name of the hook (`'before_model'` or `'after_model'`).

    Returns:
        List of jump destinations, or empty list if not configured.
    """
    # Get the base class method for comparison
    base_sync_method = getattr(AgentMiddleware, hook_name, None)
    base_async_method = getattr(AgentMiddleware, f"a{hook_name}", None)

    # Try sync method first - only if it's overridden from base class
    sync_method = getattr(middleware.__class__, hook_name, None)
    if (
        sync_method
        and sync_method is not base_sync_method
        and hasattr(sync_method, "__can_jump_to__")
    ):
        # `hasattr` proves the metadata exists at runtime, but not its value type.
        return cast("list[JumpTo]", sync_method.__can_jump_to__)

    # Try async method - only if it's overridden from base class
    async_method = getattr(middleware.__class__, f"a{hook_name}", None)
    if (
        async_method
        and async_method is not base_async_method
        and hasattr(async_method, "__can_jump_to__")
    ):
        # `hasattr` proves the metadata exists at runtime, but not its value type.
        return cast("list[JumpTo]", async_method.__can_jump_to__)

    return []


def _supports_provider_strategy(
    model: str | BaseChatModel, tools: list[BaseTool | dict[str, Any]] | None = None
) -> bool:
    """Check if a model supports provider-specific structured output.

    Args:
        model: Model name string or `BaseChatModel` instance.
        tools: Optional list of tools provided to the agent.

            Needed because some models don't support structured output together with tool calling.

    Returns:
        `True` if the model supports provider-specific structured output, `False` otherwise.
    """
    model_name: str | None = None
    if isinstance(model, str):
        model_name = model
    elif isinstance(model, BaseChatModel):
        model_name = (
            getattr(model, "model_name", None)
            or getattr(model, "model", None)
            or getattr(model, "model_id", "")
        )
        model_profile = model.profile
        if (
            model_profile is not None
            and model_profile.get("structured_output")
            # We make an exception for Gemini < 3-series models, which currently do not support
            # simultaneous tool use with structured output; 3-series can.
            and not (
                tools
                and isinstance(model_name, str)
                and "gemini" in model_name.lower()
                and "gemini-3" not in model_name.lower()
            )
        ):
            return True

    return (
        any(
            re.search(pattern, model_name.lower())
            for pattern in FALLBACK_MODELS_WITH_STRUCTURED_OUTPUT
        )
        if model_name
        else False
    )


def _is_openai_compatible_model(model: BaseChatModel) -> bool:
    """Check if a model inherits from `BaseChatOpenAI`.

    Used to redundantly set `strict=True` on tools when `response_format` is
    provided, as older versions of `langchain-openai` do not auto-set it.
    Covers `ChatOpenAI`, `ChatDeepSeek`, `ChatXAI`, etc.

    Args:
        model: The chat model to check.

    Returns:
        `True` if the model inherits from `BaseChatOpenAI`, `False` otherwise.
    """
    try:
        base_chat_openai = importlib.import_module("langchain_openai.chat_models.base")
    except ImportError:
        return False
    return isinstance(model, base_chat_openai.BaseChatOpenAI)


def _handle_structured_output_error(
    exception: Exception,
    response_format: ResponseFormat[Any],
) -> tuple[bool, str]:
    """Handle structured output error.

    Returns `(should_retry, retry_tool_message)`.
    """
    if not isinstance(response_format, ToolStrategy):
        return False, ""

    handle_errors = response_format.handle_errors

    if handle_errors is False:
        return False, ""
    if handle_errors is True:
        return True, STRUCTURED_OUTPUT_ERROR_TEMPLATE.format(error=str(exception))
    if isinstance(handle_errors, str):
        return True, handle_errors
    if isinstance(handle_errors, type):
        if issubclass(handle_errors, Exception) and isinstance(exception, handle_errors):
            return True, STRUCTURED_OUTPUT_ERROR_TEMPLATE.format(error=str(exception))
        return False, ""
    if isinstance(handle_errors, tuple):
        if any(isinstance(exception, exc_type) for exc_type in handle_errors):
            return True, STRUCTURED_OUTPUT_ERROR_TEMPLATE.format(error=str(exception))
        return False, ""
    return True, handle_errors(exception)


def _chain_tool_call_wrappers(
    wrappers: Sequence[ToolCallWrapper],
) -> ToolCallWrapper | None:
    """Compose wrappers into middleware stack (first = outermost).

    Args:
        wrappers: Wrappers in middleware order.

    Returns:
        Composed wrapper, or `None` if empty.

    Example:
        ```python
        wrapper = _chain_tool_call_wrappers([auth, cache, retry])
        # Request flows: auth -> cache -> retry -> tool
        # Response flows: tool -> retry -> cache -> auth
        ```
    """
    if not wrappers:
        return None

    if len(wrappers) == 1:
        return wrappers[0]

    def compose_two(outer: ToolCallWrapper, inner: ToolCallWrapper) -> ToolCallWrapper:
        """Compose two wrappers where outer wraps inner."""

        def composed(
            request: ToolCallRequest,
            execute: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
        ) -> ToolMessage | Command[Any]:
            # Create a callable that invokes inner with the original execute
            def call_inner(req: ToolCallRequest) -> ToolMessage | Command[Any]:
                return inner(req, execute)

            # Outer can call call_inner multiple times
            return outer(request, call_inner)

        return composed

    # Chain all wrappers: first -> second -> ... -> last
    result = wrappers[-1]
    for wrapper in reversed(wrappers[:-1]):
        result = compose_two(wrapper, result)

    return result


def _chain_async_tool_call_wrappers(
    wrappers: Sequence[
        Callable[
            [ToolCallRequest, Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]]],
            Awaitable[ToolMessage | Command[Any]],
        ]
    ],
) -> (
    Callable[
        [ToolCallRequest, Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]]],
        Awaitable[ToolMessage | Command[Any]],
    ]
    | None
):
    """Compose async wrappers into middleware stack (first = outermost).

    Args:
        wrappers: Async wrappers in middleware order.

    Returns:
        Composed async wrapper, or `None` if empty.
    """
    if not wrappers:
        return None

    if len(wrappers) == 1:
        return wrappers[0]

    def compose_two(
        outer: Callable[
            [ToolCallRequest, Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]]],
            Awaitable[ToolMessage | Command[Any]],
        ],
        inner: Callable[
            [ToolCallRequest, Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]]],
            Awaitable[ToolMessage | Command[Any]],
        ],
    ) -> Callable[
        [ToolCallRequest, Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]]],
        Awaitable[ToolMessage | Command[Any]],
    ]:
        """Compose two async wrappers where outer wraps inner."""

        async def composed(
            request: ToolCallRequest,
            execute: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
        ) -> ToolMessage | Command[Any]:
            # Create an async callable that invokes inner with the original execute
            async def call_inner(req: ToolCallRequest) -> ToolMessage | Command[Any]:
                return await inner(req, execute)

            # Outer can call call_inner multiple times
            return await outer(request, call_inner)

        return composed

    # Chain all wrappers: first -> second -> ... -> last
    result = wrappers[-1]
    for wrapper in reversed(wrappers[:-1]):
        result = compose_two(wrapper, result)

    return result


# ============================================================================
# create_agent —— LangChain v1 的 Agent 工厂（本文件 = 装配车间）
# 它做三件事：
#   1. 规整 model / tools / middleware / response_format 等入参，解析 state schema
#   2. 从零装配一张 StateGraph（ReAct 循环）：节点与边全部是"条件性添加"的
#   3. 编译出 CompiledStateGraph，附 recursion_limit=9999 等运行时配置
#
# 生成的图结构（顶层视角）：
#   START → entry_node →(before_agent/before_model 链)→ model
#   model →(条件边)→ tools（每个未兑现 tool_call 发一个 Send 并行 fan-out）
#   tools →(条件边)→ loop_entry_node（循环） / exit_node（return_direct 短路）
#   model →(条件边)→ exit_node（after_agent 链 → END）
#
# 注释集中区：节点添加(~1501)、四锚点(~1592)、条件边注册(~1620)、
# 边函数实现(_make_model_to_tools_edge ~1840 / _make_tools_to_model_edge ~1921)
# ============================================================================
# No `response_format`: there is no structured output, so `ResponseT` resolves to `Any`.
@overload
def create_agent(
    model: str | BaseChatModel,
    tools: Sequence[BaseTool | Callable[..., Any] | dict[str, Any]] | None = None,
    *,
    system_prompt: str | SystemMessage | None = None,
    middleware: Sequence[AgentMiddleware[StateT_co, ContextT]] = (),
    response_format: None = None,
    state_schema: None = None,
    context_schema: type[ContextT] | None = None,
    checkpointer: Checkpointer | None = None,
    store: BaseStore | None = None,
    interrupt_before: list[str] | None = None,
    interrupt_after: list[str] | None = None,
    debug: bool = False,
    name: str | None = None,
    cache: BaseCache[Any] | None = None,
    transformers: Sequence[TransformerFactory] | None = None,
) -> CompiledStateGraph[AgentState[Any], ContextT, InputAgentState, OutputAgentState[Any]]: ...


# Raw-dict `response_format`: structured output is an untyped `dict[str, Any]`.
@overload
def create_agent(
    model: str | BaseChatModel,
    tools: Sequence[BaseTool | Callable[..., Any] | dict[str, Any]] | None = None,
    *,
    system_prompt: str | SystemMessage | None = None,
    middleware: Sequence[AgentMiddleware[StateT_co, ContextT]] = (),
    response_format: dict[str, Any],
    state_schema: type[AgentState[dict[str, Any]]] | None = None,
    context_schema: type[ContextT] | None = None,
    checkpointer: Checkpointer | None = None,
    store: BaseStore | None = None,
    interrupt_before: list[str] | None = None,
    interrupt_after: list[str] | None = None,
    debug: bool = False,
    name: str | None = None,
    cache: BaseCache[Any] | None = None,
    transformers: Sequence[TransformerFactory] | None = None,
) -> CompiledStateGraph[
    AgentState[dict[str, Any]], ContextT, InputAgentState, OutputAgentState[dict[str, Any]]
]: ...


# Schema-typed `response_format`: `ResponseT` is inferred from the schema/type.
@overload
def create_agent(
    model: str | BaseChatModel,
    tools: Sequence[BaseTool | Callable[..., Any] | dict[str, Any]] | None = None,
    *,
    system_prompt: str | SystemMessage | None = None,
    middleware: Sequence[AgentMiddleware[StateT_co, ContextT]] = (),
    response_format: ResponseFormat[ResponseT] | type[ResponseT] | None = None,
    state_schema: type[AgentState[ResponseT]] | None = None,
    context_schema: type[ContextT] | None = None,
    checkpointer: Checkpointer | None = None,
    store: BaseStore | None = None,
    interrupt_before: list[str] | None = None,
    interrupt_after: list[str] | None = None,
    debug: bool = False,
    name: str | None = None,
    cache: BaseCache[Any] | None = None,
    transformers: Sequence[TransformerFactory] | None = None,
) -> CompiledStateGraph[
    AgentState[ResponseT], ContextT, InputAgentState, OutputAgentState[ResponseT]
]: ...


def create_agent(
    model: str | BaseChatModel,
    tools: Sequence[BaseTool | Callable[..., Any] | dict[str, Any]] | None = None,
    *,
    system_prompt: str | SystemMessage | None = None,
    middleware: Sequence[AgentMiddleware[StateT_co, ContextT]] = (),
    response_format: ResponseFormat[ResponseT] | type[ResponseT] | dict[str, Any] | None = None,
    state_schema: type[AgentState[ResponseT]] | None = None,
    context_schema: type[ContextT] | None = None,
    checkpointer: Checkpointer | None = None,
    store: BaseStore | None = None,
    interrupt_before: list[str] | None = None,
    interrupt_after: list[str] | None = None,
    debug: bool = False,
    name: str | None = None,
    cache: BaseCache[Any] | None = None,
    transformers: Sequence[TransformerFactory] | None = None,
) -> CompiledStateGraph[
    AgentState[ResponseT], ContextT, InputAgentState, OutputAgentState[ResponseT]
]:
    """Creates an agent graph that calls tools in a loop until a stopping condition is met.

    For more details on using `create_agent`,
    visit the [Agents](https://docs.langchain.com/oss/python/langchain/agents) docs.

    Args:
        model: The language model for the agent.

            Can be a string identifier (e.g., `"openai:gpt-5.5"`) or a direct chat model
            instance (e.g., [`ChatOpenAI`][langchain_openai.ChatOpenAI] or other another
            [LangChain chat model](https://docs.langchain.com/oss/python/integrations/chat)).

            For a full list of supported model strings, see
            [`init_chat_model`][langchain.chat_models.init_chat_model(model_provider)].

            !!! tip ""

                See the [Models](https://docs.langchain.com/oss/python/langchain/models)
                docs for more information.
        tools: A list of tools, `dict`, or `Callable`.

            If `None` or an empty list, the agent will consist of a model node without a
            tool calling loop.


            !!! tip ""

                See the [Tools](https://docs.langchain.com/oss/python/langchain/tools)
                docs for more information.
        system_prompt: An optional system prompt for the LLM.

            Can be a `str` (which will be converted to a `SystemMessage`) or a
            `SystemMessage` instance directly. The system message is added to the
            beginning of the message list when calling the model.
        middleware: A sequence of middleware instances to apply to the agent.

            Middleware can intercept and modify agent behavior at various stages.

            !!! tip ""

                See the [Middleware](https://docs.langchain.com/oss/python/langchain/middleware)
                docs for more information.
        response_format: An optional configuration for structured responses.

            Can be a `ToolStrategy`, `ProviderStrategy`, or a Pydantic model class.

            If provided, the agent will handle structured output during the
            conversation flow.

            Raw schemas will be wrapped in an appropriate strategy based on model
            capabilities.

            !!! tip ""

                See the [Structured output](https://docs.langchain.com/oss/python/langchain/structured-output)
                docs for more information.
        state_schema: An optional `TypedDict` schema that extends `AgentState`.

            When provided, this schema is used instead of `AgentState` as the base
            schema for merging with middleware state schemas. This allows users to
            add custom state fields without needing to create custom middleware.

            Generally, it's recommended to use `state_schema` extensions via middleware
            to keep relevant extensions scoped to corresponding hooks / tools.
        context_schema: An optional schema for runtime context.
        checkpointer: An optional checkpoint saver object.

            Used for persisting the state of the graph (e.g., as chat memory) for a
            single thread (e.g., a single conversation).
        store: An optional store object.

            Used for persisting data across multiple threads (e.g., multiple
            conversations / users).
        interrupt_before: An optional list of node names to interrupt before.

            Useful if you want to add a user confirmation or other interrupt
            before taking an action.
        interrupt_after: An optional list of node names to interrupt after.

            Useful if you want to return directly or run additional processing
            on an output.
        debug: Whether to enable verbose logging for graph execution.

            When enabled, prints detailed information about each node execution, state
            updates, and transitions during agent runtime. Useful for debugging
            middleware behavior and understanding agent execution flow.
        name: An optional name for the `CompiledStateGraph`.

            This name will be automatically used when adding the agent graph to
            another graph as a subgraph node - particularly useful for building
            multi-agent systems.
        cache: An optional `BaseCache` instance to enable caching of graph execution.
        transformers: Optional sequence of scope-aware `StreamTransformer`
            factories to register on the compiled graph in addition to
            the agent defaults. Each factory is invoked as `factory(scope)`
            so every invocation receives a fresh instance. The final order
            on the compiled graph is: `ToolCallTransformer`, then any
            factories declared by middleware via
            `AgentMiddleware.transformers`, then any factories supplied here.

    Returns:
        A compiled `StateGraph` that can be used for chat interactions.

    Raises:
        AssertionError: If duplicate middleware instances are provided.

    The agent node calls the language model with the messages list (after applying
    the system prompt). If the resulting [`AIMessage`][langchain.messages.AIMessage]
    contains `tool_calls`, the graph will then call the tools. The tools node executes
    the tools and adds the responses to the messages list as
    [`ToolMessage`][langchain.messages.ToolMessage] objects. The agent node then calls
    the language model again. The process repeats until no more `tool_calls` are present
    in the response. The agent then returns the full list of messages.

    Example:
        ```python
        from langchain.agents import create_agent


        def check_weather(location: str) -> str:
            '''Return the weather forecast for the specified location.'''
            return f"It's always sunny in {location}"


        graph = create_agent(
            model="anthropic:claude-sonnet-4-5-20250929",
            tools=[check_weather],
            system_prompt="You are a helpful assistant",
        )
        inputs = {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
        for chunk in graph.stream(inputs, stream_mode="updates"):
            print(chunk)
        ```
    """
    # init chat model
    if isinstance(model, str):
        model = init_chat_model(model)

    # Convert system_prompt to SystemMessage if needed
    system_message: SystemMessage | None = None
    if system_prompt is not None:
        if isinstance(system_prompt, SystemMessage):
            system_message = system_prompt
        else:
            system_message = SystemMessage(content=system_prompt)

    # Handle tools being None or empty
    if tools is None:
        tools = []

    # Convert response format and setup structured output tools
    # Raw schemas are wrapped in AutoStrategy to preserve auto-detection intent.
    # AutoStrategy is converted to ToolStrategy upfront to calculate tools during agent creation,
    # but may be replaced with ProviderStrategy later based on model capabilities.
    initial_response_format: ToolStrategy[Any] | ProviderStrategy[Any] | AutoStrategy[Any] | None
    if response_format is None:
        initial_response_format = None
    elif isinstance(response_format, (ToolStrategy, ProviderStrategy, AutoStrategy)):
        # Explicit Tool/Provider strategy, or AutoStrategy for later capability detection
        initial_response_format = response_format
    else:
        # Raw schema - wrap in AutoStrategy to enable auto-detection
        initial_response_format = AutoStrategy(schema=response_format)

    # For AutoStrategy, convert to ToolStrategy to setup tools upfront
    # (may be replaced with ProviderStrategy later based on model)
    tool_strategy_for_setup: ToolStrategy[Any] | None = None
    if isinstance(initial_response_format, AutoStrategy):
        tool_strategy_for_setup = ToolStrategy(schema=initial_response_format.schema)
    elif isinstance(initial_response_format, ToolStrategy):
        tool_strategy_for_setup = initial_response_format

    structured_output_tools: dict[str, OutputToolBinding[Any]] = {}
    if tool_strategy_for_setup:
        for response_schema in tool_strategy_for_setup.schema_specs:
            structured_tool_info = OutputToolBinding.from_schema_spec(response_schema)
            structured_output_tools[structured_tool_info.tool.name] = structured_tool_info
    middleware_tools = [t for m in middleware for t in getattr(m, "tools", [])]

    # Collect middleware with wrap_tool_call or awrap_tool_call hooks
    # Include middleware with either implementation to ensure NotImplementedError is raised
    # when middleware doesn't support the execution path
    middleware_w_wrap_tool_call = [
        m
        for m in middleware
        if m.__class__.wrap_tool_call is not AgentMiddleware.wrap_tool_call
        or m.__class__.awrap_tool_call is not AgentMiddleware.awrap_tool_call
    ]

    # Chain all wrap_tool_call handlers into a single composed handler
    wrap_tool_call_wrapper = None
    if middleware_w_wrap_tool_call:
        wrappers = [
            traceable(name=f"{m.name}.wrap_tool_call", process_inputs=_scrub_inputs)(
                m.wrap_tool_call
            )
            for m in middleware_w_wrap_tool_call
        ]
        wrap_tool_call_wrapper = _chain_tool_call_wrappers(wrappers)

    # Collect middleware with awrap_tool_call or wrap_tool_call hooks
    # Include middleware with either implementation to ensure NotImplementedError is raised
    # when middleware doesn't support the execution path
    middleware_w_awrap_tool_call = [
        m
        for m in middleware
        if m.__class__.awrap_tool_call is not AgentMiddleware.awrap_tool_call
        or m.__class__.wrap_tool_call is not AgentMiddleware.wrap_tool_call
    ]

    # Chain all awrap_tool_call handlers into a single composed async handler
    awrap_tool_call_wrapper = None
    if middleware_w_awrap_tool_call:
        async_wrappers = [
            traceable(name=f"{m.name}.awrap_tool_call", process_inputs=_scrub_inputs)(
                m.awrap_tool_call
            )
            for m in middleware_w_awrap_tool_call
        ]
        awrap_tool_call_wrapper = _chain_async_tool_call_wrappers(async_wrappers)

    # Setup tools
    tool_node: ToolNode | None = None
    # Extract built-in provider tools (dict format) and regular tools (BaseTool/callables)
    built_in_tools = [t for t in tools if isinstance(t, dict)]
    regular_tools = [t for t in tools if not isinstance(t, dict)]

    # Tools that require client-side execution (must be in ToolNode)
    available_tools = middleware_tools + regular_tools

    # Create ToolNode if we have client-side tools OR if middleware defines wrap_tool_call
    # (which may handle dynamically registered tools)
    tool_node = (
        ToolNode(
            tools=available_tools,
            wrap_tool_call=wrap_tool_call_wrapper,
            awrap_tool_call=awrap_tool_call_wrapper,
        )
        if available_tools or wrap_tool_call_wrapper or awrap_tool_call_wrapper
        else None
    )

    # Default tools for ModelRequest initialization
    # Use converted BaseTool instances from ToolNode (not raw callables)
    # Include built-ins and converted tools (can be changed dynamically by middleware)
    # Structured tools are NOT included - they're added dynamically based on response_format
    if tool_node:
        default_tools = list(tool_node.tools_by_name.values()) + built_in_tools
    else:
        default_tools = list(built_in_tools)

    # validate middleware
    if len({m.name for m in middleware}) != len(middleware):
        msg = "Please remove duplicate middleware instances."
        raise AssertionError(msg)
    # 按钩子类型筛选中间件，得到四个子列表（保持注册顺序）：
    #   判断方式：m.__class__.xxx is not AgentMiddleware.xxx —— 通过函数对象身份
    #   比较，子类 override 了钩子后，类的该属性指向子类方法，不再等于基类原版方法。
    #   "or" 同时检查 sync/async 两个变体，只要重写任意一个就计入该列表。
    #   - middleware_w_before_agent: 有 before_agent 钩子（agent 启动时只跑一次）
    #   - middleware_w_before_model: 有 before_model 钩子（每轮循环、调模型前执行）
    #   - middleware_w_after_model:  有 after_model 钩子（每轮循环、模型响应后执行）
    #   - middleware_w_after_agent:  有 after_agent 钩子（agent 结束时只跑一次）
    #   [0] 是第一个注册的中间件，[-1] 是最后一个。
    middleware_w_before_agent = [
        m
        for m in middleware
        if m.__class__.before_agent is not AgentMiddleware.before_agent
        or m.__class__.abefore_agent is not AgentMiddleware.abefore_agent
    ]
    middleware_w_before_model = [
        m
        for m in middleware
        if m.__class__.before_model is not AgentMiddleware.before_model
        or m.__class__.abefore_model is not AgentMiddleware.abefore_model
    ]
    middleware_w_after_model = [
        m
        for m in middleware
        if m.__class__.after_model is not AgentMiddleware.after_model
        or m.__class__.aafter_model is not AgentMiddleware.aafter_model
    ]
    middleware_w_after_agent = [
        m
        for m in middleware
        if m.__class__.after_agent is not AgentMiddleware.after_agent
        or m.__class__.aafter_agent is not AgentMiddleware.aafter_agent
    ]
    # Collect middleware with wrap_model_call or awrap_model_call hooks
    # Include middleware with either implementation to ensure NotImplementedError is raised
    # when middleware doesn't support the execution path
    middleware_w_wrap_model_call = [
        m
        for m in middleware
        if m.__class__.wrap_model_call is not AgentMiddleware.wrap_model_call
        or m.__class__.awrap_model_call is not AgentMiddleware.awrap_model_call
    ]
    # Collect middleware with awrap_model_call or wrap_model_call hooks
    # Include middleware with either implementation to ensure NotImplementedError is raised
    # when middleware doesn't support the execution path
    middleware_w_awrap_model_call = [
        m
        for m in middleware
        if m.__class__.awrap_model_call is not AgentMiddleware.awrap_model_call
        or m.__class__.wrap_model_call is not AgentMiddleware.wrap_model_call
    ]

    # Compose wrap_model_call handlers into a single middleware stack (sync)
    wrap_model_call_handler = None
    if middleware_w_wrap_model_call:
        sync_handlers = [
            traceable(name=f"{m.name}.wrap_model_call", process_inputs=_scrub_inputs)(
                m.wrap_model_call
            )
            for m in middleware_w_wrap_model_call
        ]
        wrap_model_call_handler = _chain_model_call_handlers(sync_handlers)

    # Compose awrap_model_call handlers into a single middleware stack (async)
    awrap_model_call_handler = None
    if middleware_w_awrap_model_call:
        async_handlers = [
            traceable(name=f"{m.name}.awrap_model_call", process_inputs=_scrub_inputs)(
                m.awrap_model_call
            )
            for m in middleware_w_awrap_model_call
        ]
        awrap_model_call_handler = _chain_async_model_call_handlers(async_handlers)

    base_state = state_schema if state_schema is not None else AgentState
    # Build an ordered list: middleware schemas first (in registration order),
    # base_state last so it wins any field conflict.  This lets the caller's
    # explicit state_schema override middleware annotations — e.g. passing
    # a DeltaChannel-annotated schema wins over BinaryOperatorAggregate from
    # AgentState without requiring a post-compilation patch.
    state_schemas: list[type] = [*(m.state_schema for m in middleware), base_state]

    resolved_state_schema, input_schema, output_schema = _resolve_schemas(state_schemas)

    # 创建状态图。四个 schema 分工不同：
    #   - state_schema：图核心状态（messages 等，中间件声明的字段已合并进来）
    #   - input_schema / output_schema：对外暴露的入参 / 出参（可裁剪 state 字段）
    #   - context_schema：只读上下文（随每次调用传入，不随线程持久化）
    # create graph, add nodes
    graph: StateGraph[
        AgentState[ResponseT], ContextT, InputAgentState, OutputAgentState[ResponseT]
    ] = StateGraph(
        state_schema=resolved_state_schema,
        input_schema=input_schema,
        output_schema=output_schema,
        context_schema=context_schema,
    )

    def _handle_model_output(
        output: AIMessage, effective_response_format: ResponseFormat[Any] | None
    ) -> dict[str, Any]:
        """Handle model output including structured responses.

        Args:
            output: The AI message output from the model.
            effective_response_format: The actual strategy used (may differ from initial
                if auto-detected).
        """
        # Handle structured output with provider strategy
        if isinstance(effective_response_format, ProviderStrategy):
            if not output.tool_calls:
                provider_strategy_binding = ProviderStrategyBinding.from_schema_spec(
                    effective_response_format.schema_spec
                )
                try:
                    structured_response = provider_strategy_binding.parse(output)
                except Exception as exc:
                    schema_name = getattr(
                        effective_response_format.schema_spec.schema, "__name__", "response_format"
                    )
                    validation_error = StructuredOutputValidationError(schema_name, exc, output)
                    raise validation_error from exc
                else:
                    return {"messages": [output], "structured_response": structured_response}
            return {"messages": [output]}

        # Handle structured output with tool strategy
        if (
            isinstance(effective_response_format, ToolStrategy)
            and isinstance(output, AIMessage)
            and output.tool_calls
        ):
            structured_tool_calls = [
                tc for tc in output.tool_calls if tc["name"] in structured_output_tools
            ]

            if structured_tool_calls:
                exception: StructuredOutputError | None = None
                if len(structured_tool_calls) > 1:
                    # Handle multiple structured outputs error
                    tool_names = [tc["name"] for tc in structured_tool_calls]
                    exception = MultipleStructuredOutputsError(tool_names, output)
                    should_retry, error_message = _handle_structured_output_error(
                        exception, effective_response_format
                    )
                    if not should_retry:
                        raise exception

                    # Add error messages and retry
                    tool_messages = [
                        ToolMessage(
                            content=error_message,
                            tool_call_id=tc["id"],
                            name=tc["name"],
                        )
                        for tc in structured_tool_calls
                    ]
                    return {"messages": [output, *tool_messages]}

                # Handle single structured output
                tool_call = structured_tool_calls[0]
                try:
                    structured_tool_binding = structured_output_tools[tool_call["name"]]
                    structured_response = structured_tool_binding.parse(tool_call["args"])

                    tool_message_content = (
                        effective_response_format.tool_message_content
                        or f"Returning structured response: {structured_response}"
                    )

                    return {
                        "messages": [
                            output,
                            ToolMessage(
                                content=tool_message_content,
                                tool_call_id=tool_call["id"],
                                name=tool_call["name"],
                            ),
                        ],
                        "structured_response": structured_response,
                    }
                except Exception as exc:
                    exception = StructuredOutputValidationError(tool_call["name"], exc, output)
                    should_retry, error_message = _handle_structured_output_error(
                        exception, effective_response_format
                    )
                    if not should_retry:
                        raise exception from exc

                    return {
                        "messages": [
                            output,
                            ToolMessage(
                                content=error_message,
                                tool_call_id=tool_call["id"],
                                name=tool_call["name"],
                            ),
                        ],
                    }

        return {"messages": [output]}

    def _get_bound_model(
        request: ModelRequest[ContextT],
    ) -> tuple[Runnable[Any, Any], ResponseFormat[Any] | None]:
        """Get the model with appropriate tool bindings.

        Performs auto-detection of strategy if needed based on model capabilities.

        Args:
            request: The model request containing model, tools, and response format.

        Returns:
            Tuple of `(bound_model, effective_response_format)` where
            `effective_response_format` is the actual strategy used (may differ from
            initial if auto-detected).

        Raises:
            ValueError: If middleware returned unknown client-side tool names.
            ValueError: If `ToolStrategy` specifies tools not declared upfront.
        """
        # Validate ONLY client-side tools that need to exist in tool_node
        # Skip validation when wrap_tool_call is defined, as middleware may handle
        # dynamic tools that are added at runtime via wrap_model_call
        has_wrap_tool_call = wrap_tool_call_wrapper or awrap_tool_call_wrapper

        # Build map of available client-side tools from the ToolNode
        # (which has already converted callables)
        available_tools_by_name = {}
        if tool_node:
            available_tools_by_name = tool_node.tools_by_name.copy()

        # Check if any requested tools are unknown CLIENT-SIDE tools
        # Only validate if wrap_tool_call is NOT defined (no dynamic tool handling)
        if not has_wrap_tool_call:
            unknown_tool_names = []
            for t in request.tools:
                # Only validate BaseTool instances (skip built-in dict tools)
                if isinstance(t, dict):
                    continue
                if isinstance(t, BaseTool) and t.name not in available_tools_by_name:
                    unknown_tool_names.append(t.name)

            if unknown_tool_names:
                available_tool_names = sorted(available_tools_by_name.keys())
                msg = DYNAMIC_TOOL_ERROR_TEMPLATE.format(
                    unknown_tool_names=unknown_tool_names,
                    available_tool_names=available_tool_names,
                )
                raise ValueError(msg)

        # Normalize raw schemas to AutoStrategy
        # (handles middleware override with raw Pydantic classes)
        response_format: ResponseFormat[Any] | Any | None = request.response_format
        if response_format is not None and not isinstance(
            response_format, (AutoStrategy, ToolStrategy, ProviderStrategy)
        ):
            response_format = AutoStrategy(schema=response_format)

        # Determine effective response format (auto-detect if needed)
        effective_response_format: ResponseFormat[Any] | None
        if isinstance(response_format, AutoStrategy):
            # User provided raw schema via AutoStrategy - auto-detect best strategy based on model
            if _supports_provider_strategy(request.model, tools=request.tools):
                # Model supports provider strategy - use it
                effective_response_format = ProviderStrategy(schema=response_format.schema)
            elif response_format is initial_response_format and tool_strategy_for_setup is not None:
                # Model doesn't support provider strategy - use ToolStrategy
                # Reuse the strategy from setup if possible to preserve tool names
                effective_response_format = tool_strategy_for_setup
            else:
                effective_response_format = ToolStrategy(schema=response_format.schema)
        else:
            # User explicitly specified a strategy - preserve it
            effective_response_format = response_format

        # Build final tools list including structured output tools
        # request.tools now only contains BaseTool instances (converted from callables)
        # and dicts (built-ins)
        final_tools = list(request.tools)
        if isinstance(effective_response_format, ToolStrategy):
            # Add structured output tools to final tools list
            structured_tools = [info.tool for info in structured_output_tools.values()]
            final_tools.extend(structured_tools)

        # Bind model based on effective response format
        if isinstance(effective_response_format, ProviderStrategy):
            # (Backward compatibility) Use OpenAI format structured output
            # Redundantly set strict=True on tools for OpenAI-compatible models, as older
            # versions of langchain-openai do not auto-set it in bind_tools.
            kwargs = effective_response_format.to_model_kwargs()
            bind_kwargs: dict[str, Any] = {**kwargs, **request.model_settings}
            if _is_openai_compatible_model(request.model) and not getattr(
                request.model, "use_responses_api", False
            ):
                bind_kwargs["strict"] = True
            return (
                request.model.bind_tools(final_tools, **bind_kwargs),
                effective_response_format,
            )

        if isinstance(effective_response_format, ToolStrategy):
            # Current implementation requires that tools used for structured output
            # have to be declared upfront when creating the agent as part of the
            # response format. Middleware is allowed to change the response format
            # to a subset of the original structured tools when using ToolStrategy,
            # but not to add new structured tools that weren't declared upfront.
            # Compute output binding
            for tc in effective_response_format.schema_specs:
                if tc.name not in structured_output_tools:
                    msg = (
                        f"ToolStrategy specifies tool '{tc.name}' "
                        "which wasn't declared in the original "
                        "response format when creating the agent."
                    )
                    raise ValueError(msg)

            # Force tool use if we have structured output tools
            tool_choice = "any" if structured_output_tools else request.tool_choice
            return (
                request.model.bind_tools(
                    final_tools, tool_choice=tool_choice, **request.model_settings
                ),
                effective_response_format,
            )

        # No structured output - standard model binding
        if final_tools:
            return (
                request.model.bind_tools(
                    final_tools, tool_choice=request.tool_choice, **request.model_settings
                ),
                None,
            )
        return request.model.bind(**request.model_settings), None

    def _execute_model_sync(request: ModelRequest[ContextT]) -> ModelResponse:
        """Execute model and return response.

        This is the core model execution logic wrapped by `wrap_model_call` handlers.

        Raises any exceptions that occur during model invocation.
        """
        # Get the bound model (with auto-detection if needed)
        model_, effective_response_format = _get_bound_model(request)
        messages = request.messages
        if request.system_message:
            messages = [request.system_message, *messages]

        output = model_.invoke(messages)
        if name:
            output.name = name

        # Handle model output to get messages and structured_response
        handled_output = _handle_model_output(output, effective_response_format)
        messages_list = handled_output["messages"]
        structured_response = handled_output.get("structured_response")

        return ModelResponse(
            result=messages_list,
            structured_response=structured_response,
        )

    def model_node(state: AgentState[Any], runtime: Runtime[ContextT]) -> list[Command[Any]]:
        """Sync model request handler with sequential middleware processing."""
        request = ModelRequest(
            model=model,
            tools=default_tools,
            system_message=system_message,
            response_format=initial_response_format,
            messages=state["messages"],
            tool_choice=None,
            state=state,
            runtime=runtime,
        )

        if wrap_model_call_handler is None:
            model_response = _execute_model_sync(request)
            return _build_commands(model_response)

        result = wrap_model_call_handler(request, _execute_model_sync)
        return _build_commands(result.model_response, result.commands)

    async def _execute_model_async(request: ModelRequest[ContextT]) -> ModelResponse:
        """Execute model asynchronously and return response.

        This is the core async model execution logic wrapped by `wrap_model_call`
        handlers.

        Raises any exceptions that occur during model invocation.
        """
        # Get the bound model (with auto-detection if needed)
        model_, effective_response_format = _get_bound_model(request)
        messages = request.messages
        if request.system_message:
            messages = [request.system_message, *messages]

        output = await model_.ainvoke(messages)
        if name:
            output.name = name

        # Handle model output to get messages and structured_response
        handled_output = _handle_model_output(output, effective_response_format)
        messages_list = handled_output["messages"]
        structured_response = handled_output.get("structured_response")

        return ModelResponse(
            result=messages_list,
            structured_response=structured_response,
        )

    async def amodel_node(state: AgentState[Any], runtime: Runtime[ContextT]) -> list[Command[Any]]:
        """Async model request handler with sequential middleware processing."""
        request = ModelRequest(
            model=model,
            tools=default_tools,
            system_message=system_message,
            response_format=initial_response_format,
            messages=state["messages"],
            tool_choice=None,
            state=state,
            runtime=runtime,
        )

        if awrap_model_call_handler is None:
            model_response = await _execute_model_async(request)
            return _build_commands(model_response)

        result = await awrap_model_call_handler(request, _execute_model_async)
        return _build_commands(result.model_response, result.commands)

    # 添加核心节点。
    #   - "model" 节点：封装"选消息→拼 prompt→调模型"这一整段，RunnableCallable
    #     同时挂 sync/async 两个实现（trace=False 避免 tracing 刷屏）
    #   - "tools" 节点：仅当传了工具才存在 —— create_agent(tools=[]) 时图里没有它
    # Use sync or async based on model capabilities
    graph.add_node("model", RunnableCallable(model_node, amodel_node, trace=False))

    # Only add tools node if we have tools
    if tool_node is not None:
        graph.add_node("tools", tool_node)

    # 中间件挂点节点：before_agent / before_model / after_model / after_agent
    # 四类钩子，只给"真正被中间件 override 的挂点"建节点（没 override 就不存在），
    # 节点命名规则：{中间件名}.{挂点名}（如 "history_middleware.before_model"）
    # Add middleware nodes
    for m in middleware:
        if (
            m.__class__.before_agent is not AgentMiddleware.before_agent
            or m.__class__.abefore_agent is not AgentMiddleware.abefore_agent
        ):
            # Use RunnableCallable to support both sync and async
            # Pass None for sync if not overridden to avoid signature conflicts
            sync_before_agent = (
                m.before_agent
                if m.__class__.before_agent is not AgentMiddleware.before_agent
                else None
            )
            async_before_agent = (
                m.abefore_agent
                if m.__class__.abefore_agent is not AgentMiddleware.abefore_agent
                else None
            )
            before_agent_node = RunnableCallable(sync_before_agent, async_before_agent, trace=False)
            graph.add_node(
                f"{m.name}.before_agent", before_agent_node, input_schema=resolved_state_schema
            )

        if (
            m.__class__.before_model is not AgentMiddleware.before_model
            or m.__class__.abefore_model is not AgentMiddleware.abefore_model
        ):
            # Use RunnableCallable to support both sync and async
            # Pass None for sync if not overridden to avoid signature conflicts
            sync_before = (
                m.before_model
                if m.__class__.before_model is not AgentMiddleware.before_model
                else None
            )
            async_before = (
                m.abefore_model
                if m.__class__.abefore_model is not AgentMiddleware.abefore_model
                else None
            )
            before_node = RunnableCallable(sync_before, async_before, trace=False)
            graph.add_node(
                f"{m.name}.before_model", before_node, input_schema=resolved_state_schema
            )

        if (
            m.__class__.after_model is not AgentMiddleware.after_model
            or m.__class__.aafter_model is not AgentMiddleware.aafter_model
        ):
            # Use RunnableCallable to support both sync and async
            # Pass None for sync if not overridden to avoid signature conflicts
            sync_after = (
                m.after_model
                if m.__class__.after_model is not AgentMiddleware.after_model
                else None
            )
            async_after = (
                m.aafter_model
                if m.__class__.aafter_model is not AgentMiddleware.aafter_model
                else None
            )
            after_node = RunnableCallable(sync_after, async_after, trace=False)
            graph.add_node(f"{m.name}.after_model", after_node, input_schema=resolved_state_schema)

        if (
            m.__class__.after_agent is not AgentMiddleware.after_agent
            or m.__class__.aafter_agent is not AgentMiddleware.aafter_agent
        ):
            # Use RunnableCallable to support both sync and async
            # Pass None for sync if not overridden to avoid signature conflicts
            sync_after_agent = (
                m.after_agent
                if m.__class__.after_agent is not AgentMiddleware.after_agent
                else None
            )
            async_after_agent = (
                m.aafter_agent
                if m.__class__.aafter_agent is not AgentMiddleware.aafter_agent
                else None
            )
            after_agent_node = RunnableCallable(sync_after_agent, async_after_agent, trace=False)
            graph.add_node(
                f"{m.name}.after_agent", after_agent_node, input_schema=resolved_state_schema
            )

    # 四个"锚点"决定整张图的路由骨架，后面所有边都围绕它们连：
    #   entry_node      图入口，整个 agent 只跑一次（START 落点）
    #   loop_entry_node 循环回起点：tools 跑完回到这里，开始下一轮迭代
    #   loop_exit_node  每轮迭代的出口：条件边从这里出发（可执行多次）
    #   exit_node       最终出口：整张图只收一次尾（after_agent 链 → END）
    # Determine the entry node (runs once at start): before_agent -> before_model -> model
    # 决定整张图的入口节点 entry_node（START 的落点，整个 agent 只执行一次）：
    #   - 若有 before_agent 中间件 → 入口是第一个 before_agent 节点。before_agent 是
    #     "最外层"钩子，必须先经过它，后面的 before_agent 链 → before_model 链 → model
    #     才能按顺序触发。
    #   - 否则若有 before_model 中间件 → 入口是第一个 before_model 节点。没有外层
    #     钩子时，直接从每轮循环的起点 before_model 链进入。
    #   - 两者都没有 → 入口就是 "model" 节点。
    #   注意与 loop_entry_node 的区别：entry_node 只进一次，允许以 before_agent 开头；
    #   而 loop_entry_node 是"工具执行完循环回来的起点"，必须排除 before_agent
    #   （它只在 agent 开始时跑一次），所以只用 before_model 判断。
    if middleware_w_before_agent:
        entry_node = f"{middleware_w_before_agent[0].name}.before_agent"
    elif middleware_w_before_model:
        entry_node = f"{middleware_w_before_model[0].name}.before_model"
    else:
        entry_node = "model"

    # Determine the loop entry node (beginning of agent loop, excludes before_agent)
    # This is where tools will loop back to for the next iteration
    # 决定循环回起点 loop_entry_node（工具执行完后，下一轮迭代从这里重新进入）：
    #   - 必须排除 before_agent：它只在 agent 启动时跑一次，循环回来不能重跑，
    #     否则每轮迭代都会重复执行 before_agent 钩子。
    #   - 若有 before_model 中间件 → 从第一个 before_model 节点重新进入，
    #     保证每轮都先走 before_model 链，再进 model。
    #   - 没有 before_model → 直接从 "model" 节点进入下一轮。
    #   与 entry_node 的区别：entry_node 是 START 的落点、整个 agent 只进一次，
    #   允许以 before_agent 开头；loop_entry_node 是循环路径的入口、每轮都进，
    #   所以只用 middleware_w_before_model 判断。
    if middleware_w_before_model:
        loop_entry_node = f"{middleware_w_before_model[0].name}.before_model"
    else:
        loop_entry_node = "model"

    # Determine the loop exit node (end of each iteration, can run multiple times)
    # This is after_model or model, but NOT after_agent
    # 决定每轮迭代的出口 loop_exit_node（条件边从这里出发，每轮都可能走一次）：
    #   - 每轮循环的流程：before_model 链 → model → after_model 链 → 条件边路由。
    #     after_model 链跑完后，下一步去哪（tools / 回 model / 收尾）由条件边决定，
    #     所以"循环出口"就是 after_model 链的末尾。
    #   - 若有 after_model 中间件 → 出口是第一个 after_model 节点（每个中间件的
    #     after_model 节点按注册顺序依次相连，链的末端就是 loop_exit_node）。
    #   - 没有 after_model → 出口就是 "model" 节点。
    #   为什么不是 after_agent：after_agent 是 agent 结束时才收一次尾的钩子，
    #   不属于"每轮循环"的路径，不能作为循环出口，它是 exit_node 的职责。
    if middleware_w_after_model:
        loop_exit_node = f"{middleware_w_after_model[0].name}.after_model"
    else:
        loop_exit_node = "model"

    # Determine the exit node (runs once at end): after_agent or END
    # 决定最终出口 exit_node（整张图只收一次尾，从这里走完 after_agent 链后进 END）：
    #   - 若有 after_agent 中间件 → 出口是最后一个 after_agent 节点（用 [-1] 取链尾）。
    #     所有 after_agent 节点按注册顺序相连成链，链尾连 END，保证每个中间件的
    #     after_agent 钩子都在 agent 结束时按序执行、只执行一次。
    #   - 没有 after_agent → 出口直接是 END。
    #   - 什么时候走到 exit_node：每轮循环由条件边判断"是否该收尾"，一旦决定收尾，
    #     （若配置了 after_agent 链）先跑完 after_agent 链再进 END。
    #   与 loop_exit_node 的区别：loop_exit_node 是"每轮循环"的出口（after_model 链尾），
    #   exit_node 是"整个 agent"的出口（after_agent 链尾），前者每轮都可能经过，
    #   后者整个执行只经过一次。
    if middleware_w_after_agent:
        exit_node = f"{middleware_w_after_agent[-1].name}.after_agent"
    else:
        exit_node = END

    graph.add_edge(START, entry_node)
    # 条件边注册（仅当有 tools 节点时）：
    #   1) "tools" → tools_to_model：工具跑完，路由去 loop_entry_node（循环）还是 exit_node
    #   2) loop_exit_node → model_to_tools：模型响应后，路由去 tools / loop_entry_node / exit_node
    #   条件边返回 str = 去单一目标；返回 list[Send] = 并行 fan-out 多个目标
    # add conditional edges only if tools exist
    if tool_node is not None:
        # Only include exit_node in destinations if any tool has return_direct=True
        # or if there are structured output tools
        tools_to_model_destinations = [loop_entry_node]
        if (
            any(tool.return_direct for tool in tool_node.tools_by_name.values())
            or structured_output_tools
        ):
            tools_to_model_destinations.append(exit_node)

        graph.add_conditional_edges(
            "tools",
            RunnableCallable(
                _make_tools_to_model_edge(
                    tool_node=tool_node,
                    model_destination=loop_entry_node,
                    structured_output_tools=structured_output_tools,
                    end_destination=exit_node,
                ),
                trace=False,
            ),
            tools_to_model_destinations,
        )

        # base destinations are tools and exit_node
        # we add the loop_entry node to edge destinations if:
        # - there is an after model hook(s) -- allows jump_to to model
        #   potentially artificially injected tool messages, ex HITL
        # - there is a response format -- to allow for jumping to model to handle
        #   regenerating structured output tool calls
        model_to_tools_destinations = ["tools", exit_node]
        if response_format or loop_exit_node != "model":
            model_to_tools_destinations.append(loop_entry_node)

        graph.add_conditional_edges(
            loop_exit_node,
            RunnableCallable(
                _make_model_to_tools_edge(
                    model_destination=loop_entry_node,
                    structured_output_tools=structured_output_tools,
                    end_destination=exit_node,
                ),
                trace=False,
            ),
            model_to_tools_destinations,
        )
    elif len(structured_output_tools) > 0:
        graph.add_conditional_edges(
            loop_exit_node,
            RunnableCallable(
                _make_model_to_model_edge(
                    model_destination=loop_entry_node,
                    end_destination=exit_node,
                ),
                trace=False,
            ),
            [loop_entry_node, exit_node],
        )
    elif loop_exit_node == "model":
        # If no tools and no after_model, go directly to exit_node
        graph.add_edge(loop_exit_node, exit_node)
    # No tools but we have after_model - connect after_model to exit_node
    else:
        _add_middleware_edge(
            graph,
            name=f"{middleware_w_after_model[0].name}.after_model",
            default_destination=exit_node,
            model_destination=loop_entry_node,
            end_destination=exit_node,
            can_jump_to=_get_can_jump_to(middleware_w_after_model[0], "after_model"),
        )

    # Add before_agent middleware edges
    if middleware_w_before_agent:
        for m1, m2 in itertools.pairwise(middleware_w_before_agent):
            _add_middleware_edge(
                graph,
                name=f"{m1.name}.before_agent",
                default_destination=f"{m2.name}.before_agent",
                model_destination=loop_entry_node,
                end_destination=exit_node,
                can_jump_to=_get_can_jump_to(m1, "before_agent"),
            )
        # Connect last before_agent to loop_entry_node (before_model or model)
        _add_middleware_edge(
            graph,
            name=f"{middleware_w_before_agent[-1].name}.before_agent",
            default_destination=loop_entry_node,
            model_destination=loop_entry_node,
            end_destination=exit_node,
            can_jump_to=_get_can_jump_to(middleware_w_before_agent[-1], "before_agent"),
        )

    # Add before_model middleware edges
    if middleware_w_before_model:
        for m1, m2 in itertools.pairwise(middleware_w_before_model):
            _add_middleware_edge(
                graph,
                name=f"{m1.name}.before_model",
                default_destination=f"{m2.name}.before_model",
                model_destination=loop_entry_node,
                end_destination=exit_node,
                can_jump_to=_get_can_jump_to(m1, "before_model"),
            )
        # Go directly to model after the last before_model
        _add_middleware_edge(
            graph,
            name=f"{middleware_w_before_model[-1].name}.before_model",
            default_destination="model",
            model_destination=loop_entry_node,
            end_destination=exit_node,
            can_jump_to=_get_can_jump_to(middleware_w_before_model[-1], "before_model"),
        )

    # Add after_model middleware edges
    if middleware_w_after_model:
        graph.add_edge("model", f"{middleware_w_after_model[-1].name}.after_model")
        for idx in range(len(middleware_w_after_model) - 1, 0, -1):
            m1 = middleware_w_after_model[idx]
            m2 = middleware_w_after_model[idx - 1]
            _add_middleware_edge(
                graph,
                name=f"{m1.name}.after_model",
                default_destination=f"{m2.name}.after_model",
                model_destination=loop_entry_node,
                end_destination=exit_node,
                can_jump_to=_get_can_jump_to(m1, "after_model"),
            )
        # Note: Connection from after_model to after_agent/END is handled above
        # in the conditional edges section

    # Add after_agent middleware edges
    if middleware_w_after_agent:
        # Chain after_agent middleware (runs once at the very end, before END)
        for idx in range(len(middleware_w_after_agent) - 1, 0, -1):
            m1 = middleware_w_after_agent[idx]
            m2 = middleware_w_after_agent[idx - 1]
            _add_middleware_edge(
                graph,
                name=f"{m1.name}.after_agent",
                default_destination=f"{m2.name}.after_agent",
                model_destination=loop_entry_node,
                end_destination=exit_node,
                can_jump_to=_get_can_jump_to(m1, "after_agent"),
            )

        # Connect the last after_agent to END
        _add_middleware_edge(
            graph,
            name=f"{middleware_w_after_agent[0].name}.after_agent",
            default_destination=END,
            model_destination=loop_entry_node,
            end_destination=exit_node,
            can_jump_to=_get_can_jump_to(middleware_w_after_agent[0], "after_agent"),
        )

    # Set recursion limit to 9_999
    # https://github.com/langchain-ai/langgraph/issues/7313
    config: RunnableConfig = {"recursion_limit": 9_999}
    config["metadata"] = {"ls_integration": "langchain_create_agent"}
    if name:
        config["metadata"]["lc_agent_name"] = name

    middleware_transformers = [t for m in middleware for t in getattr(m, "transformers", ())]

    return graph.compile(
        checkpointer=checkpointer,
        store=store,
        interrupt_before=interrupt_before,
        interrupt_after=interrupt_after,
        debug=debug,
        name=name,
        cache=cache,
        transformers=[
            ToolCallTransformer,
            SubagentTransformer,
            *middleware_transformers,
            *(transformers or ()),
        ],
    ).with_config(config)


def _resolve_jump(
    jump_to: JumpTo | None,
    *,
    model_destination: str,
    end_destination: str,
) -> str | None:
    if jump_to == "model":
        return model_destination
    if jump_to == "end":
        return end_destination
    if jump_to == "tools":
        return "tools"
    return None


def _fetch_last_ai_and_tool_messages(
    messages: list[AnyMessage],
) -> tuple[AIMessage | None, list[ToolMessage]]:
    """Return the last AI message and any subsequent tool messages.

    Args:
        messages: List of messages to search through.

    Returns:
        A tuple of (last_ai_message, tool_messages). If no AIMessage is found,
        returns (None, []). Callers must handle the None case appropriately.
    """
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], AIMessage):
            last_ai_message = cast("AIMessage", messages[i])
            tool_messages = [m for m in messages[i + 1 :] if isinstance(m, ToolMessage)]
            return last_ai_message, tool_messages

    return None, []


# 条件边 ①：model → tools（ReAct 循环的心脏）
# 从 loop_exit_node 出发，6 步路由决定"继续调工具 / 回模型 / 收尾"：
#   1. jump_to 中间件显式指路（HITL 等）→ 优先
#   2. 没有 AIMessage（消息被清空）→ 收尾
#   3. 模型没发起任何 tool_call → 收尾（经典退出条件）
#   4. 有"未兑现"的 tool_call → 每个发一个 Send("tools", ...) 并行 fan-out
#   5. 已有 structured_response → 收尾
#   6. tool_calls 全"已兑现"（中间件/人工注入过 ToolMessage）→ 回 model 再跑一轮
def _make_model_to_tools_edge(
    *,
    model_destination: str,
    structured_output_tools: dict[str, OutputToolBinding[Any]],
    end_destination: str,
) -> Callable[[dict[str, Any]], str | list[Send] | None]:
    def model_to_tools(
        state: dict[str, Any],
    ) -> str | list[Send] | None:
        # 1. If there's an explicit jump_to in the state, use it
        if jump_to := state.get("jump_to"):
            return _resolve_jump(
                jump_to,
                model_destination=model_destination,
                end_destination=end_destination,
            )

        last_ai_message, tool_messages = _fetch_last_ai_and_tool_messages(state["messages"])

        # 2. if no AIMessage exists (e.g., messages were cleared), exit the loop
        if last_ai_message is None:
            return end_destination

        tool_message_ids = [m.tool_call_id for m in tool_messages]

        # 3. If the model hasn't called any tools, exit the loop
        # this is the classic exit condition for an agent loop
        if len(last_ai_message.tool_calls) == 0:
            return end_destination

        pending_tool_calls = [
            c
            for c in last_ai_message.tool_calls
            if c["id"] not in tool_message_ids and c["name"] not in structured_output_tools
        ]

        # 4. If there are pending tool calls, jump to the tool node.
        # The tool node hydrates ToolRuntime.state from channels via
        # CONFIG_KEY_READ at execution time, so we no longer inline the
        # full state into each Send (previously O(N^2) in TASKS writes).
        if pending_tool_calls:
            return [Send("tools", [tool_call]) for tool_call in pending_tool_calls]

        # 5. If there is a structured response, exit the loop
        if "structured_response" in state:
            return end_destination

        # 6. AIMessage has tool calls, but there are no pending tool calls which suggests
        # the injection of artificial tool messages. Jump to the model node
        return model_destination

    return model_to_tools


# 条件边 ①'：model → model（无 tools 但配了 response_format 时）
# 结构化输出生成失败（如模型没按 schema 出）时回 model 重试，成功则收尾
def _make_model_to_model_edge(
    *,
    model_destination: str,
    end_destination: str,
) -> Callable[[dict[str, Any]], str | list[Send] | None]:
    def model_to_model(
        state: dict[str, Any],
    ) -> str | list[Send] | None:
        # 1. Priority: Check for explicit jump_to directive from middleware
        if jump_to := state.get("jump_to"):
            return _resolve_jump(
                jump_to,
                model_destination=model_destination,
                end_destination=end_destination,
            )

        # 2. Exit condition: A structured response was generated
        if "structured_response" in state:
            return end_destination

        # 3. Default: Continue the loop, there may have been an issue with structured
        # output generation, so we need to retry
        return model_destination

    return model_to_model


# 条件边 ②：tools → model（循环回去，还是结束？）
#   1. 无 AIMessage → 回 model 重试
#   2. 本轮所有 client-side 工具都 return_direct=True → 工具结果即答案，直接 END
#      （只统计 client-side：provider 工具不在 tool_node.tools_by_name 里）
#   3. 执行了结构化输出工具 → END
#   4. 默认：工具跑完 → 回 loop_entry_node 开始下一轮循环
def _make_tools_to_model_edge(
    *,
    tool_node: ToolNode,
    model_destination: str,
    structured_output_tools: dict[str, OutputToolBinding[Any]],
    end_destination: str,
) -> Callable[[dict[str, Any]], str | None]:
    def tools_to_model(state: dict[str, Any]) -> str | None:
        last_ai_message, tool_messages = _fetch_last_ai_and_tool_messages(state["messages"])

        # 1. If no AIMessage exists (e.g., messages were cleared), route to model
        if last_ai_message is None:
            return model_destination

        # 2. Exit condition: All executed tools have return_direct=True
        # Filter to only client-side tools (provider tools are not in tool_node)
        client_side_tool_calls = [
            c for c in last_ai_message.tool_calls if c["name"] in tool_node.tools_by_name
        ]
        if client_side_tool_calls and all(
            tool_node.tools_by_name[c["name"]].return_direct for c in client_side_tool_calls
        ):
            return end_destination

        # 3. Exit condition: A structured output tool was executed
        if any(t.name in structured_output_tools for t in tool_messages):
            return end_destination

        # 4. Default: Continue the loop
        #    Tool execution completed successfully, route back to the model
        #    so it can process the tool results and decide the next action.
        return model_destination

    return tools_to_model


# 中间件节点之间的连线。
# 没配 can_jump_to → 普通 add_edge（默认走到下一个中间件/锚点）；
# 配了 can_jump_to → 条件边，中间件可通过 state["jump_to"] 把图"改道"到别处
# （这是 HITL / 人工介入实现跳转的机制）
def _add_middleware_edge(
    graph: StateGraph[
        AgentState[ResponseT], ContextT, InputAgentState, OutputAgentState[ResponseT]
    ],
    *,
    name: str,
    default_destination: str,
    model_destination: str,
    end_destination: str,
    can_jump_to: list[JumpTo] | None,
) -> None:
    """Add an edge to the graph for a middleware node.

    Args:
        graph: The graph to add the edge to.
        name: The name of the middleware node.
        default_destination: The default destination for the edge.
        model_destination: The destination for the edge to the model.
        end_destination: The destination for the edge to the end.
        can_jump_to: The conditionally jumpable destinations for the edge.
    """
    if can_jump_to:

        def jump_edge(state: dict[str, Any]) -> str:
            return (
                _resolve_jump(
                    state.get("jump_to"),
                    model_destination=model_destination,
                    end_destination=end_destination,
                )
                or default_destination
            )

        destinations = [default_destination]

        if "end" in can_jump_to:
            destinations.append(end_destination)
        if "tools" in can_jump_to:
            destinations.append("tools")
        if "model" in can_jump_to and name != model_destination:
            destinations.append(model_destination)

        graph.add_conditional_edges(name, RunnableCallable(jump_edge, trace=False), destinations)

    else:
        graph.add_edge(name, default_destination)


__all__ = [
    "create_agent",
]
