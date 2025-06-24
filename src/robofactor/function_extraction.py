import ast
import enum
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from returns.io import impure_safe
from returns.result import safe

# Type alias for function definition AST nodes to improve readability.
FunctionDefNode = ast.FunctionDef | ast.AsyncFunctionDef


class ParameterKind(enum.Enum):
    """Enumeration for the different kinds of function parameters."""

    POSITIONAL_ONLY = "positional_only"
    POSITIONAL_OR_KEYWORD = "positional_or_keyword"
    VAR_POSITIONAL = "var_positional"
    KEYWORD_ONLY = "keyword_only"
    VAR_KEYWORD = "var_keyword"


@dataclass(frozen=True)
class Parameter:
    """
    Represents a function parameter with its name, kind, and optional details.

    Attributes:
        name: The name of the parameter.
        kind: The kind of the parameter (e.g., positional-only).
        annotation: The type annotation as a string, if present.
        default: The default value as a string, if present.
    """

    name: str
    kind: ParameterKind
    annotation: str | None = None
    default: str | None = None


@dataclass(frozen=True)
class Decorator:
    """
    Represents a function decorator.

    Attributes:
        name: The name of the decorator.
        args: A tuple of arguments passed to the decorator, as strings.
    """

    name: str
    args: tuple[str, ...] = ()


@dataclass(frozen=True)
class FunctionContext:
    """Represents the context where a function is defined (base class)."""

    pass


@dataclass(frozen=True)
class ModuleContext(FunctionContext):
    """
    Represents a function defined at the module level.

    Attributes:
        module_name: The name of the module.
    """

    module_name: str


@dataclass(frozen=True)
class ClassContext(FunctionContext):
    """
    Represents a function defined within a class.

    Attributes:
        class_name: The name of the class.
        parent_context: The context in which the class is defined.
    """

    class_name: str
    parent_context: FunctionContext


@dataclass(frozen=True)
class NestedContext(FunctionContext):
    """
    Represents a function defined within another function.

    Attributes:
        parent_function: The name of the enclosing function.
        parent_context: The context of the enclosing function.
    """

    parent_function: str
    parent_context: FunctionContext


@dataclass(frozen=True)
class FunctionInfo:
    """
    Stores complete metadata for a Python function.

    Attributes:
        name: The name of the function.
        line_start: The starting line number of the function definition.
        line_end: The ending line number of the function definition.
        column_start: The starting column offset of the function definition.
        column_end: The ending column offset of the function definition.
        parameters: A tuple of Parameter objects for the function's signature.
        decorators: A tuple of Decorator objects applied to the function.
        is_async: A boolean indicating if the function is asynchronous.
        context: The context (Module, Class, or Nested) where the function is defined.
        docstring: The function's docstring, if present.
        return_annotation: The return type annotation as a string, if present.
    """

    name: str
    line_start: int
    line_end: int
    column_start: int
    column_end: int
    parameters: tuple[Parameter, ...]
    decorators: tuple[Decorator, ...]
    is_async: bool
    context: FunctionContext
    docstring: str | None = None
    return_annotation: str | None = None

    @classmethod
    def from_ast_node(
        cls: type["FunctionInfo"],
        node: FunctionDefNode,
        context: FunctionContext,
    ) -> "FunctionInfo":
        """
        Factory method to create a FunctionInfo instance from an AST node.

        This method encapsulates the logic of extracting all function metadata
        from the AST, making the FunctionInfo class self-sufficient in its
        construction from a source AST node.

        Args:
            node: The AST node (FunctionDef or AsyncFunctionDef).
            context: The context (Module, Class, or Nested) where the function is defined.

        Returns:
            A new instance of FunctionInfo.
        """
        return cls(
            name=node.name,
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
            column_start=node.col_offset,
            column_end=node.end_col_offset or node.col_offset,
            parameters=tuple(extract_parameters(node)),
            decorators=extract_decorators(node.decorator_list),
            is_async=isinstance(node, ast.AsyncFunctionDef),
            context=context,
            docstring=extract_docstring(node),
            return_annotation=(ast_node_to_source(node.returns) if node.returns else None),
        )


def ast_node_to_source(node: ast.AST) -> str:
    """
    Convert an AST node back to its source code representation.

    Args:
        node: The AST node to convert.

    Returns:
        The source code string for the node, or a repr for fallback.
    """
    try:
        return ast.unparse(node)
    except Exception:
        # Fallback for nodes that ast.unparse might not handle gracefully.
        # This ensures that even complex or unusual AST structures can be represented.
        return repr(node)


def extract_decorators(decorators: Sequence[ast.expr]) -> tuple[Decorator, ...]:
    """
    Extract decorator information from an AST decorator Sequence.

    Args:
        decorators: A Sequence of decorator nodes from an AST function definition.

    Returns:
        A tuple of Decorator objects.
    """

    def parse_decorator(dec: ast.expr) -> Decorator:
        """Parses a single decorator AST node."""
        match dec:
            case ast.Name(id=name):
                return Decorator(name=name)
            case ast.Call(func=func, args=args):
                return Decorator(
                    name=ast_node_to_source(func),
                    args=tuple(ast_node_to_source(arg) for arg in args),
                )
            case _:
                # Handles other cases like attribute access decorators (e.g., a.b.c)
                return Decorator(name=ast_node_to_source(dec))

    return tuple(parse_decorator(dec) for dec in decorators)


def _map_parameter_defaults(args: ast.arguments) -> dict[str, str]:
    """
    Maps parameter names to their default value's source string.

    This helper centralizes the logic for extracting default values for both
    positional and keyword-only arguments from an `ast.arguments` node.

    Args:
        args: The `ast.arguments` node from a function definition.

    Returns:
        A dictionary mapping parameter names to their unparsed default values.
    """
    defaults_map = {}

    # Positional and positional-or-keyword defaults
    all_positional_args = args.posonlyargs + args.args
    num_defaults = len(args.defaults)
    if num_defaults > 0:
        # Defaults correspond to the last `num_defaults` positional arguments.
        args_with_defaults = all_positional_args[-num_defaults:]
        for arg, default_node in zip(args_with_defaults, args.defaults, strict=False):
            defaults_map[arg.arg] = ast_node_to_source(default_node)

    # Keyword-only defaults
    kw_defaults = {
        arg.arg: ast_node_to_source(default_node)
        for arg, default_node in zip(args.kwonlyargs, args.kw_defaults, strict=False)
        if default_node is not None
    }
    defaults_map.update(kw_defaults)

    return defaults_map


def extract_parameters(func_node: FunctionDefNode) -> Iterator[Parameter]:
    """
    Generate parameter information from a function's AST node.

    This function processes all parameter kinds in their correct order by
    delegating the creation of Parameter objects to an internal helper,
    which reduces code duplication.

    Args:
        func_node: The function definition node from the AST.

    Yields:
        Parameter objects representing the function's signature.
    """

    def _create_parameter(arg: ast.arg, kind: ParameterKind, defaults: dict[str, str]) -> Parameter:
        """Internal helper to create a Parameter instance."""
        return Parameter(
            name=arg.arg,
            annotation=ast_node_to_source(arg.annotation) if arg.annotation else None,
            default=defaults.get(arg.arg),
            kind=kind,
        )

    args = func_node.args
    defaults = _map_parameter_defaults(args)

    # Positional-only parameters
    for arg in args.posonlyargs:
        yield _create_parameter(arg, ParameterKind.POSITIONAL_ONLY, defaults)

    # Positional or keyword parameters
    for arg in args.args:
        yield _create_parameter(arg, ParameterKind.POSITIONAL_OR_KEYWORD, defaults)

    # Var-positional parameter (*args)
    if args.vararg:
        yield Parameter(
            name=args.vararg.arg,
            annotation=(
                ast_node_to_source(args.vararg.annotation) if args.vararg.annotation else None
            ),
            kind=ParameterKind.VAR_POSITIONAL,
        )

    # Keyword-only parameters
    for arg in args.kwonlyargs:
        yield _create_parameter(arg, ParameterKind.KEYWORD_ONLY, defaults)

    # Var-keyword parameter (**kwargs)
    if args.kwarg:
        yield Parameter(
            name=args.kwarg.arg,
            annotation=(
                ast_node_to_source(args.kwarg.annotation) if args.kwarg.annotation else None
            ),
            kind=ParameterKind.VAR_KEYWORD,
        )


def extract_docstring(func_node: FunctionDefNode) -> str | None:
    """
    Extract the docstring from a function node using ast.get_docstring.

    Args:
        func_node: The function definition node from the AST.

    Returns:
        The docstring string, or None if not found.
    """
    return ast.get_docstring(func_node, clean=False)


class _FunctionVisitor:
    """
    An internal AST visitor to find and yield all function definitions.

    This class traverses an AST, keeping track of the definition context
    (module, class, or nested function) and yields a `FunctionInfo`
    object for each function it encounters.
    """

    def visit(self, node: ast.AST, context: FunctionContext) -> Iterator[FunctionInfo]:
        """
        Recursively visit AST nodes and yield function information.

        This method uses structural pattern matching to dispatch to more
        specific handlers based on the node type.

        Args:
            node: The current AST node to visit.
            context: The current context (Module, Class, or Nested).

        Yields:
            FunctionInfo objects for each function found.
        """
        match node:
            case ast.FunctionDef() | ast.AsyncFunctionDef():
                yield from self._visit_function_def(node, context)
            case ast.ClassDef():
                yield from self._visit_class_def(node, context)
            case _:
                # Continue traversal for other node types.
                for child in ast.iter_child_nodes(node):
                    yield from self.visit(child, context)

    def _visit_function_def(
        self, node: FunctionDefNode, context: FunctionContext
    ) -> Iterator[FunctionInfo]:
        """Handle FunctionDef and AsyncFunctionDef nodes."""
        yield FunctionInfo.from_ast_node(node, context)

        # Recursively visit the body of the function for nested functions.
        nested_context = NestedContext(parent_function=node.name, parent_context=context)
        for child in node.body:
            yield from self.visit(child, nested_context)

    def _visit_class_def(
        self, node: ast.ClassDef, context: FunctionContext
    ) -> Iterator[FunctionInfo]:
        """Handle ClassDef nodes."""
        class_context = ClassContext(class_name=node.name, parent_context=context)
        # Recursively visit the body of the class for methods and nested classes.
        for child in node.body:
            yield from self.visit(child, class_context)


def _parse_ast_and_find_functions(tree: ast.Module, module_name: str) -> Iterator[FunctionInfo]:
    """
    Internal helper to process a parsed AST and yield function info.

    Args:
        tree: The parsed AST module.
        module_name: The name of the module being parsed.

    Yields:
        FunctionInfo objects for all functions in the AST.
    """
    module_context = ModuleContext(module_name=module_name)
    visitor = _FunctionVisitor()
    yield from visitor.visit(tree, module_context)


@impure_safe
def parse_python_file(file_path: Path) -> Iterator[FunctionInfo]:
    """
    Parse a Python file and stream all function definitions.

    This function is decorated with `@impure_safe` to handle `IO` and parsing
    errors (e.g., `FileNotFoundError`, `SyntaxError`), returning an `IOResult`.

    Args:
        file_path: The path to the Python file.

    Yields:
        An iterator of `FunctionInfo` objects for all functions in the file.
        The yielding process is wrapped in an `IOResult` container.
    """
    with open(file_path, encoding="utf-8") as f:
        source_code = f.read()
    tree = ast.parse(source_code, filename=str(file_path))
    yield from _parse_ast_and_find_functions(tree, file_path.stem)


@safe
def parse_python_source(source_code: str, module_name: str = "<string>") -> Iterator[FunctionInfo]:
    """
    Parse a Python source string and stream all function definitions.

    This function is decorated with `@safe` to handle `SyntaxError`,
    returning a `Result` container.

    Args:
        source_code: The Python code to parse.
        module_name: The name to associate with the module.

    Yields:
        An iterator of `FunctionInfo` objects for all functions in the code.
        The yielding process is wrapped in a `Result` container.
    """
    tree = ast.parse(source_code, filename=module_name)
    yield from _parse_ast_and_find_functions(tree, module_name)


def filter_by_context(
    context_type: type[FunctionContext],
    functions: Iterator[FunctionInfo],
) -> Iterator[FunctionInfo]:
    """
    Filter functions by their context type (e.g., ClassContext).

    Args:
        context_type: The context class to filter by (e.g., ClassContext).
        functions: An iterator of FunctionInfo objects to filter.

    Yields:
        The functions that match the context type.
    """
    yield from (f for f in functions if isinstance(f.context, context_type))


def filter_by_decorator(
    decorator_name: str,
    functions: Iterator[FunctionInfo],
) -> Iterator[FunctionInfo]:
    """
    Filter functions that have a specific decorator.

    Args:
        decorator_name: The name of the decorator to search for.
        functions: An iterator of FunctionInfo objects to filter.

    Yields:
        The functions that have the specified decorator.
    """
    yield from (f for f in functions if any(d.name == decorator_name for d in f.decorators))


def get_function_names(functions: Iterator[FunctionInfo]) -> Iterator[str]:
    """
    Extract just the names from a sequence of functions.

    Args:
        functions: An iterator of FunctionInfo objects.

    Yields:
        The function name strings.
    """
    yield from (f.name for f in functions)


def get_async_functions(functions: Iterator[FunctionInfo]) -> Iterator[FunctionInfo]:
    """
    Filter for and return only the asynchronous functions.

    Args:
        functions: An iterator of FunctionInfo objects.

    Yields:
        The async functions.
    """
    yield from (f for f in functions if f.is_async)


def get_methods(functions: Iterator[FunctionInfo]) -> Iterator[FunctionInfo]:
    """
    Get all functions defined within a class context (methods).

    Args:
        functions: An iterator of FunctionInfo objects.

    Yields:
        The functions defined in a ClassContext.
    """
    yield from filter_by_context(ClassContext, functions)


def format_function_signature(func: FunctionInfo) -> str:
    """
    Format a function's signature into a human-readable string.

    This function iterates over the parameters once, building the signature
    efficiently and correctly handling all parameter kinds.

    Args:
        func: The FunctionInfo object to format.

    Returns:
        A string representing the function's signature.
    """

    def format_param(p: Parameter) -> str:
        """Formats a single parameter object into a string."""
        res = p.name
        if p.annotation:
            res += f": {p.annotation}"
        if p.default is not None:
            res += f" = {p.default}"
        return res

    param_parts: Sequence[str] = []
    pos_only_ended = False
    var_pos_added = False

    for p in func.parameters:
        match p.kind:
            case ParameterKind.POSITIONAL_ONLY:
                param_parts.append(format_param(p))
            case ParameterKind.POSITIONAL_OR_KEYWORD:
                # Add '/' separator if positional-only args exist and this is the first positional-or-keyword arg.
                if not pos_only_ended and any(
                    param.kind == ParameterKind.POSITIONAL_ONLY for param in func.parameters
                ):
                    param_parts.append("/")
                    pos_only_ended = True
                param_parts.append(format_param(p))
            case ParameterKind.VAR_POSITIONAL:
                # Add '/' separator if positional-only args exist and this is the first var-positional arg.
                if not pos_only_ended and any(
                    param.kind == ParameterKind.POSITIONAL_ONLY for param in func.parameters
                ):
                    param_parts.append("/")
                    pos_only_ended = True
                param_parts.append(f"*{p.name}")
                var_pos_added = True
            case ParameterKind.KEYWORD_ONLY:
                # Add '/' separator if positional-only args exist and this is the first keyword-only arg.
                if not pos_only_ended and any(
                    param.kind == ParameterKind.POSITIONAL_ONLY for param in func.parameters
                ):
                    param_parts.append("/")
                    pos_only_ended = True
                # Add '*' separator if no var-positional arg was present and this is the first keyword-only arg.
                if not var_pos_added:
                    param_parts.append("*")
                    var_pos_added = True
                param_parts.append(format_param(p))
            case ParameterKind.VAR_KEYWORD:
                # Add '*' separator if no var-positional arg was present and this is the first var-keyword arg.
                if not var_pos_added:
                    param_parts.append("*")
                param_parts.append(f"**{p.name}")

    # Ensure '/' is added if there are positional-only args but no positional-or-keyword or var-positional args.
    if not pos_only_ended and any(
        param.kind == ParameterKind.POSITIONAL_ONLY for param in func.parameters
    ):
        param_parts.append("/")

    params_str = ", ".join(param_parts)
    async_prefix = "async " if func.is_async else ""
    return_suffix = f" -> {func.return_annotation}" if func.return_annotation else ""
    return f"{async_prefix}def {func.name}({params_str}){return_suffix}"
