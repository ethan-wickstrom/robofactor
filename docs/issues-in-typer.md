# Typer 0.16.0 "Name defined twice" Issue with Annotated and Option

## Executive Summary

In Typer 0.16.0 with Python 3.13, using `Annotated` with `Option()` can trigger a `TypeError: Name 'X' defined twice` error when the default value is specified as the first argument to `Option()`. This occurs due to how Typer and Click process parameter declarations internally.

## The Problem

### Failing Code

```python
@app.command()
def main(
    mode: Annotated[
        DiffMode,
        typer.Option(
            DiffMode.both,  # ❌ Default value as first argument
            "-m",
            "--mode",
            help="Which mode to use",
        ),
    ] = DiffMode.both,
):
    pass
```

**Error:** `TypeError: Name 'mode' defined twice`

### Working Code

```python
@app.command()
def main(
    mode: Annotated[
        DiffMode,
        typer.Option(
            "-m",
            "--mode",
            help="Which mode to use",
        ),
    ] = DiffMode.both,  # ✅ Default only as parameter default
):
    pass
```

## Root Cause Analysis

### 1. Option() Function Signature

The `Option()` function in `typer/params.py` has this signature:

```python
def Option(
    default: Optional[Any] = ...,
    *param_decls: str,
    # ... other parameters
) -> OptionInfo
```

When you call:

- `Option(DiffMode.both, "-m", "--mode")` → `default=DiffMode.both`, `param_decls=('-m', '--mode')`
- `Option("-m", "--mode")` → `default='-m'`, `param_decls=('--mode',)`

### 2. Typer's Parameter Processing

In `typer/main.py` at line 895, when processing parameters:

```python
param_decls = [param.name]  # Adds the parameter name first
if parameter_info.param_decls:
    param_decls.extend(parameter_info.param_decls)
```

For a parameter named `mode`:

- With `Option(DiffMode.both, "-m", "--mode")`: `param_decls = ['mode', '-m', '--mode']`
- With `Option("-m", "--mode")`: `param_decls = ['mode', '--mode']`

### 3. Click's \_parse_decls Method

Click's `_parse_decls` method in `click/core.py` (line 2683) processes these declarations:

```python
def _parse_decls(self, decls, expose_value):
    name = None
    for decl in decls:
        if decl.isidentifier():
            if name is not None:
                raise TypeError(f"Name '{name}' defined twice")
            name = decl
```

The method:

1. Iterates through each declaration
2. If it's an identifier (passes `.isidentifier()`), it sets it as the parameter name
3. If a name was already set, it raises the "defined twice" error

### 4. The Conflict

When `param_decls = ['mode', '-m', '--mode']`:

1. `'mode'` is processed → `name = 'mode'` (it's an identifier)
2. `'-m'` is processed → treated as option flag (not an identifier)
3. `'--mode'` is processed → Click extracts `'mode'` from it internally
4. Since `'mode'` was already set as the name, the error is raised

## Full Traceability

### Call Stack Flow

1. **User code**: Defines function with `Annotated[Type, Option(...)]`
2. **typer/main.py:341**: `Typer.__call__()` is invoked
3. **typer/main.py:377**: `get_command()` processes the command
4. **typer/main.py:586**: `get_command_from_info()` extracts command info
5. **typer/main.py:562**: `get_params_convertors_ctx_param_name_from_function()` processes parameters
6. **typer/main.py:901**: `get_click_param()` creates Click parameters
   - Line 895: Prepends parameter name to `param_decls`
7. **typer/core.py:444**: `TyperOption.__init__()` is called
8. **click/core.py:2558**: `click.Option.__init__()` is called
9. **click/core.py:2098**: `click.Parameter.__init__()` is called
10. **click/core.py:2694**: `_parse_decls()` raises the error

### Environment Details

- **Python**: 3.13
- **Typer**: 0.16.0
- **Click**: (bundled with Typer)
- **Platform**: darwin (macOS)

## Solution

### Best Practice

When using `Annotated` with `Option()`, never specify the default value as the first argument to `Option()`:

```python
# ❌ WRONG - Causes "Name defined twice" error
mode: Annotated[Type, typer.Option(default_value, "-m", "--mode")] = default_value

# ✅ CORRECT - Default only as parameter default
mode: Annotated[Type, typer.Option("-m", "--mode")] = default_value
```

### Why This Works

- Without a default in `Option()`, the first argument becomes a param_decl
- This prevents the parameter name from appearing twice in the declarations
- The default value is properly handled through Python's parameter default mechanism

## Alternative Patterns

### 1. Direct Option Usage (No Annotated)

```python
def main(
    mode: DiffMode = typer.Option(
        DiffMode.both,  # Can specify default here
        "-m",
        "--mode",
        help="Which mode to use",
    ),
):
    pass
```

### 2. Argument Instead of Option

```python
def main(
    mode: Annotated[
        DiffMode,
        typer.Argument(help="Which mode to use"),
    ] = DiffMode.both,
):
    pass
```

## Impact and Considerations

### When This Issue Occurs

- Using Typer 0.16.0
- Using `Annotated` type hints
- Specifying default value as first argument to `Option()`
- The parameter name matches (after transformation) an option name

### When This Issue Does NOT Occur

- Using direct assignment pattern (no `Annotated`)
- Not specifying default in `Option()` constructor
- Using `Argument()` instead of `Option()`

## Recommendations

1. **For New Code**: Always omit the default value from `Option()` when using `Annotated`
2. **For Migration**: Remove default values from `Option()` calls in `Annotated` contexts
3. **For Teams**: Establish coding standards that enforce this pattern
4. **For Tooling**: Consider linters or pre-commit hooks to catch this pattern

## Related Issues

This issue is specific to the interaction between:

- Typer's parameter processing
- Click's declaration parsing
- Python's `Annotated` type hints
- The overlapping namespace between parameter names and option names

The error message "Name 'X' defined twice" is misleading as it doesn't clearly indicate the source of the duplication.
