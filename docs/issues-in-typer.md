# Typer 0.16.0 Quick Fix

**Problem:** `TypeError: Name 'X' defined twice` when using `Annotated` with `typer.Option`.  
**Fix:** Never put the default inside `Option()` when you also give the parameter a default.

```python
# ❌ Breaks
mode: Annotated[DiffMode, typer.Option(DiffMode.both, "-m", "--mode")] = DiffMode.both

# ✅ Works
mode: Annotated[DiffMode, typer.Option("-m", "--mode")] = DiffMode.both
```

That’s it—remove the first positional argument from `Option()` and rely on the parameter default.
