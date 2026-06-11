"""Minimal pandas stub for editor resolution.

This stub exists only to satisfy static analysis (Pylance) in the
development workspace. It intentionally does NOT implement the real
`pandas` functionality. Install `pandas` in the project's virtualenv
to run analyses.
"""

try:
    # If the real pandas is installed, re-export it for normal behavior.
    import importlib
    _real_pandas = importlib.import_module("pandas")
    from pandas import *  # type: ignore
except Exception:  # pragma: no cover - editor/runtime shim
    class DataFrame:  # minimal placeholder type for editor
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "pandas is not installed in the environment. "
                "Install with `pip install pandas` to run the notebook."
            )

    def read_csv(*args, **kwargs):
        raise RuntimeError(
            "pandas is not installed in the environment. "
            "Install with `pip install pandas` to run the notebook."
        )

    __all__ = ["DataFrame", "read_csv"]
