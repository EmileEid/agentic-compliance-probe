"""Minimal matplotlib.pyplot stub for editor resolution.

This stub allows the notebook to import `matplotlib.pyplot` without
installing the full library. It does not produce real plots. Install
`matplotlib` in the venv to enable plotting functionality.
"""

def subplots(figsize=None):
    class Ax:
        def plot(self, *args, **kwargs):
            return None
        def set_title(self, *args, **kwargs):
            return None
        def set_ylabel(self, *args, **kwargs):
            return None
        def set_xlabel(self, *args, **kwargs):
            return None
        def legend(self, *args, **kwargs):
            return None
        def set_ylim(self, *args, **kwargs):
            return None

    return (None, Ax())

def tight_layout():
    return None

def show():
    raise RuntimeError(
        "matplotlib is not installed in the environment. "
        "Install with `pip install matplotlib` to render plots."
    )
