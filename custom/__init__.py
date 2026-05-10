"""User-owned extensions to the occhio visualization toolkit.

Usage from a notebook in ``experiments/``::

    import sys
    sys.path.insert(0, "..")
    from custom.plotting import plot_bars

    plot_bars(grid)
"""

from .plotting import plot_bars

__all__ = ["plot_bars"]
