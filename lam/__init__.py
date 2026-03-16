"""
Compatibility shim — imports from said_lam.

The canonical import is:
    from said_lam import LAM

This module re-exports for backwards compatibility only.
"""

from said_lam import *  # noqa: F401,F403
from said_lam import LAM, __version__  # noqa: F401 — explicit re-exports
