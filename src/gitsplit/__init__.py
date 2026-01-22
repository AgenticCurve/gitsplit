"""gitsplit - Semantic Git Splitter.

Transform messy git branches into clean, atomic PRs using AI.
"""

__version__ = "0.1.0"

from gitsplit.engine import SplitEngine, create_engine
from gitsplit.models import Session, Intent, ChangePlan

__all__ = [
    "__version__",
    "SplitEngine",
    "create_engine",
    "Session",
    "Intent",
    "ChangePlan",
]
