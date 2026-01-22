"""Data models for gitsplit."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class SessionPhase(str, Enum):
    """Current phase of the split session."""

    INIT = "init"
    DISCOVERY = "discovery"
    PLANNING = "planning"
    EXECUTION = "execution"
    VERIFICATION = "verification"
    COMPLETE = "complete"
    FAILED = "failed"


class ResolutionStrategy(str, Enum):
    """Strategy for resolving multi-intent file conflicts."""

    STACK = "stack"  # Intent B depends on Intent A
    MERGE = "merge"  # Combine intents into single PR
    DUPLICATE = "duplicate"  # Both intents get the shared change
    MANUAL = "manual"  # User specifies via Escape Hatch


@dataclass
class LineRange:
    """A range of lines in a file."""

    start: int
    end: int

    def overlaps(self, other: "LineRange") -> bool:
        """Check if this range overlaps with another."""
        return not (self.end < other.start or other.end < self.start)

    def contains(self, line: int) -> bool:
        """Check if a line is within this range."""
        return self.start <= line <= self.end


@dataclass
class FileChange:
    """A change to a file attributed to an intent."""

    path: str
    line_ranges: list[LineRange] = field(default_factory=list)
    is_entire_file: bool = False
    additions: int = 0
    deletions: int = 0

    @property
    def total_changes(self) -> int:
        return self.additions + self.deletions


@dataclass
class Intent:
    """A logical unit of work identified in the changes."""

    id: str
    name: str
    description: str
    files: list[FileChange] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)  # IDs of intents this depends on
    is_confirmed: bool = False
    branch_name: str | None = None
    pr_number: int | None = None
    pr_url: str | None = None

    @property
    def total_additions(self) -> int:
        return sum(f.additions for f in self.files)

    @property
    def total_deletions(self) -> int:
        return sum(f.deletions for f in self.files)


@dataclass
class MultiIntentConflict:
    """A conflict where a file has changes for multiple intents."""

    file_path: str
    intent_ids: list[str]
    overlapping_ranges: list[tuple[str, str, LineRange]]  # (intent1, intent2, overlap)
    suggested_strategy: ResolutionStrategy
    resolved: bool = False
    chosen_strategy: ResolutionStrategy | None = None


@dataclass
class ChangePlan:
    """The complete plan for splitting changes."""

    intents: list[Intent]
    conflicts: list[MultiIntentConflict] = field(default_factory=list)
    execution_order: list[str] = field(default_factory=list)  # Intent IDs in order
    is_validated: bool = False


@dataclass
class VerificationResult:
    """Result of hash verification."""

    passed: bool
    original_hash: str
    final_hash: str
    differences: list[dict[str, Any]] = field(default_factory=list)

    @property
    def diagnosis(self) -> str:
        """Get a human-readable diagnosis of the verification failure."""
        if self.passed:
            return "Verification passed - hashes match"

        if not self.differences:
            return f"Hash mismatch: expected {self.original_hash}, got {self.final_hash}"

        lines = ["Hash mismatch detected:"]
        for diff in self.differences:
            lines.append(f"  {diff.get('file', 'unknown')}: {diff.get('description', 'unknown difference')}")
        return "\n".join(lines)


@dataclass
class BacktrackInfo:
    """Information about a backtrack decision."""

    from_phase: SessionPhase
    to_phase: SessionPhase
    reason: str
    attempt: int
    preserved_intents: list[str] = field(default_factory=list)
    preserved_files: list[str] = field(default_factory=list)


@dataclass
class Session:
    """Persistent session state for split operations."""

    id: str
    branch: str
    base_branch: str
    phase: SessionPhase = SessionPhase.INIT
    original_tree_hash: str = ""

    # Phase 1 output
    discovered_intents: list[Intent] = field(default_factory=list)
    confirmed_intents: list[Intent] = field(default_factory=list)

    # Phase 2 output
    change_plan: ChangePlan | None = None

    # Phase 3 tracking
    created_branches: list[str] = field(default_factory=list)
    created_prs: list[dict[str, Any]] = field(default_factory=list)

    # Backtracking
    backtracks: list[BacktrackInfo] = field(default_factory=list)
    current_attempt: int = 1
    max_attempts: int = 5

    # Cost tracking
    total_tokens_used: int = 0
    total_cost: float = 0.0
    max_cost: float | None = None

    # Options
    auto_mode: bool = False
    auto_hint: str = ""
    babysit_mode: bool = False
    dry_run: bool = False
    verbose: bool = False
    no_verify_build: bool = False
    no_pr: bool = False

    def get_session_path(self) -> Path:
        """Get the path to the session file."""
        return Path.home() / ".gitsplit" / "sessions" / f"{self.id}.json"
