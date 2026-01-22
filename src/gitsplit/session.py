"""Session persistence for gitsplit."""

import json
import uuid
from dataclasses import asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any

from gitsplit.models import (
    Session,
    SessionPhase,
    Intent,
    FileChange,
    LineRange,
    ChangePlan,
    MultiIntentConflict,
    ResolutionStrategy,
    BacktrackInfo,
)


SESSIONS_DIR = Path.home() / ".gitsplit" / "sessions"


def ensure_sessions_dir() -> None:
    """Ensure the sessions directory exists."""
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


def generate_session_id() -> str:
    """Generate a unique session ID."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    short_uuid = str(uuid.uuid4())[:8]
    return f"{timestamp}-{short_uuid}"


def _serialize_intent(intent: Intent) -> dict[str, Any]:
    """Serialize an Intent to a dict."""
    return {
        "id": intent.id,
        "name": intent.name,
        "description": intent.description,
        "files": [
            {
                "path": f.path,
                "line_ranges": [[lr.start, lr.end] for lr in f.line_ranges],
                "is_entire_file": f.is_entire_file,
                "additions": f.additions,
                "deletions": f.deletions,
            }
            for f in intent.files
        ],
        "dependencies": intent.dependencies,
        "is_confirmed": intent.is_confirmed,
        "branch_name": intent.branch_name,
        "pr_number": intent.pr_number,
        "pr_url": intent.pr_url,
    }


def _deserialize_intent(data: dict[str, Any]) -> Intent:
    """Deserialize an Intent from a dict."""
    files = []
    for f in data.get("files", []):
        files.append(
            FileChange(
                path=f["path"],
                line_ranges=[LineRange(lr[0], lr[1]) for lr in f.get("line_ranges", [])],
                is_entire_file=f.get("is_entire_file", False),
                additions=f.get("additions", 0),
                deletions=f.get("deletions", 0),
            )
        )

    return Intent(
        id=data["id"],
        name=data["name"],
        description=data.get("description", ""),
        files=files,
        dependencies=data.get("dependencies", []),
        is_confirmed=data.get("is_confirmed", False),
        branch_name=data.get("branch_name"),
        pr_number=data.get("pr_number"),
        pr_url=data.get("pr_url"),
    )


def _serialize_change_plan(plan: ChangePlan | None) -> dict[str, Any] | None:
    """Serialize a ChangePlan to a dict."""
    if plan is None:
        return None

    return {
        "intents": [_serialize_intent(i) for i in plan.intents],
        "conflicts": [
            {
                "file_path": c.file_path,
                "intent_ids": c.intent_ids,
                "overlapping_ranges": [
                    [r[0], r[1], [r[2].start, r[2].end]]
                    for r in c.overlapping_ranges
                ],
                "suggested_strategy": c.suggested_strategy.value,
                "resolved": c.resolved,
                "chosen_strategy": c.chosen_strategy.value if c.chosen_strategy else None,
            }
            for c in plan.conflicts
        ],
        "execution_order": plan.execution_order,
        "is_validated": plan.is_validated,
    }


def _deserialize_change_plan(data: dict[str, Any] | None) -> ChangePlan | None:
    """Deserialize a ChangePlan from a dict."""
    if data is None:
        return None

    intents = [_deserialize_intent(i) for i in data.get("intents", [])]

    conflicts = []
    for c in data.get("conflicts", []):
        overlapping_ranges = []
        for r in c.get("overlapping_ranges", []):
            overlapping_ranges.append((r[0], r[1], LineRange(r[2][0], r[2][1])))

        conflicts.append(
            MultiIntentConflict(
                file_path=c["file_path"],
                intent_ids=c["intent_ids"],
                overlapping_ranges=overlapping_ranges,
                suggested_strategy=ResolutionStrategy(c["suggested_strategy"]),
                resolved=c.get("resolved", False),
                chosen_strategy=(
                    ResolutionStrategy(c["chosen_strategy"])
                    if c.get("chosen_strategy")
                    else None
                ),
            )
        )

    return ChangePlan(
        intents=intents,
        conflicts=conflicts,
        execution_order=data.get("execution_order", []),
        is_validated=data.get("is_validated", False),
    )


def _serialize_backtrack(backtrack: BacktrackInfo) -> dict[str, Any]:
    """Serialize a BacktrackInfo to a dict."""
    return {
        "from_phase": backtrack.from_phase.value,
        "to_phase": backtrack.to_phase.value,
        "reason": backtrack.reason,
        "attempt": backtrack.attempt,
        "preserved_intents": backtrack.preserved_intents,
        "preserved_files": backtrack.preserved_files,
    }


def _deserialize_backtrack(data: dict[str, Any]) -> BacktrackInfo:
    """Deserialize a BacktrackInfo from a dict."""
    return BacktrackInfo(
        from_phase=SessionPhase(data["from_phase"]),
        to_phase=SessionPhase(data["to_phase"]),
        reason=data["reason"],
        attempt=data["attempt"],
        preserved_intents=data.get("preserved_intents", []),
        preserved_files=data.get("preserved_files", []),
    )


def serialize_session(session: Session) -> dict[str, Any]:
    """Serialize a Session to a dict for JSON storage."""
    return {
        "id": session.id,
        "branch": session.branch,
        "base_branch": session.base_branch,
        "phase": session.phase.value,
        "original_tree_hash": session.original_tree_hash,
        "discovered_intents": [_serialize_intent(i) for i in session.discovered_intents],
        "confirmed_intents": [_serialize_intent(i) for i in session.confirmed_intents],
        "change_plan": _serialize_change_plan(session.change_plan),
        "created_branches": session.created_branches,
        "created_prs": session.created_prs,
        "backtracks": [_serialize_backtrack(b) for b in session.backtracks],
        "current_attempt": session.current_attempt,
        "max_attempts": session.max_attempts,
        "total_tokens_used": session.total_tokens_used,
        "total_cost": session.total_cost,
        "max_cost": session.max_cost,
        "auto_mode": session.auto_mode,
        "auto_hint": session.auto_hint,
        "babysit_mode": session.babysit_mode,
        "dry_run": session.dry_run,
        "verbose": session.verbose,
        "no_verify_build": session.no_verify_build,
        "no_pr": session.no_pr,
    }


def deserialize_session(data: dict[str, Any]) -> Session:
    """Deserialize a Session from a dict."""
    return Session(
        id=data["id"],
        branch=data["branch"],
        base_branch=data["base_branch"],
        phase=SessionPhase(data["phase"]),
        original_tree_hash=data.get("original_tree_hash", ""),
        discovered_intents=[_deserialize_intent(i) for i in data.get("discovered_intents", [])],
        confirmed_intents=[_deserialize_intent(i) for i in data.get("confirmed_intents", [])],
        change_plan=_deserialize_change_plan(data.get("change_plan")),
        created_branches=data.get("created_branches", []),
        created_prs=data.get("created_prs", []),
        backtracks=[_deserialize_backtrack(b) for b in data.get("backtracks", [])],
        current_attempt=data.get("current_attempt", 1),
        max_attempts=data.get("max_attempts", 5),
        total_tokens_used=data.get("total_tokens_used", 0),
        total_cost=data.get("total_cost", 0.0),
        max_cost=data.get("max_cost"),
        auto_mode=data.get("auto_mode", False),
        auto_hint=data.get("auto_hint", ""),
        babysit_mode=data.get("babysit_mode", False),
        dry_run=data.get("dry_run", False),
        verbose=data.get("verbose", False),
        no_verify_build=data.get("no_verify_build", False),
        no_pr=data.get("no_pr", False),
    )


def save_session(session: Session) -> Path:
    """Save a session to disk."""
    ensure_sessions_dir()
    path = SESSIONS_DIR / f"{session.id}.json"

    with open(path, "w") as f:
        json.dump(serialize_session(session), f, indent=2)

    return path


def load_session(session_id: str) -> Session | None:
    """Load a session from disk."""
    path = SESSIONS_DIR / f"{session_id}.json"

    if not path.exists():
        return None

    with open(path, "r") as f:
        data = json.load(f)

    return deserialize_session(data)


def find_latest_session(branch: str | None = None) -> Session | None:
    """Find the most recent session, optionally filtered by branch."""
    ensure_sessions_dir()

    sessions = []
    for path in SESSIONS_DIR.glob("*.json"):
        try:
            with open(path, "r") as f:
                data = json.load(f)

            if branch is None or data.get("branch") == branch:
                sessions.append((path.stem, data))
        except (json.JSONDecodeError, KeyError):
            continue

    if not sessions:
        return None

    # Sort by session ID (which starts with timestamp)
    sessions.sort(key=lambda x: x[0], reverse=True)
    return deserialize_session(sessions[0][1])


def delete_session(session_id: str) -> bool:
    """Delete a session file."""
    path = SESSIONS_DIR / f"{session_id}.json"

    if path.exists():
        path.unlink()
        return True
    return False


def list_sessions() -> list[dict[str, Any]]:
    """List all saved sessions."""
    ensure_sessions_dir()

    sessions = []
    for path in SESSIONS_DIR.glob("*.json"):
        try:
            with open(path, "r") as f:
                data = json.load(f)

            sessions.append({
                "id": data["id"],
                "branch": data["branch"],
                "phase": data["phase"],
                "created": path.stem.split("-")[0],  # Extract date from ID
            })
        except (json.JSONDecodeError, KeyError):
            continue

    sessions.sort(key=lambda x: x["id"], reverse=True)
    return sessions
