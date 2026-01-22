"""Phase 2: Change Planning - Map every line to an intent."""

from gitsplit.ai import AIClient, AIError, CHANGE_PLANNING_SYSTEM, parse_json_response
from gitsplit.git import GitOperations
from gitsplit.models import (
    Intent,
    ChangePlan,
    MultiIntentConflict,
    ResolutionStrategy,
    LineRange,
    Session,
)


class PlanningError(Exception):
    """Change planning failed."""

    pass


class ChangePlanner:
    """Phase 2: Create a detailed plan for splitting changes."""

    def __init__(self, git: GitOperations, ai: AIClient, session: Session):
        self.git = git
        self.ai = ai
        self.session = session

    def plan(self, intents: list[Intent]) -> ChangePlan:
        """
        Create a change plan from confirmed intents.

        Maps every changed line to a specific intent and identifies
        conflicts that need resolution.

        Continues the conversation from Discovery phase for context.
        """
        if not intents:
            raise PlanningError("No intents provided")

        # Get the diff for reference
        diff = self.git.get_raw_diff(self.session.base_branch)

        # Build context for AI - this continues the conversation from discovery
        context = self._build_context(intents, diff)

        # Switch to planning system prompt but keep conversation history
        # The AI will see: discovery request -> discovery response -> planning request
        self.ai._current_system = CHANGE_PLANNING_SYSTEM

        # Call AI for planning, continuing conversation
        try:
            response = self.ai.complete(
                messages=[{"role": "user", "content": context}],
                temperature=0.0,
                max_tokens=8192,
                use_conversation=True,
            )
        except AIError as e:
            raise PlanningError(f"AI planning failed: {e}")

        # Parse response
        try:
            result = parse_json_response(response.content)
        except AIError as e:
            raise PlanningError(f"Failed to parse AI response: {e}")

        # Build the change plan
        plan = self._build_plan(intents, result)

        # Update session
        self.session.change_plan = plan
        self.session.total_tokens_used += response.input_tokens + response.output_tokens
        self.session.total_cost += response.cost

        return plan

    def retry_with_error(self, error: str, diagnosis: str | None = None) -> ChangePlan:
        """
        Retry planning after an error, using conversation history.

        The AI will see the previous planning attempt and the error.
        """
        intents = self.session.confirmed_intents

        # Add error context to conversation
        self.ai.add_error_context(error, diagnosis)

        # Retry with conversation history
        try:
            response = self.ai.complete(
                temperature=0.0,
                max_tokens=8192,
                use_conversation=True,
            )
        except AIError as e:
            raise PlanningError(f"AI planning failed on retry: {e}")

        try:
            result = parse_json_response(response.content)
        except AIError as e:
            raise PlanningError(f"Failed to parse AI response: {e}")

        plan = self._build_plan(intents, result)

        self.session.change_plan = plan
        self.session.total_tokens_used += response.input_tokens + response.output_tokens
        self.session.total_cost += response.cost

        return plan

    def _build_context(self, intents: list[Intent], diff: str) -> str:
        """Build context message for AI."""
        parts = []

        parts.append("Now let's create a detailed change plan.\n\n")
        parts.append("## Confirmed Intents\n\n")
        for intent in intents:
            parts.append(f"### {intent.id}: {intent.name}\n")
            parts.append(f"{intent.description}\n")
            parts.append("Files:\n")
            for f in intent.files:
                if f.is_entire_file:
                    parts.append(f"  - {f.path} (entire file)\n")
                else:
                    ranges = ", ".join(f"{lr.start}-{lr.end}" for lr in f.line_ranges)
                    parts.append(f"  - {f.path} (lines: {ranges})\n")
            parts.append("\n")

        parts.append("## Full Diff\n```diff\n")
        parts.append(diff)
        parts.append("\n```\n")

        parts.append("\n## Instructions\n")
        parts.append("Create a precise plan that maps EVERY changed line to an intent.\n")
        parts.append("For lines that belong to multiple intents, specify the resolution strategy:\n")
        parts.append("- 'stack': Intent B depends on Intent A (B includes A's change)\n")
        parts.append("- 'merge': Combine the intents\n")
        parts.append("- 'duplicate': Both intents include the change\n")

        return "".join(parts)

    def _build_plan(self, intents: list[Intent], result: dict) -> ChangePlan:
        """Build ChangePlan from AI response."""
        # Process dependencies
        dependencies = result.get("dependencies", [])
        for dep in dependencies:
            from_id = dep.get("from")
            to_id = dep.get("to")
            if from_id and to_id:
                for intent in intents:
                    if intent.id == from_id and to_id not in intent.dependencies:
                        intent.dependencies.append(to_id)

        # Process execution order
        execution_order = result.get("execution_order", [i.id for i in intents])

        # Identify conflicts from file plans
        conflicts = []
        file_plans = result.get("file_plans", [])

        for fp in file_plans:
            path = fp.get("path", "")
            assignments = fp.get("assignments", [])

            # Check for shared lines
            shared_assignments = [a for a in assignments if a.get("intent_id") == "shared"]

            for shared in shared_assignments:
                shared_by = shared.get("shared_by", [])
                strategy = shared.get("strategy", "stack")

                if len(shared_by) >= 2:
                    lines = shared.get("lines", [0, 0])
                    line_range = LineRange(lines[0], lines[1]) if len(lines) >= 2 else LineRange(0, 0)

                    # Create overlap tuples
                    overlaps = []
                    for i in range(len(shared_by) - 1):
                        overlaps.append((shared_by[i], shared_by[i + 1], line_range))

                    conflict = MultiIntentConflict(
                        file_path=path,
                        intent_ids=shared_by,
                        overlapping_ranges=overlaps,
                        suggested_strategy=ResolutionStrategy(strategy),
                    )
                    conflicts.append(conflict)

        return ChangePlan(
            intents=intents,
            conflicts=conflicts,
            execution_order=execution_order,
            is_validated=False,
        )

    def replan(
        self,
        preserved_files: list[str],
        error_context: str,
    ) -> ChangePlan:
        """
        Re-plan after a verification failure, preserving some file plans.

        Uses conversation history so the AI understands what went wrong.

        Args:
            preserved_files: Files whose plans should be kept
            error_context: Information about why re-planning is needed

        Returns:
            Updated change plan
        """
        intents = self.session.confirmed_intents
        diff = self.git.get_raw_diff(self.session.base_branch)

        # Build context with error information
        context = self._build_replan_context(intents, diff, preserved_files, error_context)

        # Continue conversation with error context
        try:
            response = self.ai.complete(
                messages=[{"role": "user", "content": context}],
                temperature=0.0,
                max_tokens=8192,
                use_conversation=True,
            )
        except AIError as e:
            raise PlanningError(f"AI planning failed: {e}")

        try:
            result = parse_json_response(response.content)
        except AIError as e:
            raise PlanningError(f"Failed to parse AI response: {e}")

        # Build the new plan
        plan = self._build_plan(intents, result)

        # Update session
        self.session.change_plan = plan
        self.session.total_tokens_used += response.input_tokens + response.output_tokens
        self.session.total_cost += response.cost

        return plan

    def _build_replan_context(
        self,
        intents: list[Intent],
        diff: str,
        preserved_files: list[str],
        error_context: str,
    ) -> str:
        """Build context for re-planning."""
        parts = []

        parts.append("## Previous Plan Failed\n")
        parts.append(f"{error_context}\n\n")
        parts.append("Please create an updated plan that fixes these issues.\n\n")

        if preserved_files:
            parts.append("## Preserved File Plans (do not change)\n")
            for path in preserved_files:
                parts.append(f"  - {path}\n")
            parts.append("\n")

        parts.append("## Intents\n\n")
        for intent in intents:
            parts.append(f"### {intent.id}: {intent.name}\n")
            for f in intent.files:
                if f.is_entire_file:
                    parts.append(f"  - {f.path} (entire file)\n")
                else:
                    ranges = ", ".join(f"{lr.start}-{lr.end}" for lr in f.line_ranges)
                    parts.append(f"  - {f.path} (lines: {ranges})\n")
            parts.append("\n")

        parts.append("## Full Diff\n```diff\n")
        parts.append(diff)
        parts.append("\n```\n")

        parts.append("\n## Instructions\n")
        parts.append("Re-create the plan, fixing the issues mentioned above.\n")
        parts.append("Pay special attention to the lines that caused the verification failure.\n")
        parts.append("CRITICAL: Line ranges must refer to the NEW file (feature branch).\n")
        parts.append("Read hunk headers carefully: @@ -old_start,old_count +new_start,new_count @@\n\n")
        parts.append("Output as JSON (same format as before):\n")
        parts.append("{\n")
        parts.append('  "file_plans": [...],\n')
        parts.append('  "dependencies": [...],\n')
        parts.append('  "execution_order": [...]\n')
        parts.append("}\n")

        return "".join(parts)

    def resolve_conflict(
        self,
        conflict: MultiIntentConflict,
        strategy: ResolutionStrategy,
    ) -> None:
        """Resolve a conflict with the specified strategy."""
        conflict.resolved = True
        conflict.chosen_strategy = strategy

        # Update the change plan
        if self.session.change_plan:
            for c in self.session.change_plan.conflicts:
                if c.file_path == conflict.file_path:
                    c.resolved = conflict.resolved
                    c.chosen_strategy = conflict.chosen_strategy
