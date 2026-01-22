"""Main split engine with backtracking and self-healing."""

from gitsplit.ai import AIClient
from gitsplit.git import GitOperations
from gitsplit.models import (
    Session,
    SessionPhase,
    Intent,
    ChangePlan,
    BacktrackInfo,
    VerificationResult,
)
from gitsplit.phases import IntentDiscovery, ChangePlanner, Executor
from gitsplit.verification import Verifier
from gitsplit.session import save_session
from gitsplit import display


class EngineError(Exception):
    """Engine operation failed."""

    pass


class SplitEngine:
    """
    The main split engine implementing the three-phase architecture
    with backtracking and self-healing.
    """

    def __init__(
        self,
        git: GitOperations,
        ai: AIClient,
        session: Session,
    ):
        self.git = git
        self.ai = ai
        self.session = session

        self.verifier = Verifier(git)
        self.discovery = IntentDiscovery(git, ai, session)
        self.planner = ChangePlanner(git, ai, session)
        self.executor = Executor(git, self.verifier, session)

    def run(self) -> bool:
        """
        Run the split engine.

        Returns True if split completed successfully.
        """
        try:
            while self.session.current_attempt <= self.session.max_attempts:
                try:
                    # Phase 1: Discovery
                    if self.session.phase in [SessionPhase.INIT, SessionPhase.DISCOVERY]:
                        self._run_discovery()

                    # User confirmation (unless auto mode)
                    if self.session.phase == SessionPhase.DISCOVERY:
                        if self.session.auto_mode:
                            # Auto-confirm intents in auto mode
                            for intent in self.session.discovered_intents:
                                intent.is_confirmed = True
                            self.session.confirmed_intents = self.session.discovered_intents
                            self.session.phase = SessionPhase.PLANNING
                        else:
                            if not self._confirm_intents():
                                return False

                    # Phase 2: Planning
                    if self.session.phase == SessionPhase.PLANNING:
                        self._run_planning()

                    # Phase 3: Execution
                    if self.session.phase == SessionPhase.EXECUTION:
                        result = self._run_execution()

                        if result.passed:
                            self.session.phase = SessionPhase.COMPLETE
                            self._show_success()
                            return True

                        # Verification failed - try self-healing
                        if not self._handle_verification_failure(result):
                            # Self-healing exhausted
                            self.session.phase = SessionPhase.FAILED
                            display.print_error("Split failed after maximum retry attempts")
                            return False

                except Exception as e:
                    display.print_error(str(e))
                    self.session.current_attempt += 1

                    if self.session.current_attempt > self.session.max_attempts:
                        self.session.phase = SessionPhase.FAILED
                        return False

                    # Save session for potential resume
                    save_session(self.session)

            return False

        finally:
            # Always save session at end
            save_session(self.session)

    def _run_discovery(self) -> None:
        """Run Phase 1: Intent Discovery."""
        self.session.phase = SessionPhase.DISCOVERY

        display.print_scanning(self.session.branch, self.session.base_branch)

        file_diffs = self.git.get_diff(self.session.base_branch)
        display.print_file_count(len(file_diffs))

        with display.create_spinner("Identifying intents...") as progress:
            task = progress.add_task("Identifying intents...", total=None)

            hint = self.session.auto_hint if self.session.auto_mode else None
            intents = self.discovery.discover(hint)

            progress.update(task, completed=True)

        display.print_info(f"Done in {self.ai.usage.total_cost:.2f}s (estimated)")
        display.print_intents(intents, f"Found {len(intents)} distinct intents:")
        display.print_pr_stack(intents)

    def _confirm_intents(self) -> bool:
        """Get user confirmation for discovered intents."""
        choice = display.prompt_proceed()

        if choice.lower() == "n":
            display.print_info("Aborted")
            return False

        if choice.lower() == "e":
            # Escape hatch - manual editing
            self._handle_escape_hatch()

        # Mark intents as confirmed
        for intent in self.session.discovered_intents:
            intent.is_confirmed = True

        self.session.confirmed_intents = self.session.discovered_intents
        self.session.phase = SessionPhase.PLANNING
        return True

    def _handle_escape_hatch(self) -> None:
        """Handle escape hatch for manual editing."""
        action = display.print_escape_hatch_prompt()

        if action == "a":
            raise EngineError("Aborted by user")

        if action == "e":
            # Open editor with intent mapping
            # TODO: Implement editor integration
            display.print_warning("Editor integration not yet implemented")

        # s = skip, continue with current intents

    def _run_planning(self) -> None:
        """Run Phase 2: Change Planning."""
        display.print_info("Creating change plan...")

        with display.create_spinner("Mapping changes to intents...") as progress:
            task = progress.add_task("Mapping changes to intents...", total=None)

            plan = self.planner.plan(self.session.confirmed_intents)

            progress.update(task, completed=True)

        # Handle conflicts if any
        if plan.conflicts and self.session.babysit_mode:
            self._resolve_conflicts(plan)

        self.session.change_plan = plan
        self.session.phase = SessionPhase.EXECUTION

    def _resolve_conflicts(self, plan: ChangePlan) -> None:
        """Resolve conflicts in babysit mode."""
        for conflict in plan.conflicts:
            if not conflict.resolved:
                question = (
                    f"File '{conflict.file_path}' has overlapping changes "
                    f"for intents: {', '.join(conflict.intent_ids)}"
                )
                options = [
                    f"Stack (Intent {conflict.intent_ids[1]} depends on {conflict.intent_ids[0]})",
                    "Merge (combine into single PR)",
                    "Duplicate (both intents include the change)",
                ]

                choice = display.print_babysit_question(question, options)

                if "Stack" in choice:
                    from gitsplit.models import ResolutionStrategy
                    self.planner.resolve_conflict(conflict, ResolutionStrategy.STACK)
                elif "Merge" in choice:
                    self.planner.resolve_conflict(conflict, ResolutionStrategy.MERGE)
                else:
                    self.planner.resolve_conflict(conflict, ResolutionStrategy.DUPLICATE)

    def _run_execution(self) -> VerificationResult:
        """Run Phase 3: Execution."""
        if self.session.dry_run:
            display.print_dry_run_notice()

        display.print_creating_split()

        plan = self.session.change_plan
        if not plan:
            raise EngineError("No change plan available")

        def on_progress(step: int, total: int, branch: str, status: str) -> None:
            display.print_branch_progress(step, total, branch, status)

        result = self.executor.execute(plan, on_progress)

        display.print_verification_result(result)
        return result

    def _handle_verification_failure(
        self,
        result: VerificationResult,
    ) -> bool:
        """
        Handle a verification failure through self-healing.

        Uses conversation history to give the AI context about what went wrong,
        allowing it to understand and fix the issue.

        Returns True if should continue retrying, False if exhausted.
        """
        self.session.current_attempt += 1

        if self.session.current_attempt > self.session.max_attempts:
            return False

        # Diagnose the failure
        diagnosis = self.verifier.diagnose_failure(result)
        display.print_retry(
            self.session.current_attempt,
            self.session.max_attempts,
            diagnosis["likely_cause"],
        )

        # Determine backtrack target based on diagnosis
        action = diagnosis["suggested_action"]
        error_details = "\n".join(diagnosis.get("details", []))

        if action == "retry_phase2" or action == "retry_phase2_with_context":
            # Retry planning with error context in conversation
            display.print_backtrack(
                self.session.phase.value,
                SessionPhase.PLANNING.value,
                diagnosis["likely_cause"],
            )

            try:
                # Use conversation-aware retry
                plan = self.planner.retry_with_error(
                    error=result.diagnosis,
                    diagnosis=error_details,
                )
                self.session.change_plan = plan
                self.session.phase = SessionPhase.EXECUTION
            except Exception as e:
                display.print_error(f"Re-planning failed: {e}")
                # Fall back to full backtrack
                self._backtrack_to(SessionPhase.PLANNING, result.diagnosis)

        elif action == "backtrack_to_phase1":
            # Major error - need to re-discover intents
            display.print_backtrack(
                self.session.phase.value,
                SessionPhase.DISCOVERY.value,
                diagnosis["likely_cause"],
            )

            try:
                # Use conversation-aware rediscovery
                intents = self.discovery.rediscover(
                    preserved_intents=diagnosis.get("preserved_intents", []),
                    error_context=result.diagnosis,
                )
                self.session.discovered_intents = intents
                self.session.phase = SessionPhase.DISCOVERY  # Need user confirmation again
            except Exception as e:
                display.print_error(f"Re-discovery failed: {e}")
                self._backtrack_to(SessionPhase.DISCOVERY, result.diagnosis)

        else:
            # Default: retry planning with conversation
            try:
                plan = self.planner.retry_with_error(
                    error=result.diagnosis,
                    diagnosis=error_details,
                )
                self.session.change_plan = plan
                self.session.phase = SessionPhase.EXECUTION
            except Exception:
                self._backtrack_to(SessionPhase.PLANNING, result.diagnosis)

        # Escalate AI tier if needed
        if self.session.current_attempt >= 3:
            self.ai.escalate_tier()

        return True

    def _backtrack_to(
        self,
        target_phase: SessionPhase,
        reason: str,
        preserved_intents: list[str] | None = None,
        preserved_files: list[str] | None = None,
    ) -> None:
        """Backtrack to a previous phase."""
        backtrack = BacktrackInfo(
            from_phase=self.session.phase,
            to_phase=target_phase,
            reason=reason,
            attempt=self.session.current_attempt,
            preserved_intents=preserved_intents or [],
            preserved_files=preserved_files or [],
        )

        self.session.backtracks.append(backtrack)
        self.session.phase = target_phase

        display.print_backtrack(
            self.session.phase.value,
            target_phase.value,
            reason,
        )

        # Save session
        save_session(self.session)

    def _show_success(self) -> None:
        """Show success message and summary."""
        intents = self.session.confirmed_intents
        display.print_split_complete(intents)

        display.print_cost_summary(
            self.session.total_tokens_used,
            self.session.total_cost,
        )

        if not self.session.dry_run:
            display.print_info(
                f"Original branch '{self.session.branch}' preserved. "
                f"Delete it manually when ready."
            )


def create_engine(
    repo_path: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    max_cost: float | None = None,
    session: Session | None = None,
) -> SplitEngine:
    """Create a configured split engine."""
    git = GitOperations(repo_path)
    ai = AIClient(api_key, model_override=model, max_cost=max_cost)

    if session is None:
        from gitsplit.session import generate_session_id

        session = Session(
            id=generate_session_id(),
            branch=git.current_branch,
            base_branch=git.get_default_branch(),
        )

    return SplitEngine(git, ai, session)
