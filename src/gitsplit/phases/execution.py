"""Phase 3: Execution - Create branches, apply changes, create PRs."""

import re
import subprocess
from typing import Any, Callable

from gitsplit.git import GitOperations, GitError
from gitsplit.models import Intent, ChangePlan, Session, SessionPhase, FileChange
from gitsplit.verification import Verifier, VerificationResult
from gitsplit.patch import PatchGenerator, FileDiff


class ExecutionError(Exception):
    """Execution failed."""

    pass


class Executor:
    """Phase 3: Execute the split plan with surgical patch application."""

    def __init__(
        self,
        git: GitOperations,
        verifier: Verifier,
        session: Session,
    ):
        self.git = git
        self.verifier = verifier
        self.session = session
        self.patch_gen = PatchGenerator(git.repo_path)

    def execute(
        self,
        plan: ChangePlan,
        on_progress: Callable | None = None,
    ) -> VerificationResult:
        """
        Execute the split plan: create branches, apply changes, create PRs.

        Uses surgical patch generation to split files across intents.
        """
        if not plan.execution_order:
            raise ExecutionError("No execution order in plan")

        # Save original branch and hash
        original_branch = self.session.branch
        original_hash = self.verifier.get_content_hash(original_branch)
        self.session.original_tree_hash = original_hash

        intents_by_id = {i.id: i for i in plan.intents}
        total = len(plan.execution_order)

        # Parse the full diff once - we'll use this to generate patches
        base_branch = self.session.base_branch
        full_diff = self.patch_gen.get_full_diff(base_branch, original_branch)
        parsed_diffs = self.patch_gen.parse_diff(full_diff)
        diffs_by_path = {d.path: d for d in parsed_diffs}

        # Track the previous branch for stacking
        previous_branch = base_branch
        created_branches = []

        try:
            for step, intent_id in enumerate(plan.execution_order, 1):
                intent = intents_by_id.get(intent_id)
                if not intent:
                    raise ExecutionError(f"Intent {intent_id} not found in plan")

                # Generate branch name
                branch_name = self._generate_branch_name(intent)
                intent.branch_name = branch_name

                if on_progress:
                    on_progress(step, total, branch_name, "creating")

                if not self.session.dry_run:
                    # Create branch from previous branch (for stacking)
                    self._create_intent_branch(intent, branch_name, previous_branch)
                    created_branches.append(branch_name)

                    if on_progress:
                        on_progress(step, total, branch_name, "applying changes")

                    # Apply surgical patches for this intent
                    self._apply_intent_patches(
                        intent, original_branch, diffs_by_path
                    )

                    # Verify build if needed
                    if not self.session.no_verify_build:
                        if on_progress:
                            on_progress(step, total, branch_name, "verifying build")

                        success, output = self.verifier.verify_intermediate_build()
                        if not success:
                            raise ExecutionError(f"Build failed for {branch_name}: {output}")

                    # Create PR if needed
                    if not self.session.no_pr and self.git.has_remote():
                        if on_progress:
                            on_progress(step, total, branch_name, "pushing")

                        self._push_and_create_pr(intent, branch_name, previous_branch)

                    if on_progress:
                        on_progress(step, total, branch_name, "done")

                # Update for next iteration (stacking)
                previous_branch = branch_name

            # Update session
            self.session.created_branches = created_branches

            # Final verification
            if not self.session.dry_run and created_branches:
                result = self.verifier.verify_split(original_branch, created_branches[-1])
            else:
                # Dry run - assume success
                result = VerificationResult(
                    passed=True,
                    original_hash=original_hash,
                    final_hash=original_hash,
                )

            return result

        except Exception as e:
            # Cleanup on failure
            self._cleanup_branches(created_branches)
            raise ExecutionError(f"Execution failed: {e}")

        finally:
            # Return to original branch
            if not self.session.dry_run:
                try:
                    self.git.checkout_branch(original_branch)
                except GitError:
                    pass

    def _generate_branch_name(self, intent: Intent) -> str:
        """Generate a branch name from an intent."""
        name = intent.name.lower()
        name = re.sub(r"[^\w\s-]", "", name)
        name = re.sub(r"[\s_]+", "-", name)
        name = re.sub(r"-+", "-", name)
        name = name.strip("-")

        if len(name) > 50:
            name = name[:50].rsplit("-", 1)[0]

        return name

    def _create_intent_branch(
        self,
        intent: Intent,
        branch_name: str,
        base_branch: str,
    ) -> None:
        """Create a branch for an intent."""
        if self.git.branch_exists(branch_name):
            self.git.delete_branch(branch_name, force=True)

        self.git.create_branch(branch_name, base_branch)
        self.git.checkout_branch(branch_name)

    def _apply_intent_patches(
        self,
        intent: Intent,
        source_branch: str,
        diffs_by_path: dict[str, FileDiff],
    ) -> None:
        """
        Apply surgical patches for an intent's file changes.

        For each file in the intent:
        - If entire_file: copy the whole file from source
        - If line_ranges: generate and apply a patch for only those lines
        """
        changes_made = False

        for file_change in intent.files:
            path = file_change.path
            file_diff = diffs_by_path.get(path)

            if file_change.is_entire_file or not file_change.line_ranges:
                # Copy entire file from source
                if self._copy_file_from_source(path, source_branch):
                    changes_made = True
            elif file_diff:
                # Generate surgical patch for specific line ranges
                line_ranges = [(lr.start, lr.end) for lr in file_change.line_ranges]
                patch = self.patch_gen.generate_patch_for_lines(file_diff, line_ranges)

                if patch:
                    if self.patch_gen.apply_patch(patch):
                        changes_made = True
                    else:
                        # Patch failed - try copying specific lines only
                        if self._copy_lines_from_source(path, source_branch, line_ranges):
                            changes_made = True
                        else:
                            # If that also fails, raise an error - don't silently break separation
                            raise ExecutionError(
                                f"Failed to apply patch for {path} lines {line_ranges}. "
                                "Line ranges may be incorrect or overlapping."
                            )
            else:
                # No diff info available - copy entire file
                if self._copy_file_from_source(path, source_branch):
                    changes_made = True

        if changes_made:
            self.git.stage_all()
            commit_msg = f"{intent.name}\n\n{intent.description}"
            self.git.commit(commit_msg)

    def _copy_file_from_source(self, path: str, source_branch: str) -> bool:
        """Copy an entire file from the source branch."""
        try:
            subprocess.run(
                ["git", "checkout", source_branch, "--", path],
                cwd=self.git.repo_path,
                check=True,
                capture_output=True,
            )
            return True
        except subprocess.CalledProcessError:
            return False

    def _copy_lines_from_source(
        self,
        path: str,
        source_branch: str,
        line_ranges: list[tuple[int, int]],
    ) -> bool:
        """
        Copy specific lines from source file to current file.

        This is a fallback when patch application fails.
        """
        try:
            # Get source file content
            result = subprocess.run(
                ["git", "show", f"{source_branch}:{path}"],
                cwd=self.git.repo_path,
                capture_output=True,
                text=True,
                check=True,
            )
            source_lines = result.stdout.split("\n")

            # Get current file content
            current_path = self.git.repo_path / path
            if current_path.exists():
                with open(current_path, "r") as f:
                    current_lines = f.read().split("\n")
            else:
                current_lines = []

            # Determine which lines to copy from source
            lines_to_copy: set[int] = set()
            for start, end in line_ranges:
                for line_num in range(start, end + 1):
                    lines_to_copy.add(line_num)

            # Build new content - extend current if source is longer
            max_line = max(max(lines_to_copy), len(current_lines))
            result_lines = [""] * max_line

            # Start with current content
            for i, line in enumerate(current_lines):
                if i < max_line:
                    result_lines[i] = line

            # Overlay source lines for specified ranges
            for line_num in lines_to_copy:
                if line_num <= len(source_lines):
                    # Line numbers are 1-indexed
                    idx = line_num - 1
                    if idx < len(source_lines):
                        # Ensure result_lines is long enough
                        while idx >= len(result_lines):
                            result_lines.append("")
                        result_lines[idx] = source_lines[idx]

            # Write the result
            with open(current_path, "w") as f:
                f.write("\n".join(result_lines))

            return True

        except (subprocess.CalledProcessError, IOError):
            return False

    def _push_and_create_pr(
        self,
        intent: Intent,
        branch_name: str,
        base_branch: str,
    ) -> None:
        """Push branch and create PR."""
        try:
            self.git.push_branch(branch_name)

            result = subprocess.run(
                [
                    "gh", "pr", "create",
                    "--title", intent.name,
                    "--body", intent.description or "Created by gitsplit",
                    "--base", base_branch,
                    "--head", branch_name,
                ],
                cwd=self.git.repo_path,
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                pr_url = result.stdout.strip()
                intent.pr_url = pr_url

                match = re.search(r"/pull/(\d+)", pr_url)
                if match:
                    intent.pr_number = int(match.group(1))

                self.session.created_prs.append({
                    "branch": branch_name,
                    "pr_number": intent.pr_number,
                    "pr_url": pr_url,
                })

        except subprocess.CalledProcessError:
            pass

    def _cleanup_branches(self, branches: list[str]) -> None:
        """Clean up created branches on failure."""
        original_branch = self.session.branch

        try:
            self.git.checkout_branch(original_branch)
        except GitError:
            pass

        for branch in branches:
            try:
                self.git.delete_branch(branch, force=True)
            except GitError:
                pass

    def rebuild_from_plan(
        self,
        plan: ChangePlan,
        starting_from: str | None = None,
        on_progress: Callable | None = None,
    ) -> VerificationResult:
        """Rebuild branches from a plan, optionally starting from a specific intent."""
        execution_order = plan.execution_order

        if starting_from:
            try:
                start_idx = execution_order.index(starting_from)
                execution_order = execution_order[start_idx:]

                for intent_id in execution_order:
                    intent = next((i for i in plan.intents if i.id == intent_id), None)
                    if intent and intent.branch_name:
                        try:
                            self.git.delete_branch(intent.branch_name, force=True)
                        except GitError:
                            pass

            except ValueError:
                pass

        partial_plan = ChangePlan(
            intents=[i for i in plan.intents if i.id in execution_order],
            conflicts=plan.conflicts,
            execution_order=execution_order,
            is_validated=False,
        )

        return self.execute(partial_plan, on_progress)
