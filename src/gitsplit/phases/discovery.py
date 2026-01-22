"""Phase 1: Intent Discovery - Analyze changes and identify logical intents."""

import subprocess

from gitsplit.ai import AIClient, AIError, INTENT_DISCOVERY_SYSTEM, parse_json_response
from gitsplit.git import GitOperations, FileDiff
from gitsplit.models import Intent, FileChange, LineRange, Session


class DiscoveryError(Exception):
    """Intent discovery failed."""

    pass


class IntentDiscovery:
    """Phase 1: Discover intents from git diff."""

    def __init__(self, git: GitOperations, ai: AIClient, session: Session):
        self.git = git
        self.ai = ai
        self.session = session

    def discover(self, hint: str | None = None) -> list[Intent]:
        """
        Analyze the diff and discover intents.

        Args:
            hint: Optional user-provided hint to guide discovery

        Returns:
            List of discovered intents
        """
        # Get the diff
        diff = self.git.get_raw_diff(self.session.base_branch)
        file_diffs = self.git.get_diff(self.session.base_branch)

        if not diff.strip():
            raise DiscoveryError("No changes found between branches")

        # Reset conversation for discovery phase
        self.ai.reset_conversation(system=INTENT_DISCOVERY_SYSTEM)

        # Build context for AI
        context = self._build_context(diff, file_diffs, hint)

        # Call AI for intent discovery with conversation history
        try:
            response = self.ai.complete(
                messages=[{"role": "user", "content": context}],
                temperature=0.0,
                max_tokens=4096,
                use_conversation=True,
            )
        except AIError as e:
            raise DiscoveryError(f"AI analysis failed: {e}")

        # Parse response
        try:
            result = parse_json_response(response.content)
        except AIError as e:
            raise DiscoveryError(f"Failed to parse AI response: {e}")

        # Convert to Intent objects
        intents = self._parse_intents(result, file_diffs)

        # Update session
        self.session.discovered_intents = intents
        self.session.total_tokens_used += response.input_tokens + response.output_tokens
        self.session.total_cost += response.cost

        return intents

    def retry_with_error(self, error: str, diagnosis: str | None = None) -> list[Intent]:
        """
        Retry discovery after an error, using conversation history.

        The AI will see the previous attempt and the error, allowing it to
        understand what went wrong and produce a better response.
        """
        file_diffs = self.git.get_diff(self.session.base_branch)

        # Add error context to conversation
        self.ai.add_error_context(error, diagnosis)

        # Retry with conversation history
        try:
            response = self.ai.complete(
                temperature=0.0,
                max_tokens=4096,
                use_conversation=True,
            )
        except AIError as e:
            raise DiscoveryError(f"AI analysis failed on retry: {e}")

        try:
            result = parse_json_response(response.content)
        except AIError as e:
            raise DiscoveryError(f"Failed to parse AI response: {e}")

        intents = self._parse_intents(result, file_diffs)

        self.session.discovered_intents = intents
        self.session.total_tokens_used += response.input_tokens + response.output_tokens
        self.session.total_cost += response.cost

        return intents

    def _build_context(
        self,
        diff: str,
        file_diffs: list[FileDiff],
        hint: str | None,
    ) -> str:
        """Build context message for AI."""
        parts = []

        # Summary of changes
        parts.append("## Changes Summary\n")
        parts.append(f"Total files changed: {len(file_diffs)}\n")

        for fd in file_diffs:
            status = ""
            if fd.is_new:
                status = " (new file)"
            elif fd.is_deleted:
                status = " (deleted)"
            elif fd.is_renamed:
                status = f" (renamed from {fd.old_path})"

            parts.append(f"- {fd.path}{status}: +{fd.additions} -{fd.deletions}\n")

        parts.append("\n")

        # User hint if provided
        if hint:
            parts.append(f"## User Hint\n{hint}\n\n")

        # Full diff
        parts.append("## Full Diff\n```diff\n")
        parts.append(diff)
        parts.append("\n```\n")

        return "".join(parts)

    def _parse_intents(
        self,
        result: dict,
        file_diffs: list[FileDiff],
    ) -> list[Intent]:
        """Parse AI response into Intent objects."""
        intents = []

        for i, intent_data in enumerate(result.get("intents", [])):
            intent_id = intent_data.get("id", f"intent-{chr(ord('a') + i)}")

            files = []
            for file_data in intent_data.get("files", []):
                path = file_data.get("path", "")

                # Find matching file diff for stats
                matching_diff = next(
                    (fd for fd in file_diffs if fd.path == path),
                    None,
                )

                line_ranges = []
                for lr in file_data.get("line_ranges", []):
                    if isinstance(lr, list) and len(lr) >= 2:
                        line_ranges.append(LineRange(lr[0], lr[1]))

                is_entire_file = file_data.get("is_entire_file", False)

                # Calculate additions/deletions for this intent's portion
                additions = 0
                deletions = 0
                if matching_diff:
                    if is_entire_file:
                        additions = matching_diff.additions
                        deletions = matching_diff.deletions
                    else:
                        # Estimate based on line ranges
                        total_lines = sum(lr.end - lr.start + 1 for lr in line_ranges)
                        ratio = total_lines / max(
                            matching_diff.additions + matching_diff.deletions, 1
                        )
                        additions = int(matching_diff.additions * ratio)
                        deletions = int(matching_diff.deletions * ratio)

                files.append(
                    FileChange(
                        path=path,
                        line_ranges=line_ranges,
                        is_entire_file=is_entire_file,
                        additions=additions,
                        deletions=deletions,
                    )
                )

            intents.append(
                Intent(
                    id=intent_id,
                    name=intent_data.get("name", f"Intent {chr(ord('A') + i)}"),
                    description=intent_data.get("description", ""),
                    files=files,
                )
            )

        # Post-process: for files only touched by one intent, use is_entire_file
        intents = self._optimize_file_assignments(intents, file_diffs)

        # Detect and fix overlapping line ranges
        intents = self._fix_overlapping_ranges(intents)

        # Expand line ranges to complete Python blocks
        intents = self._expand_to_complete_blocks(intents, file_diffs)

        return intents

    def _expand_to_complete_blocks(
        self,
        intents: list[Intent],
        file_diffs: list[FileDiff],
    ) -> list[Intent]:
        """
        Expand line ranges to include complete code blocks.

        This ensures we don't cut off function bodies or control structures mid-way.
        """
        import ast

        for intent in intents:
            for fc in intent.files:
                if fc.is_entire_file or not fc.line_ranges:
                    continue

                # Only process Python files
                if not fc.path.endswith(".py"):
                    continue

                # Get the source file content
                source_content = self._get_file_content(fc.path)
                if not source_content:
                    continue

                # Parse the file to find block boundaries
                try:
                    tree = ast.parse(source_content)
                except SyntaxError:
                    continue

                # Get all block boundaries (functions, classes, control structures)
                blocks = self._get_block_boundaries(tree, source_content)

                # Expand each line range to complete blocks
                expanded_ranges = []
                for lr in fc.line_ranges:
                    new_start, new_end = lr.start, lr.end

                    # Find any blocks that overlap with this range and expand to include them
                    for block_start, block_end in blocks:
                        # If our range intersects with this block, expand to include it
                        if (lr.start <= block_end and lr.end >= block_start):
                            new_start = min(new_start, block_start)
                            new_end = max(new_end, block_end)

                    expanded_ranges.append(LineRange(new_start, new_end))

                # Merge adjacent/overlapping ranges and fill small gaps (empty lines)
                fc.line_ranges = self._merge_ranges_with_gaps(expanded_ranges)

        return intents

    def _merge_ranges_with_gaps(
        self,
        ranges: list[LineRange],
        max_gap: int = 3,
    ) -> list[LineRange]:
        """
        Merge ranges that are adjacent or have small gaps between them.

        This ensures empty lines between functions are included.
        """
        if not ranges:
            return ranges

        # Sort by start
        sorted_ranges = sorted(ranges, key=lambda r: r.start)

        merged = [sorted_ranges[0]]
        for lr in sorted_ranges[1:]:
            prev = merged[-1]
            # If this range starts within max_gap lines of previous end, merge them
            if lr.start <= prev.end + max_gap + 1:
                merged[-1] = LineRange(prev.start, max(prev.end, lr.end))
            else:
                merged.append(lr)

        return merged

    def _get_file_content(self, path: str) -> str | None:
        """Get file content from the current branch."""
        try:
            result = subprocess.run(
                ["git", "show", f"HEAD:{path}"],
                cwd=self.git.repo_path,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                return result.stdout
        except subprocess.CalledProcessError:
            pass
        return None

    def _get_block_boundaries(
        self,
        tree: "ast.AST",
        source: str,
    ) -> list[tuple[int, int]]:
        """Get (start_line, end_line) for all code blocks in an AST."""
        import ast

        blocks = []
        source_lines = source.split("\n")

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                # Function or class definition
                start = node.lineno
                end = node.end_lineno or start
                blocks.append((start, end))

            elif isinstance(node, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
                # Control structures
                start = node.lineno
                end = node.end_lineno or start
                blocks.append((start, end))

        return blocks

    def _fix_overlapping_ranges(self, intents: list[Intent]) -> list[Intent]:
        """
        Detect and fix overlapping line ranges across intents.

        If two intents claim overlapping lines in the same file,
        assign those lines to the first intent and remove from the second.
        """
        # Build a map of file -> list of (intent_idx, range) sorted by start
        file_ranges: dict[str, list[tuple[int, int, int, int]]] = {}  # path -> [(intent_idx, range_idx, start, end)]

        for intent_idx, intent in enumerate(intents):
            for fc in intent.files:
                if fc.is_entire_file:
                    continue
                for range_idx, lr in enumerate(fc.line_ranges):
                    if fc.path not in file_ranges:
                        file_ranges[fc.path] = []
                    file_ranges[fc.path].append((intent_idx, range_idx, lr.start, lr.end))

        # For each file, check for overlaps
        for path, ranges in file_ranges.items():
            if len(ranges) < 2:
                continue

            # Sort by start line
            ranges.sort(key=lambda x: x[2])

            # Detect overlaps
            for i in range(len(ranges) - 1):
                curr_intent, curr_range_idx, curr_start, curr_end = ranges[i]
                next_intent, next_range_idx, next_start, next_end = ranges[i + 1]

                # Check if current range overlaps with next
                if curr_end >= next_start and curr_intent != next_intent:
                    # Overlap detected - adjust the second range
                    new_start = curr_end + 1
                    if new_start <= next_end:
                        # Update the range
                        fc = intents[next_intent].files
                        for f in fc:
                            if f.path == path and len(f.line_ranges) > next_range_idx:
                                f.line_ranges[next_range_idx] = LineRange(new_start, next_end)
                    else:
                        # Range is completely consumed - remove it
                        fc = intents[next_intent].files
                        for f in fc:
                            if f.path == path and len(f.line_ranges) > next_range_idx:
                                f.line_ranges.pop(next_range_idx)

        return intents

    def _optimize_file_assignments(
        self,
        intents: list[Intent],
        file_diffs: list[FileDiff],
    ) -> list[Intent]:
        """
        Optimize file assignments for reliability.

        For files that are only touched by a single intent, set is_entire_file=True.
        This avoids line number misattribution issues.
        """
        # Count how many intents touch each file
        file_intent_count: dict[str, int] = {}
        for intent in intents:
            for fc in intent.files:
                file_intent_count[fc.path] = file_intent_count.get(fc.path, 0) + 1

        # For files touched by only one intent, use is_entire_file
        for intent in intents:
            for fc in intent.files:
                if file_intent_count.get(fc.path, 0) == 1 and not fc.is_entire_file:
                    # This file is only in this intent - use entire file
                    fc.is_entire_file = True
                    fc.line_ranges = []

                    # Update stats to full file stats
                    matching_diff = next(
                        (fd for fd in file_diffs if fd.path == fc.path),
                        None,
                    )
                    if matching_diff:
                        fc.additions = matching_diff.additions
                        fc.deletions = matching_diff.deletions

        return intents

    def rediscover(
        self,
        preserved_intents: list[str],
        error_context: str,
    ) -> list[Intent]:
        """
        Re-discover intents after a backtrack, preserving some intents.

        Uses conversation history to give the AI context about what went wrong.

        Args:
            preserved_intents: IDs of intents that should be kept
            error_context: Information about why re-discovery is needed

        Returns:
            Updated list of intents
        """
        # Get the diff
        diff = self.git.get_raw_diff(self.session.base_branch)
        file_diffs = self.git.get_diff(self.session.base_branch)

        # Build context with error information and add to conversation
        context = self._build_rediscovery_context(
            diff, file_diffs, preserved_intents, error_context
        )

        # Add to existing conversation (don't reset - keep history)
        try:
            response = self.ai.complete(
                messages=[{"role": "user", "content": context}],
                temperature=0.0,
                max_tokens=4096,
                use_conversation=True,
            )
        except AIError as e:
            raise DiscoveryError(f"AI analysis failed: {e}")

        try:
            result = parse_json_response(response.content)
        except AIError as e:
            raise DiscoveryError(f"Failed to parse AI response: {e}")

        # Parse new intents
        new_intents = self._parse_intents(result, file_diffs)

        # Merge with preserved intents
        preserved = [i for i in self.session.discovered_intents if i.id in preserved_intents]
        final_intents = preserved + new_intents

        # Update session
        self.session.discovered_intents = final_intents
        self.session.total_tokens_used += response.input_tokens + response.output_tokens
        self.session.total_cost += response.cost

        return final_intents

    def _build_rediscovery_context(
        self,
        diff: str,
        file_diffs: list[FileDiff],
        preserved_intents: list[str],
        error_context: str,
    ) -> str:
        """Build context for re-discovery."""
        parts = []

        # Error context
        parts.append("## Previous Attempt Failed\n")
        parts.append(f"{error_context}\n\n")
        parts.append("Please re-analyze and provide updated intents.\n\n")

        # Preserved intents
        if preserved_intents:
            parts.append("## Preserved Intents (do not change these)\n")
            for intent in self.session.discovered_intents:
                if intent.id in preserved_intents:
                    parts.append(f"- {intent.id}: {intent.name}\n")
                    for f in intent.files:
                        parts.append(f"  - {f.path}\n")
            parts.append("\n")

        # Changes summary (only non-preserved files)
        preserved_files = set()
        for intent in self.session.discovered_intents:
            if intent.id in preserved_intents:
                preserved_files.update(f.path for f in intent.files)

        parts.append("## Remaining Changes to Analyze\n")
        for fd in file_diffs:
            if fd.path not in preserved_files:
                parts.append(f"- {fd.path}: +{fd.additions} -{fd.deletions}\n")

        parts.append("\n## Full Diff\n```diff\n")
        parts.append(diff)
        parts.append("\n```\n")

        return "".join(parts)
