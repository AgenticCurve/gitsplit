"""Patch generation for surgical file splitting.

This module handles the complex task of splitting a file's changes
across multiple intents, generating valid unified diff patches for each.
"""

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DiffLine:
    """A single line from a diff."""

    content: str
    line_type: str  # '+', '-', ' ' (context), or header
    old_line_num: int | None = None  # Line number in old file
    new_line_num: int | None = None  # Line number in new file

    @property
    def is_addition(self) -> bool:
        return self.line_type == "+"

    @property
    def is_deletion(self) -> bool:
        return self.line_type == "-"

    @property
    def is_context(self) -> bool:
        return self.line_type == " "


@dataclass
class DiffHunk:
    """A hunk from a unified diff."""

    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: list[DiffLine] = field(default_factory=list)
    header: str = ""

    def get_lines_in_range(self, start: int, end: int, use_new: bool = True) -> list[DiffLine]:
        """Get lines that fall within a line range."""
        result = []
        for line in self.lines:
            line_num = line.new_line_num if use_new else line.old_line_num
            if line_num is not None and start <= line_num <= end:
                result.append(line)
            elif line.is_context:
                # Include context if adjacent to range
                if line_num is not None and (start - 3 <= line_num <= end + 3):
                    result.append(line)
        return result


@dataclass
class FileDiff:
    """Complete diff for a single file."""

    old_path: str
    new_path: str
    hunks: list[DiffHunk] = field(default_factory=list)
    is_new_file: bool = False
    is_deleted: bool = False
    is_binary: bool = False

    @property
    def path(self) -> str:
        return self.new_path or self.old_path


class PatchGenerator:
    """Generates patches for surgical file splitting."""

    def __init__(self, repo_path: str | Path):
        self.repo_path = Path(repo_path)

    def get_full_diff(self, base_ref: str, source_ref: str) -> str:
        """Get the full diff between two refs."""
        result = subprocess.run(
            ["git", "diff", base_ref, source_ref],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout

    def parse_diff(self, diff_text: str) -> list[FileDiff]:
        """Parse a unified diff into structured FileDiff objects."""
        files = []
        current_file = None
        current_hunk = None

        lines = diff_text.split("\n")
        i = 0

        while i < len(lines):
            line = lines[i]

            # New file diff header
            if line.startswith("diff --git"):
                if current_file:
                    if current_hunk:
                        current_file.hunks.append(current_hunk)
                    files.append(current_file)

                # Parse paths from "diff --git a/path b/path"
                match = re.match(r"diff --git a/(.*) b/(.*)", line)
                if match:
                    current_file = FileDiff(
                        old_path=match.group(1),
                        new_path=match.group(2),
                    )
                else:
                    current_file = FileDiff(old_path="", new_path="")
                current_hunk = None
                i += 1
                continue

            # File metadata
            if current_file:
                if line.startswith("new file"):
                    current_file.is_new_file = True
                elif line.startswith("deleted file"):
                    current_file.is_deleted = True
                elif line.startswith("Binary"):
                    current_file.is_binary = True

            # Hunk header
            if line.startswith("@@"):
                if current_hunk and current_file:
                    current_file.hunks.append(current_hunk)

                # Parse @@ -old_start,old_count +new_start,new_count @@
                match = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)", line)
                if match:
                    current_hunk = DiffHunk(
                        old_start=int(match.group(1)),
                        old_count=int(match.group(2)) if match.group(2) else 1,
                        new_start=int(match.group(3)),
                        new_count=int(match.group(4)) if match.group(4) else 1,
                        header=line,
                    )
                i += 1
                continue

            # Diff content lines
            if current_hunk and line:
                if line.startswith("+") and not line.startswith("+++"):
                    current_hunk.lines.append(DiffLine(
                        content=line[1:],  # Remove the + prefix
                        line_type="+",
                    ))
                elif line.startswith("-") and not line.startswith("---"):
                    current_hunk.lines.append(DiffLine(
                        content=line[1:],  # Remove the - prefix
                        line_type="-",
                    ))
                elif line.startswith(" "):
                    current_hunk.lines.append(DiffLine(
                        content=line[1:],  # Remove the space prefix
                        line_type=" ",
                    ))

            i += 1

        # Don't forget the last file/hunk
        if current_file:
            if current_hunk:
                current_file.hunks.append(current_hunk)
            files.append(current_file)

        # Assign line numbers to each diff line
        for file_diff in files:
            for hunk in file_diff.hunks:
                self._assign_line_numbers(hunk)

        return files

    def _assign_line_numbers(self, hunk: DiffHunk) -> None:
        """Assign old and new line numbers to each line in a hunk."""
        old_num = hunk.old_start
        new_num = hunk.new_start

        for line in hunk.lines:
            if line.is_context:
                line.old_line_num = old_num
                line.new_line_num = new_num
                old_num += 1
                new_num += 1
            elif line.is_deletion:
                line.old_line_num = old_num
                old_num += 1
            elif line.is_addition:
                line.new_line_num = new_num
                new_num += 1

    def generate_patch_for_lines(
        self,
        file_diff: FileDiff,
        line_ranges: list[tuple[int, int]],
    ) -> str | None:
        """
        Generate a patch containing only changes within the specified line ranges.

        Args:
            file_diff: The parsed file diff
            line_ranges: List of (start, end) tuples for lines to include

        Returns:
            A valid unified diff patch string, or None if no changes in range
        """
        if file_diff.is_binary:
            return None

        # For new files, include everything if any range overlaps
        if file_diff.is_new_file:
            return self._generate_full_file_patch(file_diff)

        # Build patch header
        patch_lines = [
            f"--- a/{file_diff.old_path}",
            f"+++ b/{file_diff.new_path}",
        ]

        hunks_added = 0

        for hunk in file_diff.hunks:
            # Check if this hunk overlaps with any of our ranges
            hunk_lines = self._filter_hunk_for_ranges(hunk, line_ranges)

            if hunk_lines:
                # Generate hunk header and content
                hunk_patch = self._generate_hunk_patch(hunk, hunk_lines)
                if hunk_patch:
                    patch_lines.extend(hunk_patch)
                    hunks_added += 1

        if hunks_added == 0:
            return None

        return "\n".join(patch_lines) + "\n"

    def _filter_hunk_for_ranges(
        self,
        hunk: DiffHunk,
        line_ranges: list[tuple[int, int]],
    ) -> list[DiffLine]:
        """Filter hunk lines to only include those in the specified ranges.

        Line ranges reference the NEW file (feature branch). For additions, we check
        new_line_num directly. For deletions (which don't exist in the new file),
        we include them if they're adjacent to additions that ARE in range.
        """
        # First pass: mark which lines are directly in range (additions only)
        # and compute virtual new line numbers for deletions
        in_range_indices: set[int] = set()

        # Compute virtual new line numbers for all lines
        # Deletions get the new_line_num of the nearest context/addition
        virtual_new_nums: list[int | None] = []
        last_new_num = hunk.new_start

        for line in hunk.lines:
            if line.new_line_num is not None:
                last_new_num = line.new_line_num
                virtual_new_nums.append(line.new_line_num)
            else:
                # Deletion - use the last known new line number
                virtual_new_nums.append(last_new_num)

        # Mark additions and deletions that are in range
        for i, line in enumerate(hunk.lines):
            virtual_num = virtual_new_nums[i]
            if virtual_num is None:
                continue

            for start, end in line_ranges:
                if start <= virtual_num <= end:
                    in_range_indices.add(i)
                    break

        # If no changes in range, return empty
        if not in_range_indices:
            return []

        # Second pass: build result with proper context
        result = []
        context_before: list[DiffLine] = []
        last_included_idx = -1

        for i, line in enumerate(hunk.lines):
            if i in in_range_indices:
                # Include up to 3 context lines before
                for ctx in context_before[-3:]:
                    if ctx not in result:
                        result.append(ctx)
                result.append(line)
                last_included_idx = len(result) - 1
                context_before = []
            elif line.is_context:
                if result and len(result) - 1 - last_included_idx < 3:
                    # Include context after included changes (up to 3 lines)
                    result.append(line)
                else:
                    context_before.append(line)

        return result

    def _generate_hunk_patch(
        self,
        original_hunk: DiffHunk,
        filtered_lines: list[DiffLine],
    ) -> list[str] | None:
        """Generate a valid hunk from filtered lines."""
        if not filtered_lines:
            return None

        # Calculate line counts
        old_count = sum(1 for l in filtered_lines if l.is_context or l.is_deletion)
        new_count = sum(1 for l in filtered_lines if l.is_context or l.is_addition)

        # Find the starting line numbers - need both old and new
        # For a valid hunk, we need correlated old/new start positions
        old_start = None
        new_start = None

        # First, try to find a context line (has both old and new)
        for line in filtered_lines:
            if line.is_context and line.old_line_num and line.new_line_num:
                old_start = line.old_line_num
                new_start = line.new_line_num
                break

        # If no context, calculate from first addition/deletion
        if old_start is None or new_start is None:
            # Find first line with known position
            first_old = None
            first_new = None
            for line in filtered_lines:
                if first_old is None and line.old_line_num:
                    first_old = line.old_line_num
                if first_new is None and line.new_line_num:
                    first_new = line.new_line_num

            # Calculate correlation from original hunk
            # The offset between old and new should be preserved
            offset = original_hunk.new_start - original_hunk.old_start

            if first_old is not None and first_new is None:
                old_start = first_old
                new_start = first_old + offset
            elif first_new is not None and first_old is None:
                new_start = first_new
                old_start = first_new - offset
            elif first_old is not None and first_new is not None:
                old_start = first_old
                new_start = first_new
            else:
                old_start = original_hunk.old_start
                new_start = original_hunk.new_start

        # Build the hunk
        result = [f"@@ -{old_start},{old_count} +{new_start},{new_count} @@"]

        for line in filtered_lines:
            if line.is_addition:
                result.append(f"+{line.content}")
            elif line.is_deletion:
                result.append(f"-{line.content}")
            else:
                result.append(f" {line.content}")

        return result

    def _generate_full_file_patch(self, file_diff: FileDiff) -> str:
        """Generate a patch for an entire new file."""
        lines = [
            f"--- /dev/null",
            f"+++ b/{file_diff.new_path}",
        ]

        for hunk in file_diff.hunks:
            lines.append(hunk.header)
            for line in hunk.lines:
                if line.is_addition:
                    lines.append(f"+{line.content}")
                elif line.is_deletion:
                    lines.append(f"-{line.content}")
                else:
                    lines.append(f" {line.content}")

        return "\n".join(lines) + "\n"

    def apply_patch(self, patch: str) -> bool:
        """Apply a patch to the working directory."""
        if not patch.strip():
            return True

        try:
            # First, check if patch applies cleanly
            result = subprocess.run(
                ["git", "apply", "--check"],
                cwd=self.repo_path,
                input=patch,
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                # Try with more lenient options
                result = subprocess.run(
                    ["git", "apply", "--check", "-3"],  # 3-way merge
                    cwd=self.repo_path,
                    input=patch,
                    capture_output=True,
                    text=True,
                )

            if result.returncode != 0:
                return False

            # Apply the patch
            subprocess.run(
                ["git", "apply"],
                cwd=self.repo_path,
                input=patch,
                check=True,
                text=True,
            )
            return True

        except subprocess.CalledProcessError:
            return False

    def get_file_at_ref(self, path: str, ref: str) -> str | None:
        """Get file contents at a specific ref."""
        try:
            result = subprocess.run(
                ["git", "show", f"{ref}:{path}"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                return result.stdout
            return None
        except subprocess.CalledProcessError:
            return None
