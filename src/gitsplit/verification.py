"""Hash verification for gitsplit - The Golden Rule."""

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gitsplit.git import GitOperations, GitError
from gitsplit.models import VerificationResult


class VerificationError(Exception):
    """Verification operation failed."""

    pass


class Verifier:
    """Handles hash verification for split operations."""

    def __init__(self, git: GitOperations):
        self.git = git

    def get_tree_hash(self, ref: str = "HEAD") -> str:
        """Get the tree hash for a ref."""
        return self.git.get_tree_hash(ref)

    def get_content_hash(self, ref: str = "HEAD") -> str:
        """Get the content hash for a ref."""
        return self.git.get_content_hash(ref)

    def verify_split(
        self,
        original_ref: str,
        split_tip_ref: str,
    ) -> VerificationResult:
        """
        Verify that the split result is identical to the original.

        The Golden Rule: Hash(Original Code) must equal Hash(Final Split Code).
        """
        try:
            original_hash = self.get_content_hash(original_ref)
            final_hash = self.get_content_hash(split_tip_ref)

            if original_hash == final_hash:
                return VerificationResult(
                    passed=True,
                    original_hash=original_hash,
                    final_hash=final_hash,
                )

            # Hash mismatch - find differences
            differences = self._find_differences(original_ref, split_tip_ref)

            return VerificationResult(
                passed=False,
                original_hash=original_hash,
                final_hash=final_hash,
                differences=differences,
            )

        except GitError as e:
            raise VerificationError(f"Failed to verify split: {e}")

    def _find_differences(
        self,
        original_ref: str,
        split_tip_ref: str,
    ) -> list[dict[str, Any]]:
        """Find detailed differences between two refs."""
        differences = []

        try:
            # Get diff between the two refs
            result = subprocess.run(
                ["git", "diff", original_ref, split_tip_ref],
                cwd=self.git.repo_path,
                capture_output=True,
                text=True,
            )

            if result.returncode == 0 and result.stdout:
                # Parse the diff to find specific differences
                current_file = None
                for line in result.stdout.split("\n"):
                    if line.startswith("diff --git"):
                        # Extract file path
                        parts = line.split()
                        if len(parts) >= 4:
                            current_file = parts[3][2:]  # Remove 'b/' prefix

                    elif line.startswith("@@") and current_file:
                        # Parse hunk header for line numbers
                        # Format: @@ -start,count +start,count @@
                        import re
                        match = re.search(r"@@ -(\d+)", line)
                        if match:
                            line_num = int(match.group(1))
                            differences.append({
                                "file": current_file,
                                "line": line_num,
                                "description": f"Difference at line {line_num}",
                            })

                    elif current_file and (line.startswith("+") or line.startswith("-")):
                        if not line.startswith("+++") and not line.startswith("---"):
                            # Record the actual difference
                            change_type = "added" if line.startswith("+") else "removed"
                            content = line[1:].strip()[:50]  # First 50 chars
                            if differences and differences[-1]["file"] == current_file:
                                # Update existing entry
                                if "changes" not in differences[-1]:
                                    differences[-1]["changes"] = []
                                differences[-1]["changes"].append({
                                    "type": change_type,
                                    "content": content,
                                })

        except subprocess.CalledProcessError:
            pass

        return differences

    def diagnose_failure(
        self,
        verification_result: VerificationResult,
    ) -> dict[str, Any]:
        """
        Diagnose a verification failure and suggest remediation.

        Returns information to feed back to the AI for self-healing.
        """
        diagnosis = {
            "severity": "unknown",
            "likely_cause": "unknown",
            "suggested_action": "retry",
            "details": [],
        }

        if verification_result.passed:
            diagnosis["severity"] = "none"
            diagnosis["likely_cause"] = "none"
            diagnosis["suggested_action"] = "none"
            return diagnosis

        differences = verification_result.differences
        if not differences:
            diagnosis["severity"] = "high"
            diagnosis["likely_cause"] = "major structural difference"
            diagnosis["suggested_action"] = "backtrack_to_phase1"
            return diagnosis

        # Analyze the differences
        num_files = len(set(d.get("file", "") for d in differences))
        total_changes = sum(len(d.get("changes", [])) for d in differences)

        if num_files == 1 and total_changes <= 5:
            # Small difference in one file
            diagnosis["severity"] = "low"
            diagnosis["likely_cause"] = "line misattribution"
            diagnosis["suggested_action"] = "retry_phase2"
            diagnosis["details"] = [
                f"File: {differences[0].get('file', 'unknown')}",
                f"Changes: {total_changes} lines affected",
            ]

        elif num_files <= 3 and total_changes <= 20:
            # Moderate differences
            diagnosis["severity"] = "medium"
            diagnosis["likely_cause"] = "multiple line misattributions"
            diagnosis["suggested_action"] = "retry_phase2_with_context"
            diagnosis["details"] = [
                f"Files affected: {num_files}",
                f"Total changes: {total_changes}",
            ]

        else:
            # Many differences - likely intent boundaries are wrong
            diagnosis["severity"] = "high"
            diagnosis["likely_cause"] = "intent boundary errors"
            diagnosis["suggested_action"] = "backtrack_to_phase1"
            diagnosis["details"] = [
                f"Files affected: {num_files}",
                f"Total changes: {total_changes}",
                "Likely need to re-discover intents",
            ]

        # Add the differences to details
        for diff in differences[:5]:  # Limit to first 5
            diagnosis["details"].append(
                f"  {diff.get('file', 'unknown')}: line {diff.get('line', '?')}"
            )

        return diagnosis

    def verify_intermediate_build(
        self,
        build_command: str | None = None,
    ) -> tuple[bool, str]:
        """
        Verify that the current state builds successfully.

        Returns (success, output).
        """
        if build_command is None:
            # Check for Python files and verify syntax
            python_check = self._check_python_syntax()
            if python_check is not None:
                return python_check

            # Try common build commands
            for cmd in ["npm run build", "make", "cargo build", "go build ./..."]:
                try:
                    result = subprocess.run(
                        cmd.split(),
                        cwd=self.git.repo_path,
                        capture_output=True,
                        text=True,
                        timeout=300,  # 5 minute timeout
                    )
                    if result.returncode == 0:
                        return True, result.stdout

                except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
                    continue

            # No build command found or all failed
            return True, "No build command found - skipping build verification"

        # User provided a build command
        try:
            result = subprocess.run(
                build_command.split(),
                cwd=self.git.repo_path,
                capture_output=True,
                text=True,
                timeout=300,
            )
            return result.returncode == 0, result.stdout + result.stderr

        except subprocess.TimeoutExpired:
            return False, "Build timed out"
        except subprocess.CalledProcessError as e:
            return False, str(e)

    def _check_python_syntax(self) -> tuple[bool, str] | None:
        """Check Python file syntax if any .py files exist."""
        import ast
        repo_path = Path(self.git.repo_path)

        # Find all Python files
        py_files = list(repo_path.glob("**/*.py"))
        if not py_files:
            return None  # No Python files

        errors = []
        for py_file in py_files:
            # Skip hidden directories and common non-source directories
            if any(part.startswith(".") or part in ("venv", "node_modules", "__pycache__")
                   for part in py_file.parts):
                continue

            try:
                # Use ast.parse instead of py_compile to avoid creating .pyc files
                with open(py_file, "r") as f:
                    source = f.read()
                ast.parse(source, filename=str(py_file))
            except SyntaxError as e:
                errors.append(f"{py_file.name}: {e.msg} at line {e.lineno}")
            except Exception as e:
                errors.append(f"{py_file.name}: {str(e)}")

        if errors:
            return False, "Python syntax errors:\n" + "\n".join(errors)

        return True, "Python syntax check passed"
