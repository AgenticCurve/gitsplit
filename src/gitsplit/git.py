"""Git operations for gitsplit."""

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path

from git import Repo
from git.exc import GitCommandError, InvalidGitRepositoryError


@dataclass
class DiffHunk:
    """A hunk from a git diff."""

    file_path: str
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    content: str
    is_new_file: bool = False
    is_deleted_file: bool = False
    is_renamed: bool = False
    old_path: str | None = None


@dataclass
class FileDiff:
    """Diff information for a single file."""

    path: str
    hunks: list[DiffHunk]
    additions: int
    deletions: int
    is_new: bool = False
    is_deleted: bool = False
    is_renamed: bool = False
    old_path: str | None = None

    @property
    def full_diff(self) -> str:
        """Get the complete diff content for this file."""
        return "\n".join(h.content for h in self.hunks)


class GitError(Exception):
    """Git operation failed."""

    pass


class GitOperations:
    """Git operations wrapper."""

    def __init__(self, repo_path: str | Path | None = None):
        self.repo_path = Path(repo_path) if repo_path else Path.cwd()
        try:
            self.repo = Repo(self.repo_path)
        except InvalidGitRepositoryError:
            raise GitError(f"Not a git repository: {self.repo_path}")

    @property
    def current_branch(self) -> str:
        """Get the current branch name."""
        if self.repo.head.is_detached:
            return self.repo.head.commit.hexsha[:8]
        return self.repo.active_branch.name

    def get_default_branch(self) -> str:
        """Get the default branch (main or master)."""
        for name in ["main", "master"]:
            try:
                self.repo.refs[name]
                return name
            except (IndexError, KeyError):
                continue

        # Fallback: try to get from remote
        try:
            for remote in self.repo.remotes:
                for ref in remote.refs:
                    if "HEAD" in ref.name:
                        # Parse the symbolic reference
                        return ref.ref.name.split("/")[-1]
        except Exception:
            pass

        return "main"  # Default fallback

    def get_diff(self, base_branch: str | None = None) -> list[FileDiff]:
        """Get the diff between current branch and base branch."""
        if base_branch is None:
            base_branch = self.get_default_branch()

        try:
            # Get merge base to find where branches diverged
            merge_base = self.repo.merge_base(base_branch, "HEAD")
            if not merge_base:
                raise GitError(f"No common ancestor between {base_branch} and HEAD")

            merge_base_commit = merge_base[0]

            # Get diff from merge base to HEAD
            diff_index = merge_base_commit.diff("HEAD", create_patch=True)

        except GitCommandError as e:
            raise GitError(f"Failed to get diff: {e}")

        file_diffs = []
        for diff in diff_index:
            path = diff.b_path or diff.a_path
            hunks = self._parse_diff_hunks(diff, path)

            additions = 0
            deletions = 0
            if diff.diff:
                for line in diff.diff.decode("utf-8", errors="replace").split("\n"):
                    if line.startswith("+") and not line.startswith("+++"):
                        additions += 1
                    elif line.startswith("-") and not line.startswith("---"):
                        deletions += 1

            file_diffs.append(
                FileDiff(
                    path=path,
                    hunks=hunks,
                    additions=additions,
                    deletions=deletions,
                    is_new=diff.new_file,
                    is_deleted=diff.deleted_file,
                    is_renamed=diff.renamed,
                    old_path=diff.a_path if diff.renamed else None,
                )
            )

        return file_diffs

    def _parse_diff_hunks(self, diff, path: str) -> list[DiffHunk]:
        """Parse diff hunks from a diff object."""
        if not diff.diff:
            return []

        content = diff.diff.decode("utf-8", errors="replace")
        hunks = []

        # Split by hunk headers
        import re

        hunk_pattern = r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@"
        parts = re.split(f"({hunk_pattern})", content)

        i = 0
        while i < len(parts):
            if parts[i].startswith("@@"):
                match = re.match(hunk_pattern, parts[i])
                if match:
                    old_start = int(match.group(1))
                    old_count = int(match.group(2)) if match.group(2) else 1
                    new_start = int(match.group(3))
                    new_count = int(match.group(4)) if match.group(4) else 1

                    # Get hunk content (next part after the header components)
                    hunk_content = ""
                    if i + 1 < len(parts):
                        hunk_content = parts[i] + parts[i + 1]
                        i += 1

                    hunks.append(
                        DiffHunk(
                            file_path=path,
                            old_start=old_start,
                            old_count=old_count,
                            new_start=new_start,
                            new_count=new_count,
                            content=hunk_content,
                            is_new_file=diff.new_file,
                            is_deleted_file=diff.deleted_file,
                            is_renamed=diff.renamed,
                            old_path=diff.a_path if diff.renamed else None,
                        )
                    )
            i += 1

        # If no hunks parsed, create a single hunk with all content
        if not hunks and content:
            hunks.append(
                DiffHunk(
                    file_path=path,
                    old_start=1,
                    old_count=0,
                    new_start=1,
                    new_count=0,
                    content=content,
                    is_new_file=diff.new_file,
                    is_deleted_file=diff.deleted_file,
                    is_renamed=diff.renamed,
                    old_path=diff.a_path if diff.renamed else None,
                )
            )

        return hunks

    def get_raw_diff(self, base_branch: str | None = None) -> str:
        """Get the raw diff output as a string."""
        if base_branch is None:
            base_branch = self.get_default_branch()

        try:
            # Get merge base
            result = subprocess.run(
                ["git", "merge-base", base_branch, "HEAD"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True,
            )
            merge_base = result.stdout.strip()

            # Get diff
            result = subprocess.run(
                ["git", "diff", merge_base, "HEAD"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout

        except subprocess.CalledProcessError as e:
            raise GitError(f"Failed to get diff: {e.stderr}")

    def get_tree_hash(self, ref: str = "HEAD") -> str:
        """Get the tree hash for a ref (commit-independent content hash)."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", f"{ref}^{{tree}}"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            raise GitError(f"Failed to get tree hash: {e.stderr}")

    def get_content_hash(self, ref: str = "HEAD") -> str:
        """
        Get a content-only hash for verification.

        This hashes all file contents and paths, excluding:
        - Commit messages
        - Timestamps
        - Author metadata
        - Branch names
        """
        try:
            # Get list of all files and their content hashes
            result = subprocess.run(
                ["git", "ls-tree", "-r", ref],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True,
            )

            # Hash the tree output (contains mode, type, hash, path for each file)
            return hashlib.sha256(result.stdout.encode()).hexdigest()[:16]

        except subprocess.CalledProcessError as e:
            raise GitError(f"Failed to get content hash: {e.stderr}")

    def create_branch(self, name: str, from_ref: str = "HEAD") -> None:
        """Create a new branch."""
        try:
            self.repo.create_head(name, from_ref)
        except GitCommandError as e:
            raise GitError(f"Failed to create branch {name}: {e}")

    def checkout_branch(self, name: str) -> None:
        """Checkout a branch."""
        try:
            self.repo.heads[name].checkout()
        except (GitCommandError, IndexError) as e:
            raise GitError(f"Failed to checkout branch {name}: {e}")

    def delete_branch(self, name: str, force: bool = False) -> None:
        """Delete a branch."""
        try:
            self.repo.delete_head(name, force=force)
        except GitCommandError as e:
            raise GitError(f"Failed to delete branch {name}: {e}")

    def branch_exists(self, name: str) -> bool:
        """Check if a branch exists."""
        return name in [h.name for h in self.repo.heads]

    def apply_patch(self, patch: str) -> None:
        """Apply a patch to the working tree."""
        try:
            result = subprocess.run(
                ["git", "apply", "--check"],
                cwd=self.repo_path,
                input=patch,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise GitError(f"Patch would not apply cleanly: {result.stderr}")

            subprocess.run(
                ["git", "apply"],
                cwd=self.repo_path,
                input=patch,
                check=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            raise GitError(f"Failed to apply patch: {e}")

    def commit(self, message: str, allow_empty: bool = False) -> str:
        """Create a commit and return the commit hash."""
        try:
            args = ["-m", message]
            if allow_empty:
                args.append("--allow-empty")

            self.repo.index.commit(message)
            return self.repo.head.commit.hexsha

        except GitCommandError as e:
            raise GitError(f"Failed to commit: {e}")

    def stage_all(self) -> None:
        """Stage all changes."""
        self.repo.git.add("-A")

    def stage_files(self, files: list[str]) -> None:
        """Stage specific files."""
        for f in files:
            self.repo.git.add(f)

    def get_file_content(self, path: str, ref: str = "HEAD") -> str | None:
        """Get the content of a file at a specific ref."""
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

    def push_branch(self, branch: str, set_upstream: bool = True) -> None:
        """Push a branch to origin."""
        try:
            args = ["git", "push"]
            if set_upstream:
                args.extend(["-u", "origin", branch])
            else:
                args.extend(["origin", branch])

            subprocess.run(
                args,
                cwd=self.repo_path,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            raise GitError(f"Failed to push branch: {e.stderr}")

    def has_remote(self) -> bool:
        """Check if repo has a remote configured."""
        return len(self.repo.remotes) > 0

    def get_remote_url(self) -> str | None:
        """Get the URL of the origin remote."""
        try:
            return self.repo.remotes.origin.url
        except (AttributeError, IndexError):
            return None

    def reset_hard(self, ref: str) -> None:
        """Hard reset to a ref."""
        try:
            self.repo.git.reset("--hard", ref)
        except GitCommandError as e:
            raise GitError(f"Failed to reset: {e}")

    def stash(self) -> bool:
        """Stash current changes. Returns True if something was stashed."""
        try:
            result = self.repo.git.stash()
            return "No local changes" not in result
        except GitCommandError:
            return False

    def stash_pop(self) -> None:
        """Pop the latest stash."""
        try:
            self.repo.git.stash("pop")
        except GitCommandError as e:
            raise GitError(f"Failed to pop stash: {e}")
