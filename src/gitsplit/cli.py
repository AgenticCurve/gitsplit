"""CLI for gitsplit."""

import sys

import rich_click as click
from rich.console import Console

from gitsplit import display

# Configure rich-click for pretty help output
click.rich_click.USE_RICH_MARKUP = True
click.rich_click.USE_MARKDOWN = True
click.rich_click.SHOW_ARGUMENTS = True
click.rich_click.GROUP_ARGUMENTS_OPTIONS = True
click.rich_click.STYLE_ERRORS_SUGGESTION = "magenta italic"
click.rich_click.ERRORS_SUGGESTION = "Try running '--help' for more information."
click.rich_click.MAX_WIDTH = 100
click.rich_click.STYLE_OPTION = "bold cyan"
click.rich_click.STYLE_ARGUMENT = "bold cyan"
click.rich_click.STYLE_COMMAND = "bold green"
click.rich_click.STYLE_SWITCH = "bold yellow"
from gitsplit.engine import create_engine, EngineError
from gitsplit.git import GitOperations, GitError
from gitsplit.models import Session, SessionPhase
from gitsplit.session import (
    generate_session_id,
    find_latest_session,
    load_session,
    list_sessions,
    save_session,
)


console = Console()


@click.group(invoke_without_command=True)
@click.option(
    "--auto", "-a", "auto_hint", default=None, metavar="HINT",
    help="Run in headless mode with a context hint to guide AI intent discovery. "
         "Skips all interactive prompts. Example: --auto 'split logging and auth'"
)
@click.option(
    "--dry-run", is_flag=True,
    help="Preview the split without making any changes. Shows discovered intents, "
         "planned branches, and verifies the hash would match. Safe to run anytime."
)
@click.option(
    "--resume", is_flag=True,
    help="Continue the most recent interrupted session for the current branch. "
         "Sessions are saved automatically and can be resumed after failures."
)
@click.option(
    "--verify-only", is_flag=True,
    help="Verify an existing split matches the original branch hash. "
         "Use after manual adjustments to confirm the Golden Rule still holds."
)
@click.option(
    "--diagnose", is_flag=True,
    help="When hash verification fails, show a detailed diff of exactly "
         "what differs between original and split result."
)
@click.option(
    "--branch", "-b", default=None, metavar="NAME",
    help="Branch to split. Defaults to current branch."
)
@click.option(
    "--base", default=None, metavar="NAME",
    help="Base branch to split against. Defaults to 'main' or 'master'."
)
@click.option(
    "--babysit", is_flag=True,
    help="Interactive mode that prompts for confirmation at every decision point: "
         "intent discovery, conflict resolution, and branch creation."
)
@click.option(
    "--max-attempts", default=5, metavar="N", show_default=True,
    help="Maximum retry attempts when AI makes mistakes. Each retry uses "
         "conversation history for context-aware self-correction."
)
@click.option(
    "--max-cost", default=None, type=float, metavar="USD",
    help="Budget limit for API calls in USD. Stops execution if exceeded. "
         "Useful for preventing runaway costs on large diffs."
)
@click.option(
    "--model", default=None, metavar="NAME",
    help="Override the AI model via OpenRouter. Defaults to a fast model, "
         "escalating to stronger models on retry failures."
)
@click.option(
    "--no-verify", is_flag=True,
    help="Skip build/syntax verification between splits. Faster but won't "
         "catch broken intermediate states."
)
@click.option(
    "--no-pr", is_flag=True,
    help="Create local branches only, don't push or create GitHub PRs. "
         "Useful for reviewing the split locally before publishing."
)
@click.option(
    "--progressive", is_flag=True,
    help="Force file-by-file splitting even when the AI suggests grouping. "
         "More granular but may create more PRs than necessary."
)
@click.option(
    "--verbose", "-v", is_flag=True,
    help="Show detailed output including AI prompts, responses, and reasoning. "
         "Helpful for debugging or understanding AI decisions."
)
@click.option(
    "--json", "json_output", is_flag=True,
    help="Output final session state as JSON. Useful for scripting or "
         "integrating with other tools."
)
@click.pass_context
def cli(
    ctx,
    auto_hint,
    dry_run,
    resume,
    verify_only,
    diagnose,
    branch,
    base,
    babysit,
    max_attempts,
    max_cost,
    model,
    no_verify,
    no_pr,
    progressive,
    verbose,
    json_output,
):
    """**gitsplit** - AI-powered semantic Git branch splitter.

    Transform messy branches with mixed changes into clean, atomic PRs.
    Uses AI to identify distinct logical intents in your changes and
    automatically splits them into a stack of dependent branches.

    **The Golden Rule:**

        Hash(Original Branch) == Hash(Final Split Result)

    No code is ever lost or modified in the splitting process.

    **Examples:**

        gitsplit                        Interactive split of current branch

        gitsplit --dry-run              Preview without making changes

        gitsplit --auto "auth and ui"   Headless mode with AI hint

        gitsplit -b feature --base dev  Split specific branch

        gitsplit --no-pr                Local branches only, no PRs

    **Workflow (3 phases):**

        1. DISCOVERY    AI identifies logical intents in your diff

        2. PLANNING     Maps each line to an intent, resolves conflicts

        3. EXECUTION    Creates branches, applies patches, verifies hash

    **Environment:**

        OPENROUTER_API_KEY   Required. Get one at https://openrouter.ai/keys
    """
    # If a subcommand is invoked, skip the main logic
    if ctx.invoked_subcommand is not None:
        return

    display.print_header()

    try:
        git = GitOperations()
    except GitError as e:
        display.print_error(str(e))
        sys.exit(1)

    # Handle resume
    if resume:
        session = find_latest_session(branch or git.current_branch)
        if session is None:
            display.print_error("No session found to resume")
            sys.exit(1)

        display.print_session_resumed(session.id)
    else:
        # Create new session
        session = Session(
            id=generate_session_id(),
            branch=branch or git.current_branch,
            base_branch=base or git.get_default_branch(),
            auto_mode=auto_hint is not None,
            auto_hint=auto_hint or "",
            babysit_mode=babysit,
            dry_run=dry_run,
            verbose=verbose,
            no_verify_build=no_verify,
            no_pr=no_pr,
            max_attempts=max_attempts,
            max_cost=max_cost,
        )

    # Handle verify-only
    if verify_only:
        _verify_only(git, session, diagnose)
        return

    # Run the engine
    try:
        engine = create_engine(
            api_key=None,  # Uses OPENROUTER_API_KEY env var
            model=model,
            max_cost=max_cost,
            session=session,
        )

        success = engine.run()

        if json_output:
            _output_json(session)

        sys.exit(0 if success else 1)

    except EngineError as e:
        display.print_error(str(e))
        sys.exit(1)


def _verify_only(git: GitOperations, session: Session, diagnose: bool) -> None:
    """Verify an existing split."""
    from gitsplit.verification import Verifier

    verifier = Verifier(git)

    # Find the tip of the split stack
    if not session.created_branches:
        display.print_error("No branches found to verify")
        sys.exit(1)

    tip_branch = session.created_branches[-1]
    result = verifier.verify_split(session.branch, tip_branch)

    display.print_verification_result(result)

    if not result.passed and diagnose:
        diagnosis = verifier.diagnose_failure(result)
        console.print("\n[bold]Diagnosis:[/bold]")
        console.print(f"  Severity: {diagnosis['severity']}")
        console.print(f"  Likely cause: {diagnosis['likely_cause']}")
        console.print(f"  Suggested action: {diagnosis['suggested_action']}")
        for detail in diagnosis.get("details", []):
            console.print(f"  {detail}")

    sys.exit(0 if result.passed else 1)


def _output_json(session: Session) -> None:
    """Output session as JSON."""
    import json
    from gitsplit.session import serialize_session

    print(json.dumps(serialize_session(session), indent=2))


@cli.command()
def sessions():
    """List all saved sessions.

    Sessions are automatically saved during execution and can be resumed
    if interrupted. This command shows all sessions with their status.

    **Session phases:**

        init        Session created, not yet started
        discovery   Identifying intents from diff
        planning    Mapping lines to intents
        execution   Creating branches and PRs
        complete    Successfully finished
        failed      Stopped due to error
    """
    saved = list_sessions()

    if not saved:
        console.print("No saved sessions found")
        return

    from rich.table import Table

    table = Table(title="Saved Sessions")
    table.add_column("ID")
    table.add_column("Branch")
    table.add_column("Phase")
    table.add_column("Created")

    for s in saved:
        table.add_row(
            s["id"],
            s["branch"],
            s["phase"],
            s["created"],
        )

    console.print(table)


@cli.command("resume-session")
@click.argument("session_id", metavar="SESSION_ID")
def resume_session(session_id: str):
    """Resume a specific session by its ID.

    Use `gitsplit sessions` to list available session IDs.
    The session will continue from where it left off, using the
    preserved conversation history for better AI context.

    **Example:**

        gitsplit resume-session 20240115-143022-abc123
    """
    session = load_session(session_id)

    if session is None:
        display.print_error(f"Session not found: {session_id}")
        sys.exit(1)

    display.print_header()
    display.print_session_resumed(session.id)

    try:
        engine = create_engine(session=session)
        success = engine.run()
        sys.exit(0 if success else 1)

    except EngineError as e:
        display.print_error(str(e))
        sys.exit(1)


@cli.command()
def version():
    """Show version and environment information.

    Displays the gitsplit version and checks for required
    environment configuration.
    """
    import os
    from gitsplit import __version__

    console.print(f"gitsplit version {__version__}")
    console.print()

    # Check environment
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if api_key:
        masked = api_key[:10] + "..." + api_key[-4:]
        console.print(f"[green]✓[/green] OPENROUTER_API_KEY: {masked}")
    else:
        console.print("[red]✗[/red] OPENROUTER_API_KEY: not set")
        console.print("  Set this environment variable to use gitsplit.")
        console.print("  Get a key at: https://openrouter.ai/keys")


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
