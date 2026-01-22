"""Rich terminal display for gitsplit."""

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text
from rich.prompt import Confirm, Prompt

from gitsplit.models import Intent, ChangePlan, VerificationResult, Session


console = Console()


def print_header() -> None:
    """Print the gitsplit header."""
    console.print()
    console.print("[bold cyan]gitsplit[/bold cyan] - Semantic Git Splitter")
    console.print()


def print_scanning(branch: str, base: str) -> None:
    """Print scanning message."""
    console.print(f"Analyzing changes on branch '[bold]{branch}[/bold]'...")
    console.print(f"  Comparing against '[bold]{base}[/bold]'")


def print_file_count(count: int) -> None:
    """Print number of files being scanned."""
    console.print(f"  Scanning [bold]{count}[/bold] files")


def create_spinner(message: str) -> Progress:
    """Create a spinner progress indicator."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    )


def print_intents(intents: list[Intent], title: str = "Found intents:") -> None:
    """Print discovered intents in formatted tables."""
    console.print()
    console.print(f"[bold]{title}[/bold]")
    console.print()

    for i, intent in enumerate(intents):
        # Create table for this intent
        table = Table(
            show_header=True,
            header_style="bold",
            box=None,
            padding=(0, 1),
            collapse_padding=True,
        )

        # Intent header
        intent_label = chr(ord("A") + i)
        header = f"INTENT {intent_label}: {intent.name}"
        console.print(Panel(header, style="cyan", width=70))

        # File table
        file_table = Table(show_header=False, box=None, padding=(0, 2))
        file_table.add_column("File", style="white")
        file_table.add_column("Changes", style="green")
        file_table.add_column("Lines", style="dim")

        for file_change in intent.files:
            changes = f"+{file_change.additions} -{file_change.deletions}"

            if file_change.is_entire_file:
                lines = "entire file"
            else:
                ranges = []
                for lr in file_change.line_ranges:
                    if lr.start == lr.end:
                        ranges.append(f"line {lr.start}")
                    else:
                        ranges.append(f"lines {lr.start}-{lr.end}")
                lines = ", ".join(ranges) if ranges else ""

            file_table.add_row(file_change.path, changes, lines)

        console.print(file_table)
        console.print()


def print_pr_stack(intents: list[Intent]) -> None:
    """Print the proposed PR stack order."""
    labels = [chr(ord("A") + i) for i in range(len(intents))]
    stack = " -> ".join(labels)
    console.print(f"Proposed PR stack: [bold cyan]{stack}[/bold cyan]")
    console.print()


def prompt_proceed() -> str:
    """Prompt user to proceed with split."""
    return Prompt.ask(
        "Proceed with this split?",
        choices=["y", "n", "e"],
        default="y",
    )


def prompt_confirm(message: str, default: bool = True) -> bool:
    """Simple yes/no confirmation."""
    return Confirm.ask(message, default=default)


def prompt_choice(message: str, choices: list[str]) -> str:
    """Prompt for a choice from a list."""
    return Prompt.ask(message, choices=choices)


def print_creating_split() -> None:
    """Print creating split message."""
    console.print()
    console.print("[bold]Creating split...[/bold]")
    console.print()


def print_branch_progress(step: int, total: int, branch_name: str, status: str) -> None:
    """Print progress for branch creation."""
    status_colors = {
        "creating": "yellow",
        "applying": "yellow",
        "verifying": "yellow",
        "pushing": "yellow",
        "done": "green",
        "skipped": "dim",
        "failed": "red",
    }
    color = status_colors.get(status.split()[0].lower(), "white")
    console.print(f"  [{step}/{total}] Creating branch '[bold]{branch_name}[/bold]'")
    console.print(f"        [{color}]{status}[/{color}]")


def print_pr_created(url: str) -> None:
    """Print PR creation message."""
    console.print(f"        Creating PR... [green]done[/green] -> {url}")


def print_verification_result(result: VerificationResult) -> None:
    """Print hash verification result."""
    console.print()
    console.print("[bold]Verifying final state...[/bold]")
    console.print()

    if result.passed:
        console.print("[bold green]HASH CHECK: PASSED[/bold green]")
        console.print()

        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Label")
        table.add_column("Value")
        table.add_row("Original branch:", result.original_hash)
        table.add_row("After split:", result.final_hash)
        table.add_row("", "")
        table.add_row("Status:", "[bold green]IDENTICAL[/bold green]")

        console.print(Panel(table, width=50))
    else:
        console.print("[bold red]HASH CHECK: FAILED[/bold red]")
        console.print()

        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Label")
        table.add_column("Value")
        table.add_row("Expected:", result.original_hash)
        table.add_row("Got:", result.final_hash)

        console.print(Panel(table, width=50))

        if result.differences:
            console.print()
            console.print("[bold]Differences found:[/bold]")
            for diff in result.differences:
                console.print(f"  {diff.get('file', 'unknown')}: {diff.get('description', '')}")


def print_split_complete(intents: list[Intent]) -> None:
    """Print split completion summary."""
    console.print()
    console.print("[bold green]Split complete![/bold green]")
    console.print()
    console.print(f"  Created {len(intents)} PRs:")

    for i, intent in enumerate(intents):
        pr_num = f"#{intent.pr_number}" if intent.pr_number else "(no PR)"
        base = f"(base: #{intents[i-1].pr_number})" if i > 0 and intents[i-1].pr_number else "(base: main)"
        console.print(f"    {pr_num} {intent.name} {base}")


def print_error(message: str) -> None:
    """Print an error message."""
    console.print(f"[bold red]Error:[/bold red] {message}")


def print_warning(message: str) -> None:
    """Print a warning message."""
    console.print(f"[bold yellow]Warning:[/bold yellow] {message}")


def print_info(message: str) -> None:
    """Print an info message."""
    console.print(f"[dim]{message}[/dim]")


def print_success(message: str) -> None:
    """Print a success message."""
    console.print(f"[bold green]{message}[/bold green]")


def print_retry(attempt: int, max_attempts: int, message: str) -> None:
    """Print retry information."""
    console.print()
    console.print(f"[yellow]Retry {attempt}/{max_attempts}:[/yellow] {message}")


def print_backtrack(from_phase: str, to_phase: str, reason: str) -> None:
    """Print backtrack information."""
    console.print()
    console.print(f"[yellow]Backtracking:[/yellow] {from_phase} -> {to_phase}")
    console.print(f"  Reason: {reason}")


def print_cost_summary(tokens: int, cost: float) -> None:
    """Print cost summary."""
    console.print()
    console.print(f"[dim]Tokens used: {tokens:,} | Cost: ${cost:.4f}[/dim]")


def print_session_saved(path: str) -> None:
    """Print session saved message."""
    console.print(f"[dim]Session saved to: {path}[/dim]")


def print_session_resumed(session_id: str) -> None:
    """Print session resumed message."""
    console.print(f"[dim]Resumed session: {session_id}[/dim]")


def print_dry_run_notice() -> None:
    """Print dry run notice."""
    console.print()
    console.print(Panel(
        "[yellow]DRY RUN MODE[/yellow] - No changes will be made",
        style="yellow",
    ))


def print_babysit_question(question: str, options: list[str]) -> str:
    """Print a babysit mode question and get user input."""
    console.print()
    console.print(Panel(question, title="Decision Required", style="yellow"))

    for i, opt in enumerate(options):
        console.print(f"  [{i + 1}] {opt}")

    choice = Prompt.ask("Choose", choices=[str(i + 1) for i in range(len(options))])
    return options[int(choice) - 1]


def print_escape_hatch_prompt() -> str:
    """Print escape hatch prompt."""
    console.print()
    console.print(Panel(
        "Escape Hatch: You can manually edit the intent mapping.\n"
        "Options:\n"
        "  [e] Open editor\n"
        "  [s] Skip and continue\n"
        "  [a] Abort",
        title="Manual Intervention",
        style="cyan",
    ))
    return Prompt.ask("Choose", choices=["e", "s", "a"])
