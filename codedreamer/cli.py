"""
Command-line interface for CodeDreamer.

Commands:
    index   - Index a codebase for dreaming
    dream   - Run dream generation (once or continuously)
    briefing - Show recent dreams
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

from .config import settings
from .dreamer import Dreamer
from .indexer import CodebaseIndexer

app = typer.Typer(
    name="codedreamer",
    help="Autonomous code improvement suggestions via LLM dreaming.",
    no_args_is_help=True,
)
console = Console()


def setup_logging(verbose: bool = False) -> None:
    """Configure logging with rich output."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_path=False)],
    )


@app.command()
def index(
    path: Path = typer.Argument(..., help="Path to codebase to index"),  # noqa: B008
    clear: bool = typer.Option(False, "--clear", "-c", help="Clear existing index first"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """Index a codebase for dream generation."""
    setup_logging(verbose)

    if not path.exists():
        console.print(f"[red]Error:[/red] Path does not exist: {path}")
        raise typer.Exit(1)

    console.print(Panel(f"[bold]Indexing Codebase[/bold]\n{path}"))

    indexer = CodebaseIndexer()

    if clear:
        console.print("[yellow]Clearing existing index...[/yellow]")
        indexer.clear()

    stats = indexer.index_directory(path)

    # Display results
    table = Table(title="Indexing Complete")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Files processed", str(stats.files_processed))
    table.add_row("Chunks created", str(stats.chunks_created))
    table.add_row("Chunks skipped", str(stats.chunks_skipped))
    table.add_row("Errors", str(len(stats.errors)))

    console.print(table)

    if stats.errors:
        console.print("\n[yellow]Errors:[/yellow]")
        for error in stats.errors[:5]:
            console.print(f"  - {error}")
        if len(stats.errors) > 5:
            console.print(f"  ... and {len(stats.errors) - 5} more")


@app.command()
def dream(
    once: bool = typer.Option(False, "--once", help="Run single dream cycle and exit"),
    iterations: int = typer.Option(5, "--iterations", "-n", help="Max dreams per cycle"),
    interval: int = typer.Option(
        None, "--interval", "-i", help="Seconds between cycles (default: from config)"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """Generate code improvement dreams."""
    setup_logging(verbose)

    console.print(Panel("[bold]CodeDreamer[/bold]\nGenerating improvement suggestions..."))

    dreamer = Dreamer()

    if once:
        # Single cycle
        dreams, stats = dreamer.run_cycle(max_iterations=iterations)
        _display_cycle_results(dreams, stats)
    else:
        # Continuous mode with scheduler
        from apscheduler.schedulers.blocking import BlockingScheduler

        interval_sec = interval or settings.dream_interval_sec

        console.print(
            f"[cyan]Running continuously. "
            f"Interval: {interval_sec}s. Press Ctrl+C to stop.[/cyan]\n"
        )

        def run_and_display() -> None:
            dreams, stats = dreamer.run_cycle(max_iterations=iterations)
            _display_cycle_results(dreams, stats)

        # Run immediately, then on schedule
        run_and_display()

        scheduler = BlockingScheduler()
        scheduler.add_job(run_and_display, "interval", seconds=interval_sec)

        try:
            scheduler.start()
        except KeyboardInterrupt:
            console.print("\n[yellow]Shutting down...[/yellow]")
            scheduler.shutdown()


@app.command()
def briefing(
    hours: int = typer.Option(12, "--hours", "-h", help="Show dreams from last N hours"),
    category: str | None = typer.Option(None, "--category", "-c", help="Filter by category"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """Display recent dreams (morning briefing)."""
    setup_logging(verbose)

    dreams_dir = settings.dreams_dir
    if not dreams_dir.exists():
        console.print("[yellow]No dreams directory found. Run 'codedreamer dream' first.[/yellow]")
        raise typer.Exit(0)

    cutoff = datetime.now() - timedelta(hours=hours)

    # Find recent dream files
    dream_files = sorted(dreams_dir.glob("dream_*.md"), reverse=True)
    recent_dreams = []

    for filepath in dream_files:
        # Parse timestamp from filename: dream_YYYYMMDD_HHMMSS_category.md
        try:
            parts = filepath.stem.split("_")
            ts_str = f"{parts[1]}_{parts[2]}"
            file_time = datetime.strptime(ts_str, "%Y%m%d_%H%M%S")

            if file_time < cutoff:
                continue

            dream_category = "_".join(parts[3:])
            if category and dream_category != category:
                continue

            recent_dreams.append((filepath, file_time, dream_category))
        except (IndexError, ValueError):
            continue

    if not recent_dreams:
        console.print(f"[yellow]No dreams found in the last {hours} hours.[/yellow]")
        raise typer.Exit(0)

    console.print(
        Panel(
            f"[bold]While You Were Away...[/bold]\n"
            f"{len(recent_dreams)} dreams in the last {hours} hours"
        )
    )

    for filepath, file_time, dream_category in recent_dreams:
        content = filepath.read_text()

        # Extract first paragraph after the header
        lines = content.split("\n")
        summary_lines = []
        in_content = False
        for line in lines:
            if line.startswith("---"):
                if in_content:
                    break
                in_content = True
                continue
            if in_content and line.strip():
                summary_lines.append(line)
                if len(summary_lines) >= 5:
                    break

        summary = "\n".join(summary_lines)

        console.print(f"\n[cyan][{dream_category}][/cyan] {file_time.strftime('%H:%M')}")
        console.print(summary[:300] + "..." if len(summary) > 300 else summary)
        console.print(f"[dim]{filepath}[/dim]")


@app.command()
def config() -> None:
    """Display current configuration."""
    table = Table(title="CodeDreamer Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    for field_name, _field_info in settings.model_fields.items():
        value = getattr(settings, field_name)
        table.add_row(field_name, str(value))

    console.print(table)


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(8080, "--port", "-p", help="Port to bind to"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload for development"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """Run the HTTP API server."""
    setup_logging(verbose)

    console.print(
        Panel(
            f"[bold]CodeDreamer Server[/bold]\n"
            f"Starting on http://{host}:{port}\n"
            f"API docs: http://{host}:{port}/docs"
        )
    )

    from .server import run_server

    run_server(host=host, port=port, reload=reload)


@app.command()
def graph(
    action: str = typer.Argument("stats", help="Action: stats, hot, decay, prune, backfill, entangled"),
    limit: int = typer.Option(10, "--limit", "-n", help="Max nodes to show"),
    semantic: bool = typer.Option(False, "--semantic", "-s", help="Include semantic edges (for backfill)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """Manage the knowledge graph."""
    setup_logging(verbose)

    from .graph import get_graph

    kg = get_graph()

    if action == "stats":
        stats = kg.stats()
        table = Table(title="Knowledge Graph Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Total Nodes", str(stats["total_nodes"]))
        table.add_row("Total Edges", str(stats["total_edges"]))
        table.add_row("Avg Entanglement", f"{stats.get('avg_entanglement', 0):.3f}")
        for tier, count in stats["tiers"].items():
            table.add_row(f"Tier: {tier}", str(count))
        for ntype, count in stats["types"].items():
            table.add_row(f"Type: {ntype}", str(count))
        console.print(table)

    elif action == "hot":
        nodes = kg.query_hot(limit=limit)
        if not nodes:
            console.print("[yellow]No nodes in graph.[/yellow]")
            return

        console.print(Panel(f"[bold]Hottest {len(nodes)} Nodes[/bold]"))
        for node in nodes:
            console.print(
                f"[cyan][{node.node_type.name}][/cyan] "
                f"momentum={node.momentum:.2f} tier={node.tier.name}"
            )
            console.print(f"  {node.content[:80]}...")
            console.print()

    elif action == "decay":
        archived = kg.decay_all()
        kg.save()
        console.print(f"[green]Decay applied. {archived} nodes below threshold.[/green]")

    elif action == "prune":
        removed = kg.prune_cold()
        kg.save()
        console.print(f"[green]Pruned {removed} cold nodes.[/green]")

    elif action == "backfill":
        if semantic:
            console.print("[cyan]Backfilling edges (including semantic)...[/cyan]")
        else:
            console.print("[cyan]Backfilling edges for existing nodes...[/cyan]")
        edges_created = kg.backfill_edges(include_semantic=semantic)
        kg.save()
        console.print(f"[green]Backfill complete. {edges_created} edges created.[/green]")

    elif action == "entangled":
        nodes = kg.get_most_entangled(limit=limit)
        if not nodes:
            console.print("[yellow]No nodes in graph.[/yellow]")
            return

        console.print(Panel(f"[bold]Top {len(nodes)} Most Entangled Nodes[/bold]"))
        for node, score in nodes:
            console.print(
                f"[magenta]E={score:.2f}[/magenta] "
                f"[cyan][{node.node_type.name}][/cyan] "
                f"momentum={node.momentum:.2f}"
            )
            console.print(f"  {node.content[:80]}...")
            console.print()

    else:
        console.print(f"[red]Unknown action: {action}[/red]")
        console.print("Available: stats, hot, decay, prune, backfill, entangled")


@app.command()
def daemon(
    codebase: Path = typer.Option(  # noqa: B008
        None, "--codebase", "-c", help="Path to codebase to analyze"
    ),
    interval: int = typer.Option(
        None, "--interval", "-i", help="Seconds between dream cycles"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """
    Run autonomous dream daemon.

    Indexes codebase on startup, then continuously generates
    code improvement suggestions. Runs until interrupted.
    """
    setup_logging(verbose)

    from .daemon import run_daemon

    codebase_path = codebase or settings.codebase_path

    if not codebase_path:
        console.print(
            "[red]Error:[/red] No codebase path provided.\n"
            "Use --codebase or set DREAMER_CODEBASE_PATH environment variable."
        )
        raise typer.Exit(1)

    console.print(
        Panel(
            f"[bold]CodeDreamer Daemon[/bold]\n"
            f"Codebase: {codebase_path}\n"
            f"Interval: {interval or settings.dream_interval_sec}s\n"
            f"Dreams: {settings.dreams_dir}\n\n"
            "[dim]Press Ctrl+C to stop[/dim]"
        )
    )

    run_daemon(codebase_path=codebase_path, dream_interval=interval)


def _display_cycle_results(dreams: list, stats) -> None:  # type: ignore[no-untyped-def]
    """Display results from a dream cycle."""
    if dreams:
        console.print(f"\n[green]Saved {len(dreams)} dreams:[/green]")
        for dream in dreams:
            console.print(
                f"  [{dream.category}] novelty={dream.novelty_score:.2f} "
                f"- {dream.content[:60]}..."
            )
    else:
        console.print("[yellow]No dreams passed validation this cycle.[/yellow]")

    if stats.rejection_reasons:
        console.print("\n[dim]Rejection reasons:[/dim]")
        for reason, count in stats.rejection_reasons.items():
            console.print(f"  [dim]{reason}: {count}[/dim]")


@app.command()
def dreams(
    action: str = typer.Argument("list", help="Action: list, actionable, mark, reflect"),
    dream_id: str = typer.Option(None, "--id", help="Dream ID for mark action"),
    status: str = typer.Option(None, "--status", "-s", help="Status: applied, rejected, deferred"),
    priority: str = typer.Option(None, "--priority", "-p", help="Priority: critical, high, medium, low"),
    min_score: float = typer.Option(0.0, "--min-score", help="Minimum novelty score"),
    limit: int = typer.Option(20, "--limit", "-n", help="Max results to show"),
    reason: str = typer.Option("", "--reason", "-r", help="Reason for rejection/deferral"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """
    Manage and filter dreams.

    Actions:
        list       - List all dreams (with optional filters)
        actionable - Show pending dreams sorted by priority
        mark       - Update dream status (requires --id and --status)
        reflect    - Run reflection cycle to prioritize unranked dreams
    """
    setup_logging(verbose)
    
    from .leaderboard import Priority, ReviewStatus, get_leaderboard

    lb = get_leaderboard()

    if action == "list":
        entries = lb.entries
        
        # Apply filters
        if min_score > 0:
            entries = [e for e in entries if e.novelty_score >= min_score]
        if priority:
            entries = [e for e in entries if e.priority == priority.lower()]
        if status:
            entries = [e for e in entries if e.status == status.lower()]
        
        entries = entries[:limit]
        
        if not entries:
            console.print("[yellow]No dreams match filters.[/yellow]")
            return

        table = Table(title=f"Dreams ({len(entries)} shown)")
        table.add_column("Rank", style="dim", width=5)
        table.add_column("Score", width=6)
        table.add_column("Priority", width=10)
        table.add_column("Status", width=10)
        table.add_column("Source", width=15)
        table.add_column("Summary", width=50)

        priority_colors = {
            "critical": "red bold",
            "high": "yellow",
            "medium": "cyan",
            "low": "dim",
            "unranked": "white",
        }
        status_colors = {
            "pending": "white",
            "applied": "green",
            "rejected": "red dim",
            "deferred": "yellow dim",
        }

        for e in entries:
            p_style = priority_colors.get(e.priority, "white")
            s_style = status_colors.get(e.status, "white")
            summary = e.content[:50].replace("\n", " ") if e.content else ""
            
            table.add_row(
                str(e.rank),
                f"{e.novelty_score:.2f}",
                f"[{p_style}]{e.priority.upper()}[/{p_style}]",
                f"[{s_style}]{e.status}[/{s_style}]",
                e.source_file[:15],
                summary + "...",
            )

        console.print(table)

    elif action == "actionable":
        entries = lb.get_actionable()[:limit]
        
        if not entries:
            console.print("[green]No pending dreams![/green]")
            return

        console.print(Panel("[bold]Actionable Dreams[/bold] (sorted by priority)"))
        
        for e in entries:
            p_color = {"critical": "red", "high": "yellow", "medium": "cyan", "low": "dim"}.get(e.priority, "white")
            console.print(
                f"[{p_color}][{e.priority.upper()}][/{p_color}] "
                f"[cyan]{e.dream_id}[/cyan] "
                f"({e.source_file}) "
                f"score={e.novelty_score:.2f}"
            )
            # Show first 100 chars of content
            summary = e.content[:100].replace("\n", " ") if e.content else ""
            console.print(f"  [dim]{summary}...[/dim]\n")

    elif action == "mark":
        if not dream_id:
            console.print("[red]Error: --id required for mark action[/red]")
            raise typer.Exit(1)
        if not status:
            console.print("[red]Error: --status required for mark action[/red]")
            raise typer.Exit(1)

        try:
            review_status = ReviewStatus(status.lower())
        except ValueError:
            console.print(f"[red]Invalid status: {status}[/red]")
            console.print("Valid: pending, applied, rejected, deferred")
            raise typer.Exit(1)

        if lb.update_status(dream_id, review_status, reason):
            console.print(f"[green]Updated {dream_id} → {status}[/green]")
        else:
            console.print(f"[red]Dream not found: {dream_id}[/red]")

    elif action == "reflect":
        from .models import get_orchestra
        from .reflection import get_reflection_cycle

        console.print("[cyan]Running reflection cycle...[/cyan]")
        orchestra = get_orchestra()
        reflection = get_reflection_cycle(orchestra)
        prioritized = reflection.run(batch_size=limit)
        console.print(f"[green]Prioritized {prioritized} dreams[/green]")

    else:
        console.print(f"[red]Unknown action: {action}[/red]")
        console.print("Available: list, actionable, mark, reflect")


if __name__ == "__main__":
    app()

