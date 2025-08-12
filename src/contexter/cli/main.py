"""
Main CLI entry point for Contexter
"""

import logging
import sys

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install

# Install rich traceback handler for better error display
install(show_locals=True)

# Initialize console
console = Console()

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)],
)


@click.group()
@click.version_option(version="1.0.0", prog_name="contexter")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option("--no-color", is_flag=True, help="Disable colored output")
@click.pass_context
def cli(ctx: click.Context, verbose: bool, no_color: bool) -> None:
    """
    Contexter - Documentation Downloader

    High-performance CLI for comprehensive documentation retrieval using
    intelligent proxy rotation and advanced deduplication.

    Examples:
      contexter download fastapi                    # Download FastAPI docs
      contexter download django --output ./docs    # Download to specific directory
      contexter config wizard                      # Interactive configuration setup
      contexter status                            # Check system health
    """
    ctx.ensure_object(dict)

    # Configure console
    if no_color:
        ctx.obj["console"] = Console(force_terminal=False, no_color=True)
    else:
        ctx.obj["console"] = console

    # Configure logging level
    if verbose:
        logging.getLogger().setLevel(logging.INFO)
        logging.getLogger("contexter").setLevel(logging.DEBUG)
        ctx.obj["verbose"] = True
    else:
        ctx.obj["verbose"] = False

    # Store global options
    ctx.obj["no_color"] = no_color


# Import and register commands at module level to support testing
from .commands import config, download, search, status  # noqa: E402

cli.add_command(download.download)
cli.add_command(search.search)
cli.add_command(config.config)
cli.add_command(status.status)


def main() -> None:
    """Main entry point for the CLI application"""
    try:
        # Run CLI (commands already registered)
        cli()

    except KeyboardInterrupt:
        console.print("\n[yellow]Operation interrupted by user[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        if "--verbose" in sys.argv or "-v" in sys.argv:
            console.print_exception()
        sys.exit(1)


if __name__ == "__main__":
    main()
