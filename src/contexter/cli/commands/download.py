"""
Download command implementation with Rich progress visualization
"""

import os
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.live import Live

from ...core.config_manager import ConfigurationManager
from ...core.download_engine import AsyncDownloadEngine
from ...integration.context7_client import Context7Client
from ...integration.proxy_manager import BrightDataProxyManager
from ...models.download_models import DownloadRequest
from ..ui.display import create_error_display
from ..ui.progress import DownloadProgressTracker, create_dry_run_display
from ..utils.async_runner import GracefulKiller, async_command
from ..utils.validation import (
    get_validation_suggestions,
    show_validation_error,
    validate_concurrency_limit,
    validate_library_name,
    validate_output_directory,
    validate_proxy_mode,
)


@click.command()
@click.argument("library", required=True)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default=".",
    help="Output directory for downloaded documentation",
)
@click.option(
    "--max-concurrent",
    "-c",
    type=int,
    default=10,
    help="Maximum concurrent download connections (1-50)",
)
@click.option(
    "--proxy-mode",
    type=click.Choice(["auto", "brightdata", "none"]),
    default="auto",
    help="Proxy configuration mode",
)
@click.option(
    "--dry-run", is_flag=True, help="Show what would be downloaded without executing"
)
@click.option(
    "--force",
    is_flag=True,
    help="Force re-download even if cached documentation exists",
)
@click.option("--contexts", type=int, help="Override number of contexts to generate")
@click.pass_context
@async_command
async def download(
    ctx: click.Context,
    library: str,
    output: str,
    max_concurrent: int,
    proxy_mode: str,
    dry_run: bool,
    force: bool,
    contexts: Optional[int],
) -> None:
    """
    Download comprehensive documentation for the specified library.

    LIBRARY is the name of the library to download (e.g., 'fastapi', 'django', 'react').

    The command will:
    1. Generate multiple documentation contexts
    2. Download documentation using proxy rotation
    3. Deduplicate and merge content
    4. Store compressed documentation locally

    Examples:
      contexter download fastapi
      contexter download django --output ./docs --max-concurrent 5
      contexter download react --proxy-mode brightdata --force
    """
    console: Console = ctx.obj["console"]
    verbose: bool = ctx.obj.get("verbose", False)

    # Validate inputs
    is_valid, error_msg = validate_library_name(library)
    if not is_valid:
        suggestions = get_validation_suggestions("library_name", library)
        show_validation_error(console, error_msg or "", suggestions)
        ctx.exit(2)

    is_valid, error_msg, output_path = validate_output_directory(output)
    if not is_valid:
        suggestions = get_validation_suggestions("output_directory", output)
        show_validation_error(console, error_msg or "", suggestions)
        ctx.exit(2)

    is_valid, error_msg = validate_concurrency_limit(max_concurrent)
    if not is_valid:
        suggestions = get_validation_suggestions("concurrency", str(max_concurrent))
        show_validation_error(console, error_msg or "", suggestions)
        ctx.exit(2)

    is_valid, error_msg = validate_proxy_mode(proxy_mode)
    if not is_valid:
        suggestions = get_validation_suggestions("proxy_mode", proxy_mode)
        show_validation_error(console, error_msg or "", suggestions)
        ctx.exit(2)

    # Set up graceful shutdown
    killer = GracefulKiller()

    try:
        # Initialize components
        console.print(f"[cyan]Initializing download for library: {library}[/cyan]")

        config_manager = ConfigurationManager()

        # Validate configuration
        try:
            config = await config_manager.load_config()
        except Exception as e:
            console.print(create_error_display(e, "Configuration Error"))
            console.print(
                "\n[yellow]Run 'contexter config wizard' to set up configuration[/yellow]"
            )
            ctx.exit(1)

        # Initialize proxy manager if credentials available
        proxy_manager = None
        if config.proxy.customer_id:
            brightdata_password = os.getenv("BRIGHTDATA_PASSWORD")
            if brightdata_password:
                proxy_manager = BrightDataProxyManager(
                    customer_id=config.proxy.customer_id,
                    password=brightdata_password,
                    zone_name=config.proxy.zone_name,
                    dns_resolution=config.proxy.dns_resolution,
                )
                # Initialize proxy pool
                console.print("[cyan]Initializing proxy connections...[/cyan]")
                await proxy_manager.initialize_pool(pool_size=config.proxy.pool_size)

        # Initialize download engine
        download_engine = AsyncDownloadEngine(
            proxy_manager=proxy_manager, max_concurrent=max_concurrent
        )

        # Check for dry run
        if dry_run:
            if output_path:
                await _show_dry_run_info(
                    console, library, download_engine, output_path, contexts
                )
            return

        # Execute download with progress tracking
        if output_path:
            await _execute_download_with_progress(
                console=console,
                download_engine=download_engine,
                library=library,
                output_path=output_path,
                verbose=verbose,
                force=force,
                contexts=contexts,
                killer=killer,
            )

    except KeyboardInterrupt:
        console.print("\n[yellow]Download interrupted by user[/yellow]")
        ctx.exit(130)
    except Exception as e:
        console.print(create_error_display(e, "Download Error"))
        if verbose:
            console.print_exception()
        ctx.exit(1)
    finally:
        # Clean up resources
        if proxy_manager:
            await proxy_manager.shutdown()


async def _show_dry_run_info(
    console: Console,
    library: str,
    download_engine: AsyncDownloadEngine,
    output_path: Path,
    contexts: Optional[int],
) -> None:
    """Show dry-run information without executing download"""

    try:
        # Get Context7 client for estimation
        context7_client = Context7Client()

        with console.status(f"[cyan]Analyzing library: {library}[/cyan]"):
            # Search for library to validate it exists
            search_results = await context7_client.resolve_library_id(library)

            if not search_results:
                console.print(f"[red]Library '{library}' not found[/red]")
                return

            # Use first search result (currently placeholder)
            # library_info = search_results[0]

            # Estimate contexts
            estimated_contexts = contexts or 15  # Default estimate

            # Estimate size (rough calculation)
            estimated_size_mb = estimated_contexts * 2.5  # ~2.5MB per context
            estimated_size = f"{estimated_size_mb:.1f} MB"

        # Display dry-run information
        dry_run_panel = create_dry_run_display(
            library_name=library,
            contexts=estimated_contexts,
            estimated_size=estimated_size,
            output_path=str(output_path),
        )

        console.print(dry_run_panel)

    except Exception as e:
        console.print(f"[red]Error during dry-run analysis: {e}[/red]")


async def _execute_download_with_progress(
    console: Console,
    download_engine: AsyncDownloadEngine,
    library: str,
    output_path: Path,
    verbose: bool,
    force: bool,
    contexts: Optional[int],
    killer: GracefulKiller,
) -> None:
    """Execute download with Rich progress visualization"""

    progress_tracker = DownloadProgressTracker(console, library)

    # Register cleanup function
    killer.register_cleanup(lambda: console.print("[yellow]Cleaning up...[/yellow]"))

    try:
        with Live(
            progress_tracker.progress_manager.progress,
            console=console,
            refresh_per_second=10,
        ):
            # Create download request
            download_request = DownloadRequest(
                library_id=library,
                max_contexts=contexts or 7,
                timeout_seconds=30.0,
                retry_count=3,
                metadata={"output_directory": str(output_path), "force_refresh": force},
            )

            # Execute download
            result = await download_engine.download_library(
                request=download_request,
                progress_callback=None,  # progress_tracker.progress_callback
            )

            # Check for interruption
            if killer.kill_now:
                console.print(
                    "\n[yellow]Download interrupted, partial results may be available[/yellow]"
                )
                return

        # Store documentation to disk
        from ...core.simple_storage import SimpleStorageManager
        
        if result.chunks:
            console.print("\n[cyan]Storing documentation to disk...[/cyan]")
            
            storage_manager = SimpleStorageManager(str(output_path))
            storage_result = await storage_manager.store_documentation(
                library_id=library,
                chunks=result.chunks
            )
            
            if storage_result.success:
                console.print(f"[green]✅ Documentation saved successfully![/green]")
                console.print(f"   File: {storage_result.file_path}")
                console.print(f"   Size: {storage_result.compressed_size:,} bytes")
                console.print(f"   Total tokens: {result.total_tokens:,}")
            else:
                console.print(f"[red]❌ Storage failed: {storage_result.error}[/red]")
        
        # Show completion summary
        result_dict = {
            "successful_contexts": result.successful_contexts,
            "failed_contexts": result.failed_contexts,
            "total_tokens": result.total_tokens,
            "total_download_time": result.total_download_time,
        }
        progress_tracker.show_completion_summary(result_dict)

        if (
            verbose
            and hasattr(result, "metadata")
            and result.metadata
            and "files" in result.metadata
        ):
            files = result.metadata["files"]
            console.print(f"Files created: {len(files)}")
            for file_path in files[:5]:  # Show first 5 files
                console.print(f"  • {file_path}")
            if len(files) > 5:
                console.print(f"  • ... and {len(files) - 5} more files")

    except Exception as e:
        progress_tracker.show_error_summary(e)
        raise


# Helper functions for command examples
def _get_command_examples() -> list[str]:
    """Get example commands for help display"""
    return [
        "contexter download fastapi",
        "contexter download django --output ./docs",
        "contexter download react --max-concurrent 5",
        "contexter download vue --proxy-mode brightdata",
        "contexter download pandas --dry-run",
        "contexter download numpy --force --verbose",
    ]


# Command examples are defined in the help text above
