"""
Rich progress tracking components
"""

from datetime import datetime
from typing import Any, Dict, Optional

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


class ProgressManager:
    """
    Manages multiple progress bars and status displays
    """

    def __init__(self, console: Console):
        self.console = console
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(elapsed_when_finished=True),
            console=console,
            expand=True,
        )
        self.tasks: Dict[str, TaskID] = {}
        self.status_info: Dict[str, str] = {}

    def add_task(
        self, name: str, description: str, total: Optional[int] = None
    ) -> TaskID:
        """Add a new progress task"""
        task_id = self.progress.add_task(description, total=total)
        self.tasks[name] = task_id
        return task_id

    def update_task(
        self,
        name: str,
        completed: Optional[int] = None,
        total: Optional[int] = None,
        description: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Update an existing task"""
        if name not in self.tasks:
            return

        task_id = self.tasks[name]
        if completed is not None:
            self.progress.update(task_id, completed=completed)
        if total is not None:
            self.progress.update(task_id, total=total)
        if description is not None:
            self.progress.update(task_id, description=description)
        # Handle any additional kwargs
        for key, value in kwargs.items():
            if key == "advance":
                self.progress.update(task_id, advance=value)

    def complete_task(self, name: str) -> None:
        """Mark a task as completed"""
        if name in self.tasks:
            self.progress.update(self.tasks[name], completed=True)

    def set_status(self, key: str, value: str) -> None:
        """Set status information"""
        self.status_info[key] = value

    def get_status_panel(self) -> Panel:
        """Get status information as a Rich panel"""
        if not self.status_info:
            return Panel("No status information", title="Status")

        status_text = "\n".join([f"{k}: {v}" for k, v in self.status_info.items()])
        return Panel(status_text, title="Status", border_style="blue")


class DownloadProgressTracker:
    """
    Specialized progress tracker for download operations
    """

    def __init__(self, console: Console, library_name: str):
        self.console = console
        self.library_name = library_name
        self.progress_manager = ProgressManager(console)
        self.start_time = datetime.now()

        # Create progress tasks
        self.main_task = self.progress_manager.add_task(
            "main", f"Downloading {library_name}", total=100
        )

        self.context_task = self.progress_manager.add_task(
            "context", "Generating contexts", total=None
        )

        self.proxy_task = self.progress_manager.add_task(
            "proxy", "Proxy status", total=None
        )

    async def progress_callback(
        self,
        stage: str,
        current: int,
        total: int,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Progress callback for download operations
        """
        details = details or {}

        if stage == "context_generation":
            if total > 0:
                self.progress_manager.update_task(
                    "context",
                    completed=current,
                    total=total,
                    description=f"Generated {current}/{total} contexts",
                )
            else:
                self.progress_manager.update_task(
                    "context", description=f"Generated {current} contexts"
                )

        elif stage == "downloading":
            progress_percent = (current / total * 100) if total > 0 else 0

            # Update main progress
            self.progress_manager.update_task(
                "main", completed=int(progress_percent), total=100
            )

            # Add speed information
            speed = details.get("speed", 0)
            if speed > 0:
                speed_text = f" ({speed:.1f} MB/s)"
            else:
                speed_text = ""

            self.progress_manager.update_task(
                "main", description=f"Downloading {self.library_name}{speed_text}"
            )

        elif stage == "proxy_status":
            status = details.get("status", "unknown")
            health = details.get("health", "unknown")

            status_color = {
                "healthy": "green",
                "degraded": "yellow",
                "unhealthy": "red",
                "unknown": "white",
            }.get(health, "white")

            self.progress_manager.update_task(
                "proxy", description=f"[{status_color}]Proxy: {status}[/{status_color}]"
            )

        elif stage == "deduplication":
            self.progress_manager.set_status(
                "Deduplication", f"{current}/{total} chunks processed"
            )

        elif stage == "storage":
            self.progress_manager.set_status(
                "Storage", details.get("status", "Writing")
            )

    def show_completion_summary(self, result: Dict[str, Any]) -> None:
        """Show download completion summary"""
        duration = datetime.now() - self.start_time

        # Create summary panel
        summary_text = []
        summary_text.append(f"Library: {self.library_name}")
        summary_text.append(f"Duration: {duration.total_seconds():.1f}s")

        if "files_downloaded" in result:
            summary_text.append(f"Files downloaded: {result['files_downloaded']}")

        if "total_size" in result:
            size_mb = result["total_size"] / (1024 * 1024)
            summary_text.append(f"Total size: {size_mb:.1f} MB")

        if "deduplication_ratio" in result:
            summary_text.append(f"Deduplication: {result['deduplication_ratio']:.1f}%")

        if "success_rate" in result:
            summary_text.append(f"Success rate: {result['success_rate']:.1f}%")

        summary_panel = Panel(
            "\n".join(summary_text),
            title="[green]Download Complete[/green]",
            border_style="green",
        )

        self.console.print(summary_panel)

    def show_error_summary(self, error: Exception) -> None:
        """Show download error summary"""
        duration = datetime.now() - self.start_time

        error_text = []
        error_text.append(f"Library: {self.library_name}")
        error_text.append(f"Duration: {duration.total_seconds():.1f}s")
        error_text.append(f"Error: {str(error)}")

        # Add resolution suggestions based on error type
        if "proxy" in str(error).lower():
            error_text.append("\nSuggestions:")
            error_text.append("• Check proxy configuration with 'c7doc config show'")
            error_text.append("• Try with '--proxy-mode none' to bypass proxy")
            error_text.append("• Verify BrightData credentials")

        elif "network" in str(error).lower():
            error_text.append("\nSuggestions:")
            error_text.append("• Check internet connectivity")
            error_text.append(
                "• Try again with reduced concurrency (--max-concurrent 2)"
            )
            error_text.append("• Verify library name is correct")

        elif "permission" in str(error).lower():
            error_text.append("\nSuggestions:")
            error_text.append("• Check write permissions for output directory")
            error_text.append("• Try with a different output directory")
            error_text.append("• Run with appropriate user privileges")

        error_panel = Panel(
            "\n".join(error_text),
            title="[red]Download Failed[/red]",
            border_style="red",
        )

        self.console.print(error_panel)


def create_dry_run_display(
    library_name: str, contexts: int, estimated_size: str, output_path: str
) -> Panel:
    """Create dry-run information display"""

    dry_run_text = []
    dry_run_text.append(f"Library: {library_name}")
    dry_run_text.append(f"Contexts to generate: {contexts}")
    dry_run_text.append(f"Estimated download size: {estimated_size}")
    dry_run_text.append(f"Output directory: {output_path}")
    dry_run_text.append("")
    dry_run_text.append("Use the command without --dry-run to proceed with download.")

    return Panel(
        "\n".join(dry_run_text),
        title="[blue]Dry Run - What Would Be Downloaded[/blue]",
        border_style="blue",
    )
