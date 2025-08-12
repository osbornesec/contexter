"""
Rich display components for status and formatting
"""

from typing import Any, Dict, List, Optional

from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree


def create_config_table(
    config_data: Dict[str, Any], title: str = "Configuration"
) -> Table:
    """
    Create a Rich table for configuration display
    """
    table = Table(title=title, show_header=True, header_style="bold blue")
    table.add_column("Setting", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")
    table.add_column("Source", style="yellow")

    for key, value_info in config_data.items():
        # Handle different value info formats
        if isinstance(value_info, dict):
            value = value_info.get("value", "not set")
            source = value_info.get("source", "unknown")
        else:
            value = str(value_info) if value_info is not None else "not set"
            source = "config file"

        # Mask sensitive values
        if any(
            sensitive in key.lower()
            for sensitive in ["password", "secret", "token", "key"]
        ):
            if value and value != "not set":
                display_value = "***masked***"
            else:
                display_value = "not set"
        else:
            display_value = str(value)

        table.add_row(key, display_value, source)

    return table


def create_status_panel(component: str, status_info: Dict[str, Any]) -> Panel:
    """
    Create a status panel for a system component
    """
    is_healthy = status_info.get("healthy", False)

    # Determine colors based on health status
    if is_healthy:
        color = "green"
        status_text = "✓ Healthy"
    else:
        color = "red"
        status_text = "✗ Unhealthy"

    # Build content
    content_lines = [f"[{color}]{status_text}[/{color}]"]

    # Add details if available
    if "version" in status_info:
        content_lines.append(f"Version: {status_info['version']}")

    if "last_check" in status_info:
        content_lines.append(f"Last check: {status_info['last_check']}")

    if "details" in status_info and status_info["details"]:
        content_lines.append("")
        content_lines.append(status_info["details"])

    # Add error information for unhealthy components
    if not is_healthy and "error" in status_info:
        content_lines.append("")
        content_lines.append(f"[red]Error: {status_info['error']}[/red]")

    return Panel(
        "\n".join(content_lines), title=component, border_style=color, padding=(0, 1)
    )


def create_stats_display(stats: Dict[str, Any]) -> Panel:
    """
    Create statistics display panel
    """
    stats_text = []

    # Download statistics
    if "downloads" in stats:
        downloads = stats["downloads"]
        stats_text.append(f"Total downloads: {downloads.get('total', 0)}")
        stats_text.append(f"Successful: {downloads.get('successful', 0)}")
        stats_text.append(f"Failed: {downloads.get('failed', 0)}")

        if downloads.get("total", 0) > 0:
            success_rate = (downloads.get("successful", 0) / downloads["total"]) * 100
            stats_text.append(f"Success rate: {success_rate:.1f}%")

    stats_text.append("")

    # Storage statistics
    if "storage" in stats:
        storage = stats["storage"]
        stats_text.append(f"Total libraries: {storage.get('libraries_count', 0)}")

        total_size_mb = storage.get("total_size_mb", 0)
        stats_text.append(f"Total size: {total_size_mb:.1f} MB")

        if "compression_ratio" in storage:
            stats_text.append(f"Compression ratio: {storage['compression_ratio']:.1f}%")

    stats_text.append("")

    # Performance statistics
    if "performance" in stats:
        perf = stats["performance"]
        stats_text.append(
            f"Average download time: {perf.get('avg_download_time', 0):.1f}s"
        )
        stats_text.append(f"Average speed: {perf.get('avg_speed_mbps', 0):.1f} MB/s")

    return Panel(
        "\n".join(stats_text), title="Download Statistics", border_style="cyan"
    )


def create_library_tree(libraries: List[Dict[str, Any]]) -> Tree:
    """
    Create a tree display of downloaded libraries
    """
    tree = Tree("Downloaded Libraries")

    for library in libraries:
        name = library.get("name", "Unknown")

        # Create library branch
        library_branch = tree.add(f"[bold cyan]{name}[/bold cyan]")

        # Add metadata
        if "version" in library:
            library_branch.add(f"Version: {library['version']}")

        if "download_date" in library:
            library_branch.add(f"Downloaded: {library['download_date']}")

        if "size_mb" in library:
            library_branch.add(f"Size: {library['size_mb']:.1f} MB")

        if "contexts_count" in library:
            library_branch.add(f"Contexts: {library['contexts_count']}")

        # Add files if available
        if "files" in library and library["files"]:
            files_branch = library_branch.add("Files")
            for file_info in library["files"][:5]:  # Show first 5 files
                files_branch.add(file_info.get("path", "Unknown"))

            if len(library["files"]) > 5:
                files_branch.add(f"... and {len(library['files']) - 5} more files")

    return tree


def create_error_display(error: Exception, context: Optional[str] = None) -> Panel:
    """
    Create formatted error display with suggestions
    """
    error_lines = []

    if context:
        error_lines.append(f"Context: {context}")
        error_lines.append("")

    error_lines.append(f"Error: {str(error)}")
    error_lines.append("")

    # Add specific suggestions based on error type
    error_type = type(error).__name__.lower()
    suggestions = []

    if "proxy" in error_type or "proxy" in str(error).lower():
        suggestions.extend(
            [
                "Check proxy configuration: c7doc config show",
                "Verify BrightData credentials in environment variables",
                "Try bypassing proxy: --proxy-mode none",
                "Check network connectivity",
            ]
        )

    elif "network" in error_type or "connection" in str(error).lower():
        suggestions.extend(
            [
                "Check internet connectivity",
                "Verify library name is correct",
                "Try with reduced concurrency: --max-concurrent 2",
                "Check if Context7 API is accessible",
            ]
        )

    elif "permission" in str(error).lower() or "access" in str(error).lower():
        suggestions.extend(
            [
                "Check write permissions for output directory",
                "Try with a different output directory",
                "Run command with appropriate user privileges",
                "Verify parent directory exists",
            ]
        )

    elif "config" in error_type or "config" in str(error).lower():
        suggestions.extend(
            [
                "Run configuration wizard: c7doc config wizard",
                "Check configuration file: c7doc config show",
                "Set missing configuration values",
                "Verify environment variables",
            ]
        )

    elif "timeout" in str(error).lower():
        suggestions.extend(
            [
                "Try with reduced concurrency: --max-concurrent 2",
                "Check network stability",
                "Retry the operation",
                "Use --verbose to see detailed progress",
            ]
        )

    else:
        suggestions.extend(
            [
                "Run with --verbose for detailed error information",
                "Check system status: c7doc status",
                "Verify configuration: c7doc config show",
                "Try with different options",
            ]
        )

    if suggestions:
        error_lines.append("Suggestions:")
        for suggestion in suggestions:
            error_lines.append(f"  • {suggestion}")

    return Panel(
        "\n".join(error_lines),
        title="[red]Error[/red]",
        border_style="red",
        padding=(1, 2),
    )


def create_help_display(command: str, examples: List[str]) -> Panel:
    """
    Create help display with examples
    """
    help_lines = [f"Usage examples for '{command}':"]
    help_lines.append("")

    for example in examples:
        help_lines.append(f"  {example}")

    return Panel("\n".join(help_lines), title="Examples", border_style="blue")


def format_duration(seconds: float) -> str:
    """Format duration in a human-readable way"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def format_size(bytes_size: int) -> str:
    """Format file size in a human-readable way"""
    size_float = float(bytes_size)
    for unit in ["B", "KB", "MB", "GB"]:
        if size_float < 1024:
            return f"{size_float:.1f} {unit}"
        size_float /= 1024
    return f"{size_float:.1f} TB"
