"""
Output formatting utilities
"""

from datetime import datetime, timedelta
from typing import Any, List, Optional

from rich.text import Text


def format_timestamp(dt: datetime) -> str:
    """Format timestamp for display"""
    now = datetime.now()
    diff = now - dt

    if diff < timedelta(seconds=60):
        return f"{int(diff.total_seconds())}s ago"
    elif diff < timedelta(hours=1):
        return f"{int(diff.total_seconds() // 60)}m ago"
    elif diff < timedelta(days=1):
        return f"{int(diff.total_seconds() // 3600)}h ago"
    elif diff < timedelta(days=7):
        return f"{diff.days}d ago"
    else:
        return dt.strftime("%Y-%m-%d")


def format_proxy_status(status: str) -> Text:
    """Format proxy status with color coding"""
    status_colors = {
        "healthy": "green",
        "degraded": "yellow",
        "unhealthy": "red",
        "unknown": "white",
        "connecting": "blue",
        "disconnected": "red",
    }

    color = status_colors.get(status.lower(), "white")
    return Text(status.title(), style=color)


def format_download_speed(bytes_per_second: float) -> str:
    """Format download speed for display"""
    if bytes_per_second < 1024:
        return f"{bytes_per_second:.0f} B/s"
    elif bytes_per_second < 1024 * 1024:
        return f"{bytes_per_second / 1024:.1f} KB/s"
    else:
        return f"{bytes_per_second / (1024 * 1024):.1f} MB/s"


def format_success_rate(successful: int, total: int) -> Text:
    """Format success rate with color coding"""
    if total == 0:
        return Text("N/A", style="white")

    rate = (successful / total) * 100

    if rate >= 95:
        color = "green"
    elif rate >= 80:
        color = "yellow"
    else:
        color = "red"

    return Text(f"{rate:.1f}%", style=color)


def format_library_name(name: str, max_length: int = 20) -> str:
    """Format library name with truncation if needed"""
    if len(name) <= max_length:
        return name

    return f"{name[: max_length - 3]}..."


def format_config_value(key: str, value: Any) -> str:
    """Format configuration value for display"""
    # Mask sensitive values
    if any(
        sensitive in key.lower() for sensitive in ["password", "secret", "token", "key"]
    ):
        if value and str(value).strip():
            return "***masked***"
        else:
            return "not set"

    if value is None:
        return "not set"
    elif isinstance(value, bool):
        return "enabled" if value else "disabled"
    elif isinstance(value, (list, tuple)):
        return f"[{len(value)} items]"
    elif isinstance(value, dict):
        return f"{{{len(value)} keys}}"
    else:
        return str(value)


def format_error_message(error: Exception, include_type: bool = True) -> str:
    """Format error message for display"""
    error_type = type(error).__name__
    message = str(error)

    if include_type and error_type != "Exception":
        return f"{error_type}: {message}"
    else:
        return message


def format_progress_description(
    stage: str, current: int, total: Optional[int] = None
) -> str:
    """Format progress description based on stage"""
    stage_descriptions = {
        "initializing": "Initializing download engine",
        "context_generation": "Generating documentation contexts",
        "downloading": "Downloading documentation",
        "deduplication": "Processing and deduplicating content",
        "storage": "Storing documentation files",
        "cleanup": "Cleaning up temporary files",
    }

    base_desc = stage_descriptions.get(stage, stage.replace("_", " ").title())

    if total is not None and total > 0:
        return f"{base_desc} ({current}/{total})"
    else:
        return f"{base_desc} ({current})"


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """Truncate text to specified length"""
    if len(text) <= max_length:
        return text

    return text[: max_length - len(suffix)] + suffix


def format_table_value(value: Any, max_length: int = 50) -> str:
    """Format value for table display with truncation"""
    str_value = str(value) if value is not None else "N/A"
    return truncate_text(str_value, max_length)


def colorize_status(status: str) -> Text:
    """Apply color to status based on common status values"""
    status_lower = status.lower()

    if status_lower in ["success", "completed", "healthy", "ok", "enabled"]:
        return Text(status, style="green")
    elif status_lower in ["warning", "degraded", "partial", "slow"]:
        return Text(status, style="yellow")
    elif status_lower in ["error", "failed", "unhealthy", "disabled", "critical"]:
        return Text(status, style="red")
    elif status_lower in ["pending", "waiting", "queued"]:
        return Text(status, style="blue")
    elif status_lower in ["running", "processing", "downloading"]:
        return Text(status, style="cyan")
    else:
        return Text(status, style="white")


def format_command_help(command: str, description: str, examples: List[str]) -> str:
    """Format command help text with examples"""
    help_text = [description]

    if examples:
        help_text.append("")
        help_text.append("Examples:")
        for example in examples:
            help_text.append(f"  {example}")

    return "\n".join(help_text)


def format_validation_error(
    field: str, error: str, suggestions: Optional[List[str]] = None
) -> str:
    """Format validation error message"""
    error_text = [f"Validation error in {field}: {error}"]

    if suggestions:
        error_text.append("")
        error_text.append("Suggestions:")
        for suggestion in suggestions:
            error_text.append(f"  • {suggestion}")

    return "\n".join(error_text)
