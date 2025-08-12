"""
Input validation utilities for CLI commands
"""

import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

from rich.console import Console


def validate_library_name(library_name: str) -> Tuple[bool, Optional[str]]:
    """
    Validate library name format

    Returns:
        (is_valid, error_message)
    """
    if not library_name:
        return False, "Library name cannot be empty"

    if len(library_name) > 100:
        return False, "Library name too long (max 100 characters)"

    # Allow alphanumeric, hyphens, underscores, dots, and forward slashes for Context7 format
    if not re.match(r"^[a-zA-Z0-9._/-]+$", library_name):
        return (
            False,
            "Library name contains invalid characters (allowed: a-z, A-Z, 0-9, -, _, ., /)",
        )

    return True, None


def validate_output_directory(
    output_path: str,
) -> Tuple[bool, Optional[str], Optional[Path]]:
    """
    Validate and create output directory if needed

    Returns:
        (is_valid, error_message, resolved_path)
    """
    try:
        path = Path(output_path).expanduser().resolve()

        # Check if parent directory exists
        if not path.parent.exists():
            return False, f"Parent directory does not exist: {path.parent}", None

        # Check write permissions on parent directory
        if not os.access(path.parent, os.W_OK):
            return False, f"No write permission for directory: {path.parent}", None

        # Create directory if it doesn't exist
        if not path.exists():
            try:
                path.mkdir(parents=True, exist_ok=True)
            except PermissionError:
                return (
                    False,
                    f"Cannot create directory (permission denied): {path}",
                    None,
                )
            except Exception as e:
                return False, f"Cannot create directory: {e}", None

        # Check if it's a directory
        if path.exists() and not path.is_dir():
            return False, f"Path exists but is not a directory: {path}", None

        return True, None, path

    except Exception as e:
        return False, f"Invalid path: {e}", None


def validate_concurrency_limit(max_concurrent: int) -> Tuple[bool, Optional[str]]:
    """
    Validate concurrency limit

    Returns:
        (is_valid, error_message)
    """
    if max_concurrent < 1:
        return False, "Maximum concurrent connections must be at least 1"

    if max_concurrent > 50:
        return False, "Maximum concurrent connections cannot exceed 50"

    return True, None


def validate_proxy_mode(proxy_mode: str) -> Tuple[bool, Optional[str]]:
    """
    Validate proxy mode selection

    Returns:
        (is_valid, error_message)
    """
    valid_modes = {"auto", "brightdata", "none"}

    if proxy_mode not in valid_modes:
        return False, f"Invalid proxy mode. Must be one of: {', '.join(valid_modes)}"

    return True, None


def validate_config_key(key: str) -> Tuple[bool, Optional[str]]:
    """
    Validate configuration key format

    Returns:
        (is_valid, error_message)
    """
    if not key:
        return False, "Configuration key cannot be empty"

    # Allow dot notation for nested keys, but don't allow leading/trailing dots or consecutive dots
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9_]*(?:\.[a-zA-Z][a-zA-Z0-9_]*)*$", key):
        return (
            False,
            "Invalid key format (must start with letter, use dot notation for nested keys, no trailing dots)",
        )

    return True, None


def show_validation_error(
    console: Console, error_message: str, suggestions: Optional[List[str]] = None
) -> None:
    """
    Display validation error with helpful suggestions
    """
    console.print(f"[red]Validation Error:[/red] {error_message}")

    if suggestions:
        console.print("\n[yellow]Suggestions:[/yellow]")
        for suggestion in suggestions:
            console.print(f"  • {suggestion}")


def get_validation_suggestions(error_type: str, value: str) -> List[str]:
    """
    Get validation suggestions based on error type
    """
    suggestions = []

    if error_type == "library_name":
        suggestions.extend(
            [
                "Use lowercase letters and hyphens (e.g., 'fast-api' instead of 'FastAPI')",
                "Check the library name on the package index",
                "Remove any special characters except hyphens, underscores, and dots",
            ]
        )

    elif error_type == "output_directory":
        suggestions.extend(
            [
                "Ensure the parent directory exists",
                "Check that you have write permissions",
                "Use absolute paths to avoid confusion",
                f"Try creating the directory manually: mkdir -p {value}",
            ]
        )

    elif error_type == "concurrency":
        suggestions.extend(
            [
                "Use a value between 1 and 50",
                "Start with a lower value (5-10) for testing",
                "Higher values may trigger rate limiting",
            ]
        )

    elif error_type == "proxy_mode":
        suggestions.extend(
            [
                "Use 'auto' for automatic proxy selection",
                "Use 'brightdata' to force BrightData proxy usage",
                "Use 'none' to disable proxy usage",
            ]
        )

    elif error_type == "config_key":
        suggestions.extend(
            [
                "Use dot notation for nested keys (e.g., 'brightdata.customer_id')",
                "Start with a letter",
                "Use only letters, numbers, dots, and underscores",
            ]
        )

    return suggestions
