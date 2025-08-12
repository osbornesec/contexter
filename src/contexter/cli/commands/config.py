"""
Configuration management commands
"""

import os
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm

from ...core.config_manager import ConfigurationError, ConfigurationManager
from ..ui.display import create_config_table, create_error_display
from ..utils.async_runner import async_command
from ..utils.validation import (
    get_validation_suggestions,
    show_validation_error,
    validate_config_key,
)


@click.group()
def config() -> None:
    """
    Configuration management commands.

    Manage C7DocDownloader configuration including BrightData proxy settings,
    storage paths, performance parameters, and other system settings.
    """
    pass


@config.command()
@click.pass_context
@async_command
async def show(ctx: click.Context) -> None:
    """
    Display current configuration with sources.

    Shows all configuration values including those from:
    - Configuration file
    - Environment variables
    - Default values

    Sensitive values (passwords, tokens) are masked for security.
    """
    console: Console = ctx.obj["console"]

    try:
        config_manager = ConfigurationManager()

        # Load configuration
        config = await config_manager.load_config()

        # Build display data with sources
        config_data = {}

        # BrightData proxy settings
        config_data["proxy.customer_id"] = {
            "value": config.proxy.customer_id,
            "source": _get_value_source(
                "BRIGHTDATA_CUSTOMER_ID", config.proxy.customer_id
            ),
        }
        config_data["proxy.zone_name"] = {
            "value": config.proxy.zone_name,
            "source": "config file",
        }
        config_data["proxy.pool_size"] = {
            "value": str(config.proxy.pool_size),
            "source": "config file",
        }

        # Download settings
        config_data["download.max_concurrent"] = {
            "value": str(config.download.max_concurrent),
            "source": _get_value_source(
                "CONTEXTER_MAX_CONCURRENT", str(config.download.max_concurrent)
            ),
        }
        config_data["download.request_timeout"] = {
            "value": str(config.download.request_timeout),
            "source": "config file",
        }

        # Storage settings
        config_data["storage.base_path"] = {
            "value": config.storage.base_path,
            "source": _get_value_source(
                "CONTEXTER_STORAGE_PATH", str(config.storage.base_path)
            ),
        }
        config_data["storage.compression_level"] = {
            "value": str(config.storage.compression_level),
            "source": "config file",
        }

        # Create and display table
        table = create_config_table(config_data, "C7DocDownloader Configuration")
        console.print(table)

        # Show config file location
        config_file_path = config_manager.config_path
        console.print(f"\n[dim]Configuration file: {config_file_path}[/dim]")

        if config_file_path and not config_file_path.exists():
            console.print(
                "[yellow]Configuration file does not exist. Run 'contexter config wizard' to create one.[/yellow]"
            )

    except ConfigurationError as e:
        console.print(create_error_display(e, "Configuration Error"))
        ctx.exit(1)
    except Exception as e:
        console.print(create_error_display(e, "Unexpected Error"))
        ctx.exit(1)


@config.command()
@click.option(
    "--key", required=True, help="Configuration key to set (e.g., proxy.customer_id)"
)
@click.option("--value", required=True, help="Configuration value")
@click.pass_context
@async_command
async def set(ctx: click.Context, key: str, value: str) -> None:
    """
    Set a configuration value.

    Use dot notation for nested keys:
    - brightdata.customer_id
    - brightdata.password
    - storage.base_path
    - performance.max_concurrent

    Examples:
      contexter config set --key brightdata.customer_id --value your_customer_id
      contexter config set --key storage.base_path --value ~/.contexter/downloads
    """
    console: Console = ctx.obj["console"]

    # Validate key format
    is_valid, error_msg = validate_config_key(key)
    if not is_valid:
        suggestions = get_validation_suggestions("config_key", key)
        show_validation_error(
            console, error_msg or "Invalid configuration key", suggestions
        )
        ctx.exit(2)

    try:
        config_manager = ConfigurationManager()
        await config_manager.load_config()

        # For now, show message that dynamic config changes aren't supported
        console.print(
            "[yellow]Dynamic configuration updates will be supported in a future version.[/yellow]"
        )
        console.print(
            f"To change '{key}', please edit the configuration file directly or use environment variables."
        )

        # Show where to make changes
        config_file_path = config_manager.config_path
        console.print(f"\nConfiguration file: {config_file_path}")

        if "proxy.customer_id" in key:
            console.print("Or set environment variable: BRIGHTDATA_CUSTOMER_ID")
        elif "storage.base_path" in key:
            console.print("Or set environment variable: CONTEXTER_STORAGE_PATH")
        elif "download.max_concurrent" in key:
            console.print("Or set environment variable: CONTEXTER_MAX_CONCURRENT")

        return

    except ConfigurationError as e:
        console.print(create_error_display(e, "Configuration Error"))
        ctx.exit(1)
    except Exception as e:
        console.print(create_error_display(e, "Unexpected Error"))
        ctx.exit(1)


@config.command()
@click.option("--force", is_flag=True, help="Reset without confirmation prompt")
@click.pass_context
@async_command
async def reset(ctx: click.Context, force: bool) -> None:
    """
    Reset configuration to default values.

    This will remove all custom configuration and restore defaults.
    Environment variables will still override defaults.
    """
    console: Console = ctx.obj["console"]

    if not force:
        if not Confirm.ask(
            "Are you sure you want to reset all configuration to defaults?"
        ):
            console.print("[yellow]Reset cancelled[/yellow]")
            return

    try:
        config_manager = ConfigurationManager()

        # Generate default configuration
        await config_manager.generate_default_config()

        console.print("[green]Configuration reset to defaults successfully[/green]")
        console.print("\nEnvironment variables will still be used if set.")
        console.print("Run 'contexter config show' to see current effective configuration.")

    except ConfigurationError as e:
        console.print(create_error_display(e, "Configuration Error"))
        ctx.exit(1)
    except Exception as e:
        console.print(create_error_display(e, "Unexpected Error"))
        ctx.exit(1)


@config.command()
@click.pass_context
@async_command
async def wizard(ctx: click.Context) -> None:
    """
    Interactive configuration wizard.

    Guides you through setting up C7DocDownloader configuration including:
    - BrightData proxy credentials
    - Storage locations
    - Performance settings
    - Advanced options
    """
    console: Console = ctx.obj["console"]

    try:
        config_manager = ConfigurationManager()

        # Welcome message
        welcome_panel = Panel(
            "Welcome to the C7DocDownloader Configuration Guide!\n\n"
            "This guide will show you how to configure C7DocDownloader.\n"
            "Configuration can be done via environment variables or configuration file.",
            title="[bold blue]Configuration Guide[/bold blue]",
            border_style="blue",
        )
        console.print(welcome_panel)

        # BrightData configuration
        console.print("\n[bold cyan]1. BrightData Proxy Configuration[/bold cyan]")
        console.print("Set these environment variables for BrightData proxy access:")
        console.print("  export BRIGHTDATA_CUSTOMER_ID='your_customer_id'")
        console.print("  export BRIGHTDATA_PASSWORD='your_password'")
        console.print("")
        console.print("Or add to your configuration file (~/.contexter/config.yaml):")
        console.print("  proxy:")
        console.print("    customer_id: 'your_customer_id'")

        # Storage configuration
        console.print("\n[bold cyan]2. Storage Configuration[/bold cyan]")
        console.print("Set storage directory (optional):")
        console.print("  export CONTEXTER_STORAGE_PATH='~/my-docs'")
        console.print("")
        console.print("Default: ~/.contexter/downloads")

        # Performance configuration
        console.print("\n[bold cyan]3. Performance Configuration[/bold cyan]")
        console.print("Set maximum concurrent downloads (optional):")
        console.print("  export CONTEXTER_MAX_CONCURRENT='15'")
        console.print("")
        console.print("Default: 10 concurrent downloads")

        # Show current configuration
        console.print("\n[bold cyan]4. Verify Configuration[/bold cyan]")
        console.print("Check your current configuration:")
        console.print("  contexter config show")
        console.print("")
        console.print("Test system health:")
        console.print("  contexter status")

        # Generate default config file
        if Confirm.ask(
            "\nWould you like to generate a default configuration file?", default=True
        ):
            await config_manager.generate_default_config()
            config_file_path = (
                config_manager.config_path or Path.home() / ".contexter" / "config.yaml"
            )
            console.print(
                f"\n[green]✓[/green] Default configuration created at: {config_file_path}"
            )
            console.print("Edit this file to customize your settings.")

        # Completion message
        completion_panel = Panel(
            "Configuration guide completed!\n\n"
            "Next steps:\n"
            "1. Set your BrightData credentials via environment variables\n"
            "2. Run 'contexter status' to verify connectivity\n"
            "3. Try downloading documentation: 'contexter download fastapi'",
            title="[bold green]Setup Complete[/bold green]",
            border_style="green",
        )
        console.print(completion_panel)

    except KeyboardInterrupt:
        console.print("\n[yellow]Configuration wizard cancelled[/yellow]")
        ctx.exit(130)
    except ConfigurationError as e:
        console.print(create_error_display(e, "Configuration Error"))
        ctx.exit(1)
    except Exception as e:
        console.print(create_error_display(e, "Unexpected Error"))
        ctx.exit(1)


def _get_value_source(env_var: str, value: Any) -> str:
    """Determine the source of a configuration value"""
    if os.getenv(env_var):
        return "environment"
    elif value:
        return "config file"
    else:
        return "default"


# Command examples are defined in the help text above
