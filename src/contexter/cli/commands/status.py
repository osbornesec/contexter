"""
Status and health check commands
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import click
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel

from ...core.config_manager import ConfigurationManager
from ...integration.context7_client import Context7Client
from ...integration.proxy_manager import BrightDataProxyManager
from ..ui.display import create_stats_display, create_status_panel
from ..ui.formatters import format_timestamp
from ..utils.async_runner import async_command


@click.command()
@click.option("--detailed", is_flag=True, help="Show detailed status information")
@click.pass_context
@async_command
async def status(ctx: click.Context, detailed: bool) -> None:
    """
    Check system status and component health.

    Verifies connectivity and health of all system components:
    - Configuration validity
    - BrightData proxy connectivity
    - Context7 API accessibility
    - Storage system health
    - Recent download statistics

    Use --detailed for comprehensive diagnostic information.
    """
    console: Console = ctx.obj["console"]
    verbose: bool = ctx.obj.get("verbose", False)

    console.print("[cyan]Checking system status...[/cyan]")

    try:
        with console.status("Running health checks..."):
            # Run all health checks concurrently
            config_status = await _check_configuration_health()
            proxy_status = await _check_proxy_health()
            api_status = await _check_api_connectivity()
            storage_status = await _check_storage_health()

        # Display status panels
        status_panels = [
            create_status_panel("Configuration", config_status),
            create_status_panel("BrightData Proxy", proxy_status),
            create_status_panel("Context7 API", api_status),
            create_status_panel("Storage System", storage_status),
        ]

        console.print(Columns(status_panels, equal=True))

        # Show detailed information if requested
        if detailed:
            await _show_detailed_status(console, verbose)

        # Show overall system health
        overall_healthy = all(
            [
                config_status.get("healthy", False),
                proxy_status.get("healthy", False),
                api_status.get("healthy", False),
                storage_status.get("healthy", False),
            ]
        )

        if overall_healthy:
            console.print("\n[bold green]✓ All systems operational[/bold green]")
            ctx.exit(0)
        else:
            console.print("\n[bold yellow]⚠ Some systems have issues[/bold yellow]")
            console.print("Run 'contexter status --detailed' for more information")
            ctx.exit(1)

    except KeyboardInterrupt:
        console.print("\n[yellow]Status check interrupted[/yellow]")
        ctx.exit(130)
    except Exception as e:
        console.print(f"[red]Error during status check: {e}[/red]")
        if verbose:
            console.print_exception()
        ctx.exit(1)


async def _check_configuration_health() -> Dict[str, Any]:
    """Check configuration system health"""
    try:
        config_manager = ConfigurationManager()
        config = await config_manager.load_config()

        # Check for required settings
        issues = []

        if not config.proxy.customer_id:
            issues.append("BrightData customer ID not set")

        # Note: password comes from environment, not config
        password = os.getenv("BRIGHTDATA_PASSWORD")
        if not password:
            issues.append("BrightData password not set")

        storage_path = Path(config.storage.base_path).expanduser()
        if not storage_path.exists():
            try:
                storage_path.mkdir(parents=True, exist_ok=True)
            except Exception:
                issues.append("Cannot access storage directory")

        return {
            "healthy": len(issues) == 0,
            "details": (
                "\n".join(issues) if issues else "All required settings configured"
            ),
            "version": "1.0.0",
            "last_check": format_timestamp(datetime.now()),
        }

    except Exception as e:
        return {
            "healthy": False,
            "error": str(e),
            "details": "Configuration file may be corrupted or missing",
        }


async def _check_proxy_health() -> Dict[str, Any]:
    """Check BrightData proxy connectivity"""
    try:
        config_manager = ConfigurationManager()
        config = await config_manager.load_config()

        if not config.proxy.customer_id:
            return {
                "healthy": False,
                "details": "BrightData credentials not configured",
                "error": "Missing credentials",
            }

        # Try to initialize proxy manager with DNS resolution configuration
        brightdata_password = os.getenv("BRIGHTDATA_PASSWORD")
        if not brightdata_password:
            return {
                "healthy": False,
                "details": "BrightData password not configured",
                "error": "Missing password",
            }

        proxy_manager = BrightDataProxyManager(
            customer_id=config.proxy.customer_id,
            password=brightdata_password,
            zone_name=config.proxy.zone_name,
            dns_resolution=config.proxy.dns_resolution,
        )

        # Test proxy connectivity
        is_connected = await proxy_manager.test_connectivity()

        if is_connected:
            return {
                "healthy": True,
                "details": "Proxy connectivity test passed",
                "last_check": format_timestamp(datetime.now()),
            }
        else:
            return {
                "healthy": False,
                "details": "Proxy connectivity test failed",
                "error": "Connection failed",
            }

    except Exception as e:
        return {
            "healthy": False,
            "error": str(e),
            "details": "Failed to test proxy connectivity",
        }


async def _check_api_connectivity() -> Dict[str, Any]:
    """Check Context7 API connectivity"""
    try:
        context7_client = Context7Client()

        # Test API connectivity with a simple search
        is_connected = await context7_client.test_connectivity()

        if is_connected:
            return {
                "healthy": True,
                "details": "Context7 API is reachable and responding",
                "version": "1.0",
                "last_check": format_timestamp(datetime.now()),
            }
        else:
            return {
                "healthy": False,
                "details": "Context7 API unreachable",
                "error": "Connection failed",
            }

    except Exception as e:
        return {
            "healthy": False,
            "error": str(e),
            "details": "Failed to connect to Context7 API",
        }


async def _check_storage_health() -> Dict[str, Any]:
    """Check storage system health"""
    try:
        config_manager = ConfigurationManager()
        config = await config_manager.load_config()

        storage_path = Path(config.storage.base_path).expanduser()

        # Check directory accessibility
        if not storage_path.exists():
            storage_path.mkdir(parents=True, exist_ok=True)

        if not os.access(storage_path, os.R_OK | os.W_OK):
            return {
                "healthy": False,
                "error": "No read/write access to storage directory",
                "details": f"Path: {storage_path}",
            }

        # Test storage operations - simple write test
        test_file = storage_path / "health_check.txt"
        try:
            test_file.write_text("health check test")
            test_content = test_file.read_text()
            assert test_content == "health check test"
        finally:
            if test_file.exists():
                test_file.unlink()

        # Get storage statistics
        stats = _get_storage_statistics(storage_path)

        return {
            "healthy": True,
            "details": f"Storage accessible, {stats['file_count']} files, {stats['total_size_mb']:.1f} MB",
            "last_check": format_timestamp(datetime.now()),
        }

    except Exception as e:
        return {
            "healthy": False,
            "error": str(e),
            "details": "Storage system not accessible",
        }


async def _show_detailed_status(console: Console, verbose: bool) -> None:
    """Show detailed system status information"""

    console.print("\n[bold cyan]Detailed System Information[/bold cyan]")

    # Environment information
    env_info = []
    env_info.append(
        f"Python version: {'.'.join(map(str, __import__('sys').version_info[:3]))}"
    )
    env_info.append(
        f"Platform: {__import__('platform').system()} {__import__('platform').release()}"
    )

    # Check key environment variables
    env_vars = [
        "BRIGHTDATA_CUSTOMER_ID",
        "BRIGHTDATA_PASSWORD",
        "CONTEXTER_STORAGE_PATH",
        "CONTEXTER_CONFIG_PATH",
        "CONTEXTER_LOG_LEVEL",
    ]

    for var in env_vars:
        value = os.getenv(var)
        if value:
            if "password" in var.lower() or "secret" in var.lower():
                env_info.append(f"{var}: ***set***")
            else:
                env_info.append(f"{var}: {value}")
        else:
            env_info.append(f"{var}: not set")

    env_panel = Panel("\n".join(env_info), title="Environment", border_style="blue")
    console.print(env_panel)

    # Storage statistics
    try:
        config_manager = ConfigurationManager()
        config = await config_manager.load_config()
        storage_path = Path(config.storage.base_path).expanduser()
        stats = _get_storage_statistics(storage_path)

        stats_panel = create_stats_display(
            {
                "storage": {
                    "libraries_count": stats["library_count"],
                    "total_size_mb": stats["total_size_mb"],
                    "file_count": stats["file_count"],
                    "compression_ratio": stats.get("compression_ratio", 0),
                }
            }
        )
        console.print(stats_panel)

    except Exception as e:
        console.print(f"[red]Could not load storage statistics: {e}[/red]")

    # Recent activity
    if verbose:
        console.print(
            "\n[dim]Use --verbose with other commands for detailed logging[/dim]"
        )


def _get_storage_statistics(storage_path: Path) -> Dict[str, Any]:
    """Get storage directory statistics"""
    stats = {
        "file_count": 0,
        "library_count": 0,
        "total_size_mb": 0.0,
        "compression_ratio": 0.0,
    }

    if not storage_path.exists():
        return stats

    total_size_bytes = 0
    library_dirs = set()

    for file_path in storage_path.rglob("*"):
        if file_path.is_file():
            stats["file_count"] += 1
            total_size_bytes += file_path.stat().st_size

            # Count unique library directories
            relative_path = file_path.relative_to(storage_path)
            if len(relative_path.parts) > 0:
                library_dirs.add(relative_path.parts[0])

    stats["library_count"] = len(library_dirs)
    stats["total_size_mb"] = total_size_bytes / (1024 * 1024)

    return stats


# Command examples are defined in the help text above
