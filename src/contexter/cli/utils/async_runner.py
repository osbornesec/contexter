"""
Async execution utilities for CLI commands
"""

import asyncio
import signal
import sys
from functools import wraps
from typing import Any, Awaitable, Callable, TypeVar

from rich.console import Console

F = TypeVar("F", bound=Callable[..., Awaitable[Any]])


def async_command(f: F) -> Callable[..., Any]:
    """
    Decorator to run async functions in CLI context with proper error handling
    """

    @wraps(f)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return asyncio.run(f(*args, **kwargs))  # type: ignore
        except KeyboardInterrupt:
            console = Console()
            console.print("\n[yellow]Operation interrupted by user[/yellow]")
            sys.exit(130)  # Standard exit code for SIGINT
        except Exception as e:
            console = Console()
            console.print(f"[red]Operation failed: {e}[/red]")
            sys.exit(1)

    return wrapper


class GracefulKiller:
    """
    Handle graceful shutdown for async operations
    """

    def __init__(self) -> None:
        self.kill_now = False
        self.cleanup_functions: list[Callable[[], None]] = []
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum: int, frame: Any) -> None:
        """Handle shutdown signals"""
        self.kill_now = True
        # Run cleanup functions
        for cleanup_func in self.cleanup_functions:
            try:
                cleanup_func()
            except Exception as e:
                console = Console()
                console.print(f"[yellow]Cleanup error: {e}[/yellow]")

    def register_cleanup(self, func: Callable[[], None]) -> None:
        """Register a cleanup function to run on shutdown"""
        self.cleanup_functions.append(func)


def run_with_cancellation(
    coro: Callable[[], Awaitable[Any]], killer: GracefulKiller
) -> Any:
    """
    Run async coroutine with cancellation support
    """

    async def wrapped_coro() -> Any:
        task: asyncio.Task[Any] = asyncio.create_task(coro())  # type: ignore
        while not killer.kill_now and not task.done():
            await asyncio.sleep(0.1)

        if killer.kill_now and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        return task.result() if task.done() else None

    return asyncio.run(wrapped_coro())
