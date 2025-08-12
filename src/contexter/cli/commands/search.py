"""
Search command for finding libraries in the Context7 database.
"""

import click
from rich.console import Console
from rich.table import Table
from rich.text import Text
from typing import Optional

from ...integration.context7_client import Context7Client
from ..utils.async_runner import async_command


@click.command()
@click.argument("query", required=True)
@click.option(
    "--limit",
    "-l", 
    type=int,
    default=10,
    help="Maximum number of results to show (1-50)"
)
@click.option(
    "--format",
    "-f",
    type=click.Choice(["list", "table", "json"], case_sensitive=False),
    default="list",
    help="Output format: list (default), table, or json"
)
@click.pass_context
@async_command
async def search(
    ctx: click.Context,
    query: str,
    limit: int,
    format: str,
) -> None:
    """
    Search for libraries in the Context7 database.

    Search for libraries using natural language queries. This helps you find
    the exact library ID needed for downloading documentation.

    QUERY is the search term or phrase to look for.

    Examples:
      contexter search "react framework"
      contexter search "python web framework" --limit 5 --format table
      contexter search nextjs --format json
      contexter search "machine learning" --format table
    """
    console: Console = ctx.obj["console"]
    verbose: bool = ctx.obj.get("verbose", False)

    # Validate limit
    if not (1 <= limit <= 50):
        console.print("[red]Error: Limit must be between 1 and 50[/red]")
        ctx.exit(1)

    try:
        # Initialize Context7 client
        if verbose:
            console.print(f"[cyan]Searching for: '{query}'[/cyan]")
        
        context7_client = Context7Client()
        
        # Perform search
        with console.status(f"[cyan]Searching Context7 database...[/cyan]"):
            results = await context7_client.resolve_library_id(query, limit=limit)
        
        # Handle no results
        if not results:
            console.print(f"[yellow]No libraries found matching: '{query}'[/yellow]")
            console.print("\n[dim]Try:")
            console.print("• Using different keywords")
            console.print("• Making your query more general")
            console.print("• Checking for typos[/dim]")
            return
        
        # Display results in requested format
        if format == "json":
            _display_json_results(console, results, query)
        elif format == "table":
            _display_table_results(console, results, query)
        else:  # default list format
            _display_list_results(console, results, query)
        
        # Show usage tip (not for JSON output)
        if format != "json":
            console.print(f"\n[dim]💡 To download documentation: [bold]contexter download <library-id>[/bold][/dim]")

    except Exception as e:
        console.print(f"[red]Search failed: {e}[/red]")
        if verbose:
            console.print_exception()
        ctx.exit(1)


def _display_table_results(console: Console, results, query: str) -> None:
    """Display search results in table format."""
    
    table = Table(
        title=f"Libraries matching '{query}' ({len(results)} found)",
        show_header=True,
        header_style="bold cyan",
        title_style="bold blue"
    )
    
    table.add_column("Library ID", style="green", no_wrap=False)
    table.add_column("Name", style="white", no_wrap=False) 
    table.add_column("Description", style="dim white")
    table.add_column("Tokens", justify="right", style="magenta")
    table.add_column("Stars", justify="right", style="yellow")
    table.add_column("Trust", justify="right", style="cyan")
    
    for result in results:
        # Format star count
        stars = "⭐" + str(result.star_count) if result.star_count > 0 else "-"
        
        # Format trust score
        trust = f"{result.trust_score:.1f}" if result.trust_score > 0 else "-"
        
        # Format token count
        if result.total_tokens > 0:
            if result.total_tokens >= 1_000_000:
                tokens = f"{result.total_tokens / 1_000_000:.1f}M"
            elif result.total_tokens >= 1_000:
                tokens = f"{result.total_tokens / 1_000:.1f}K"
            else:
                tokens = str(result.total_tokens)
        else:
            tokens = "-"
        
        # Truncate description
        desc = result.description
        if len(desc) > 80:
            desc = desc[:77] + "..."
        
        table.add_row(
            result.library_id,
            result.name,
            desc,
            tokens,
            stars,
            trust
        )
    
    console.print(table)


def _display_list_results(console: Console, results, query: str) -> None:
    """Display search results in simple list format."""
    
    console.print(f"[bold blue]Libraries matching '{query}' ({len(results)} found):[/bold blue]\n")
    
    for i, result in enumerate(results, 1):
        # Create formatted entry
        console.print(f"[bold green]{i}. {result.library_id}[/bold green]")
        console.print(f"   Name: [white]{result.name}[/white]")
        
        # Show description (truncated)
        desc = result.description
        if len(desc) > 100:
            desc = desc[:97] + "..."
        console.print(f"   Description: [dim]{desc}[/dim]")
        
        # Show metadata if available
        metadata_parts = []
        
        # Add token count (most important)
        if result.total_tokens > 0:
            if result.total_tokens >= 1_000_000:
                token_display = f"{result.total_tokens / 1_000_000:.1f}M tokens"
            elif result.total_tokens >= 1_000:
                token_display = f"{result.total_tokens / 1_000:.1f}K tokens"
            else:
                token_display = f"{result.total_tokens} tokens"
            metadata_parts.append(token_display)
        
        if result.star_count > 0:
            metadata_parts.append(f"⭐ {result.star_count}")
        if result.trust_score > 0:
            metadata_parts.append(f"Trust: {result.trust_score:.1f}")
        
        if metadata_parts:
            console.print(f"   [{' • '.join(metadata_parts)}]")
        
        console.print()  # Empty line between results


def _display_json_results(console: Console, results, query: str) -> None:
    """Display search results in JSON format."""
    import json
    
    # Convert results to serializable format
    json_results = []
    for result in results:
        json_results.append({
            "library_id": result.library_id,
            "name": result.name,
            "description": result.description,
            "star_count": result.star_count,
            "trust_score": result.trust_score,
            "search_relevance": result.search_relevance,
            "total_tokens": result.total_tokens,
            "total_snippets": result.total_snippets,
            "total_pages": result.total_pages,
            "metadata": result.metadata
        })
    
    output = {
        "query": query,
        "total_results": len(results),
        "libraries": json_results
    }
    
    console.print(json.dumps(output, indent=2, ensure_ascii=False))


# Helper function for command examples
def _get_command_examples() -> list[str]:
    """Get example commands for help display"""
    return [
        "contexter search 'react framework'",
        "contexter search 'python web framework' --limit 5 --format table",
        "contexter search nextjs --format json", 
        "contexter search 'machine learning' --format table",
        "contexter search django",
        "contexter search 'vue components' --limit 3 --format table"
    ]