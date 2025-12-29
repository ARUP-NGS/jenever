"""
Helper functions for displaying stage statistics in text-based tables and bar plots.
Uses the 'rich' library for beautiful terminal output.
"""

from typing import Dict, Any, List, Optional
import math

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.style import Style
from rich import box


# Create a default console
console = Console()


def _format_time(seconds: Optional[float], precision: int = 3) -> str:
    """Format time in seconds with appropriate units."""
    if seconds is None:
        return "N/A"
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.{precision}f} µs"
    elif seconds < 1:
        return f"{seconds * 1000:.{precision}f} ms"
    elif seconds < 60:
        return f"{seconds:.{precision}f} s"
    else:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"


def _colored_bar(value: float, max_value: float, width: int = 30, 
                 color: str = "green", bg_color: str = "grey23") -> Text:
    """Create a colored bar using rich Text."""
    if max_value == 0:
        ratio = 0
    else:
        ratio = min(value / max_value, 1.0)
    
    filled = int(ratio * width)
    empty = width - filled
    
    bar = Text()
    bar.append("█" * filled, style=color)
    bar.append("░" * empty, style=bg_color)
    return bar


def _get_time_color(value: float, max_value: float) -> str:
    """Get color based on relative value (green=fast, red=slow)."""
    if max_value == 0:
        return "white"
    ratio = value / max_value
    if ratio < 0.25:
        return "bright_green"
    elif ratio < 0.5:
        return "green"
    elif ratio < 0.75:
        return "yellow"
    else:
        return "red"


def _get_throughput_color(value: float, max_value: float) -> str:
    """Get color based on relative throughput (green=high, red=low)."""
    if max_value == 0:
        return "white"
    ratio = value / max_value
    if ratio > 0.75:
        return "bright_green"
    elif ratio > 0.5:
        return "green"
    elif ratio > 0.25:
        return "yellow"
    else:
        return "red"


def display_stats_table(stats_list: List[Dict[str, Any]], stage_names: Optional[List[str]] = None) -> Table:
    """
    Create a formatted table showing key metrics for multiple stages.
    
    Args:
        stats_list: List of stats dictionaries from Stage.get_stats()
        stage_names: Optional list of stage names. If not provided, uses "Stage 0", "Stage 1", etc.
    
    Returns:
        rich.Table object with the stats
    """
    if not stats_list:
        return Table(title="No stats provided")
    
    if stage_names is None:
        stage_names = [f"Stage {i}" for i in range(len(stats_list))]
    
    table = Table(
        title="STAGE STATISTICS",
        box=box.ROUNDED,
        header_style="bold magenta",
        title_style="bold bright_cyan",
        border_style="blue",
    )
    
    # Add columns
    table.add_column("Stage", style="cyan", no_wrap=True)
    table.add_column("Status", justify="center")
    table.add_column("Workers", justify="center", style="dim")
    table.add_column("Items Recv", justify="right")
    table.add_column("Items Proc", justify="right")
    table.add_column("Items Skip", justify="right")
    table.add_column("Avg Wait", justify="right")
    table.add_column("Avg Proc", justify="right")
    table.add_column("First Item", justify="right")
    
    for name, stats in zip(stage_names, stats_list):
        status = stats.get('status', 'N/A')
        workers = stats.get('total_workers', 0)
        items_recv = stats.get('items_received', stats.get('items_processed', 0))
        items_proc = stats.get('items_processed', 0)
        items_skip = stats.get('items_skipped', 0)
        avg_wait = stats.get('avg_wait_time')
        avg_proc = stats.get('avg_process_time')
        first_item = stats.get('first_item_time')
        
        # Color status
        status_style = "green" if status == "running" else "yellow" if status == "stopped" else "dim"
        
        table.add_row(
            name,
            Text(str(status), style=status_style),
            str(workers),
            str(items_recv),
            Text(str(items_proc), style="green" if items_proc > 0 else "dim"),
            Text(str(items_skip), style="yellow" if items_skip > 0 else "dim"),
            _format_time(avg_wait),
            _format_time(avg_proc),
            _format_time(first_item),
        )
    
    return table


def display_timing_barplot(stats_list: List[Dict[str, Any]], stage_names: Optional[List[str]] = None, 
                           bar_width: int = 40) -> Table:
    """
    Create colored bar plots for timing metrics (avg wait time and avg process time).
    
    Args:
        stats_list: List of stats dictionaries from Stage.get_stats()
        stage_names: Optional list of stage names
        bar_width: Width of the bar chart
    
    Returns:
        rich.Table object with bar plots
    """
    if not stats_list:
        return Table(title="No stats provided")
    
    if stage_names is None:
        stage_names = [f"Stage {i}" for i in range(len(stats_list))]
    
    # Create combined timing table
    table = Table(
        title="TIMING METRICS",
        box=box.ROUNDED,
        header_style="bold magenta",
        title_style="bold bright_cyan",
        border_style="blue",
        show_lines=True,
    )
    
    table.add_column("Stage", style="cyan", no_wrap=True)
    table.add_column("Avg Wait Time", justify="left", no_wrap=True)
    table.add_column("Value", justify="right", style="bright_white")
    table.add_column("Avg Process Time", justify="left", no_wrap=True)
    table.add_column("Value", justify="right", style="bright_white")
    
    wait_times = [stats.get('avg_wait_time', 0) or 0 for stats in stats_list]
    proc_times = [stats.get('avg_process_time', 0) or 0 for stats in stats_list]
    max_wait = max(wait_times) if wait_times else 1
    max_proc = max(proc_times) if proc_times else 1
    
    for name, wait_time, proc_time in zip(stage_names, wait_times, proc_times):
        wait_color = _get_time_color(wait_time, max_wait)
        proc_color = _get_time_color(proc_time, max_proc)
        
        wait_bar = _colored_bar(wait_time, max_wait, bar_width // 2, wait_color)
        proc_bar = _colored_bar(proc_time, max_proc, bar_width // 2, proc_color)
        
        table.add_row(
            name,
            wait_bar,
            Text(_format_time(wait_time), style=wait_color),
            proc_bar,
            Text(_format_time(proc_time), style=proc_color),
        )
    
    return table


def display_first_item_barplot(stats_list: List[Dict[str, Any]], stage_names: Optional[List[str]] = None,
                                bar_width: int = 40) -> Optional[Table]:
    """
    Create colored bar plot for time to first item.
    
    Args:
        stats_list: List of stats dictionaries from Stage.get_stats()
        stage_names: Optional list of stage names
        bar_width: Width of the bar chart
    
    Returns:
        rich.Table object with bar plot, or None if no first item times
    """
    if not stats_list:
        return None
    
    if stage_names is None:
        stage_names = [f"Stage {i}" for i in range(len(stats_list))]
    
    first_item_times = [(stats.get('first_item_time', 0) or 0) for stats in stats_list]
    
    if not any(t > 0 for t in first_item_times):
        return None
    
    max_first = max(first_item_times) if first_item_times else 1
    
    table = Table(
        title="TIME TO FIRST ITEM",
        box=box.ROUNDED,
        header_style="bold magenta",
        title_style="bold bright_cyan",
        border_style="blue",
    )
    
    table.add_column("Stage", style="cyan", no_wrap=True)
    table.add_column("Time", justify="left", no_wrap=True)
    table.add_column("Value", justify="right", style="bright_white")
    
    for name, first_time in zip(stage_names, first_item_times):
        color = _get_time_color(first_time, max_first) if first_time > 0 else "dim"
        bar = _colored_bar(first_time, max_first, bar_width, color)
        time_str = _format_time(first_time) if first_time > 0 else "N/A"
        
        table.add_row(
            name,
            bar,
            Text(time_str, style=color),
        )
    
    return table


def display_total_time_table(stats_list: List[Dict[str, Any]], stage_names: Optional[List[str]] = None,
                              bar_width: int = 40) -> Table:
    """
    Create a table showing total wait time and total process time for each stage.
    
    Args:
        stats_list: List of stats dictionaries from Stage.get_stats()
        stage_names: Optional list of stage names
        bar_width: Width of the bar chart
    
    Returns:
        rich.Table object with total time metrics
    """
    if not stats_list:
        return Table(title="No stats provided")
    
    if stage_names is None:
        stage_names = [f"Stage {i}" for i in range(len(stats_list))]
    
    table = Table(
        title="TOTAL TIME BY STAGE",
        box=box.ROUNDED,
        header_style="bold magenta",
        title_style="bold bright_cyan",
        border_style="blue",
        show_lines=True,
    )
    
    table.add_column("Stage", style="cyan", no_wrap=True)
    table.add_column("Total Wait Time", justify="left", no_wrap=True)
    table.add_column("Wait", justify="right", style="bright_white")
    table.add_column("Total Process Time", justify="left", no_wrap=True)
    table.add_column("Process", justify="right", style="bright_white")
    table.add_column("Total", justify="right", style="bold bright_white")
    
    wait_times = [stats.get('total_wait_time', 0) or 0 for stats in stats_list]
    proc_times = [stats.get('total_process_time', 0) or 0 for stats in stats_list]
    total_times = [w + p for w, p in zip(wait_times, proc_times)]
    
    max_wait = max(wait_times) if wait_times else 1
    max_proc = max(proc_times) if proc_times else 1
    max_total = max(total_times) if total_times else 1
    
    for name, wait_time, proc_time, total_time in zip(stage_names, wait_times, proc_times, total_times):
        wait_color = _get_time_color(wait_time, max_wait)
        proc_color = _get_time_color(proc_time, max_proc)
        total_color = _get_time_color(total_time, max_total)
        
        wait_bar = _colored_bar(wait_time, max_wait, bar_width // 2, wait_color)
        proc_bar = _colored_bar(proc_time, max_proc, bar_width // 2, proc_color)
        
        table.add_row(
            name,
            wait_bar,
            Text(_format_time(wait_time), style=wait_color),
            proc_bar,
            Text(_format_time(proc_time), style=proc_color),
            Text(_format_time(total_time), style=total_color),
        )
    
    return table


def display_throughput_barplot(stats_list: List[Dict[str, Any]], stage_names: Optional[List[str]] = None,
                                bar_width: int = 40) -> Table:
    """
    Create colored bar plots for throughput metrics (items processed, items per second).
    
    Args:
        stats_list: List of stats dictionaries from Stage.get_stats()
        stage_names: Optional list of stage names
        bar_width: Width of the bar chart
    
    Returns:
        rich.Table object with bar plots
    """
    if not stats_list:
        return Table(title="No stats provided")
    
    if stage_names is None:
        stage_names = [f"Stage {i}" for i in range(len(stats_list))]
    
    # Calculate metrics
    items_processed = [stats.get('items_processed', 0) for stats in stats_list]
    max_items = max(items_processed) if items_processed else 1
    
    throughputs = []
    for stats in stats_list:
        total_proc_time = stats.get('total_process_time', 0)
        items = stats.get('items_processed', 0)
        if total_proc_time > 0 and items > 0:
            throughput = items / total_proc_time * stats.get('total_workers', 1)
            throughputs.append(throughput)
        else:
            throughputs.append(0)
    
    max_throughput = max(throughputs) if throughputs else 1
    
    table = Table(
        title="THROUGHPUT METRICS",
        box=box.ROUNDED,
        header_style="bold magenta",
        title_style="bold bright_cyan",
        border_style="blue",
        show_lines=True,
    )
    
    table.add_column("Stage", style="cyan", no_wrap=True)
    table.add_column("Items Processed", justify="left", no_wrap=True)
    table.add_column("Count", justify="right")
    table.add_column("Throughput", justify="left", no_wrap=True)
    table.add_column("Rate", justify="right")
    
    for name, items, throughput in zip(stage_names, items_processed, throughputs):
        items_color = _get_throughput_color(items, max_items)
        throughput_color = _get_throughput_color(throughput, max_throughput)
        
        items_bar = _colored_bar(items, max_items, bar_width // 2, items_color)
        throughput_bar = _colored_bar(throughput, max_throughput, bar_width // 2, throughput_color)
        
        throughput_str = f"{throughput:.2f}/s" if throughput > 0 else "N/A"
        
        table.add_row(
            name,
            items_bar,
            Text(str(items), style=items_color),
            throughput_bar,
            Text(throughput_str, style=throughput_color),
        )
    
    return table


def display_worker_stats(stats: Dict[str, Any], stage_name: str = "Stage", bar_width: int = 30) -> Optional[Table]:
    """
    Create a detailed view of per-worker statistics for a single stage.
    
    Args:
        stats: Stats dictionary from Stage.get_stats()
        stage_name: Name of the stage
        bar_width: Width of the bar chart
    
    Returns:
        rich.Table object with worker statistics, or None if no workers
    """
    total_workers = stats.get('total_workers', 0)
    if total_workers == 0:
        return None
    
    # Collect worker data
    worker_items_recv = []
    worker_items_proc = []
    worker_proc_times = []
    worker_wait_times = []
    
    for i in range(total_workers):
        recv = stats.get(f'worker_{i}_items_received', 0)
        proc = stats.get(f'worker_{i}_items_processed', 0)
        proc_time = stats.get(f'worker_{i}_process_time', 0)
        wait_time = stats.get(f'worker_{i}_wait_time', 0)
        worker_items_recv.append(recv)
        worker_items_proc.append(proc)
        worker_proc_times.append(proc_time)
        worker_wait_times.append(wait_time)
    
    max_proc = max(worker_items_proc) if worker_items_proc else 1
    
    # Calculate load balance
    mean_items = sum(worker_items_proc) / len(worker_items_proc) if worker_items_proc else 0
    if mean_items > 0 and len(worker_items_proc) > 1:
        std_items = math.sqrt(sum((x - mean_items) ** 2 for x in worker_items_proc) / len(worker_items_proc))
        cv = std_items / mean_items * 100
        balance_color = "bright_green" if cv < 10 else "green" if cv < 20 else "yellow" if cv < 30 else "red"
        balance_text = f"CV = {cv:.1f}%"
    else:
        balance_color = "dim"
        balance_text = "N/A"
    
    table = Table(
        title=f"WORKER DETAILS: {stage_name}  [dim](Load Balance: [{balance_color}]{balance_text}[/{balance_color}])[/dim]",
        box=box.ROUNDED,
        header_style="bold magenta",
        title_style="bold bright_cyan",
        border_style="blue",
    )
    
    table.add_column("Worker", style="cyan", no_wrap=True)
    table.add_column("Items Recv", justify="right")
    table.add_column("Items Proc", justify="right")
    table.add_column("Wait Time", justify="right")
    table.add_column("Process Time", justify="right")
    table.add_column("Distribution", justify="left", no_wrap=True)
    
    # Assign colors to workers
    worker_colors = ["bright_blue", "bright_green", "bright_yellow", "bright_magenta", 
                     "bright_cyan", "orange1", "deep_pink1", "spring_green1"]
    
    for i in range(total_workers):
        recv = worker_items_recv[i]
        proc = worker_items_proc[i]
        proc_time = worker_proc_times[i]
        wait_time = worker_wait_times[i]
        color = worker_colors[i % len(worker_colors)]
        
        bar = _colored_bar(proc, max_proc, bar_width, color)
        
        table.add_row(
            f"Worker {i}",
            str(recv),
            Text(str(proc), style=color),
            _format_time(wait_time),
            _format_time(proc_time),
            bar,
        )
    
    return table


def display_exceptions(stats_list: List[Dict[str, Any]], stage_names: Optional[List[str]] = None) -> Optional[Panel]:
    """
    Display any exceptions that occurred during processing.
    
    Args:
        stats_list: List of stats dictionaries from Stage.get_stats()
        stage_names: Optional list of stage names
    
    Returns:
        rich.Panel with exception details, or None if no exceptions
    """
    if stage_names is None:
        stage_names = [f"Stage {i}" for i in range(len(stats_list))]
    
    all_exceptions = []
    for name, stats in zip(stage_names, stats_list):
        exceptions = stats.get('exceptions', [])
        for exc in exceptions:
            all_exceptions.append((name, exc))
    
    if not all_exceptions:
        return None
    
    text = Text()
    for stage_name, exc_info in all_exceptions:
        text.append(f"Stage: ", style="bold")
        text.append(f"{stage_name}\n", style="cyan")
        text.append(f"  Type: ", style="bold")
        text.append(f"{exc_info.get('exception_type', 'Unknown')}\n", style="red")
        text.append(f"  Message: ", style="bold")
        text.append(f"{exc_info.get('exception_message', 'No message')}\n", style="yellow")
        text.append(f"  Item: ", style="bold")
        item_str = str(exc_info.get('item', 'Unknown'))[:50]
        text.append(f"{item_str}...\n\n", style="dim")
    
    return Panel(
        text,
        title="[bold red]⚠️  EXCEPTIONS DETECTED[/bold red]",
        border_style="red",
        box=box.DOUBLE,
    )


def print_stats(stats_list: List[Dict[str, Any]], stage_names: Optional[List[str]] = None,
                show_worker_details: bool = True, bar_width: int = 40):
    """
    Print comprehensive statistics to stdout using rich formatting.
    
    Args:
        stats_list: List of stats dictionaries from Stage.get_stats()
        stage_names: Optional list of stage names
        show_worker_details: Whether to show per-worker details for each stage
        bar_width: Width of the bar charts
    """
    if not stats_list:
        console.print("[yellow]No stats provided.[/yellow]")
        return
    
    if stage_names is None:
        stage_names = [f"Stage {i}" for i in range(len(stats_list))]
    
    # Main summary table
    console.print(display_stats_table(stats_list, stage_names))
    console.print()
    
    # Timing bar plots (average times)
    console.print(display_timing_barplot(stats_list, stage_names, bar_width))
    console.print()
    
    # Total time by stage
    console.print(display_total_time_table(stats_list, stage_names, bar_width))
    console.print()
    
    # Time to first item
    first_item_table = display_first_item_barplot(stats_list, stage_names, bar_width)
    if first_item_table:
        console.print(first_item_table)
        console.print()
    
    # Throughput bar plots
    console.print(display_throughput_barplot(stats_list, stage_names, bar_width))
    console.print()
    
    # Per-worker details (if requested and if there are multi-worker stages)
    if show_worker_details:
        for name, stats in zip(stage_names, stats_list):
            if stats.get('total_workers', 0) > 1:
                worker_table = display_worker_stats(stats, name)
                if worker_table:
                    console.print(worker_table)
                    console.print()
    
    # Exceptions
    exceptions_panel = display_exceptions(stats_list, stage_names)
    if exceptions_panel:
        console.print(exceptions_panel)


# Convenience functions for individual components
def print_table(stats_list: List[Dict[str, Any]], stage_names: Optional[List[str]] = None):
    """Print just the summary table."""
    console.print(display_stats_table(stats_list, stage_names))


def print_timing_bars(stats_list: List[Dict[str, Any]], stage_names: Optional[List[str]] = None, bar_width: int = 40):
    """Print just the timing bar plots."""
    console.print(display_timing_barplot(stats_list, stage_names, bar_width))


def print_throughput_bars(stats_list: List[Dict[str, Any]], stage_names: Optional[List[str]] = None, bar_width: int = 40):
    """Print just the throughput bar plots."""
    console.print(display_throughput_barplot(stats_list, stage_names, bar_width))


def print_total_time_table(stats_list: List[Dict[str, Any]], stage_names: Optional[List[str]] = None, bar_width: int = 40):
    """Print just the total time table."""
    console.print(display_total_time_table(stats_list, stage_names, bar_width))
