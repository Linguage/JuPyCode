from rich.table import Table
from rich.console import Console
import math

print(math.pi)

console = Console()

table = Table(show_header=True, header_style="bold magenta")
table.add_column("Name", style="dim", width=12)
table.add_column("Age")
table.add_column("City")

table.add_row("Peter", "23", "New York")
table.add_row("Sarah", "27", "London")
table.add_row("John", "30", "Sydney")

console.print(table)
