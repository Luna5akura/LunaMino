# ai/ui.py

from datetime import datetime
from collections import deque
import os

try:
    from rich.console import Console
    from rich.table import Table
    from rich.live import Live
    from rich.panel import Panel
    from rich import box
    from rich.layout import Layout
    from rich.text import Text
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

class TrainingDashboard:
    def __init__(self, use_rich=True):
        self.use_rich = use_rich and HAS_RICH
        self.stats = {
            'game_idx': 0, 'score': 0.0, 'lines': 0, 'steps': 0,
            'loss': 0.0, 'buffer_size': 0, 'mode': 'Init',
            'recent_scores': deque(maxlen=20), 'best_score': 0.0
        }
        self.logs = deque(maxlen=8)
        
        if self.use_rich:
            self.console = Console()
            self.layout = Layout()
            self.layout.split(
                Layout(name="header", size=3),
                Layout(name="main", ratio=1),
                Layout(name="footer", size=10)
            )

    def log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        msg = f"[{timestamp}] {message}"
        if self.use_rich: self.logs.append(msg)
        else: print(msg)

    def update_stats(self, **kwargs):
        for k, v in kwargs.items():
            if k in self.stats: self.stats[k] = v
            if k == 'score':
                self.stats['recent_scores'].append(v)
                if v > self.stats['best_score']:
                    self.stats['best_score'] = v
                    self.log(f"[bold green]New High Score: {v:.1f}![/bold green]")

    def _generate_table(self):
        table = Table(box=box.ROUNDED, expand=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        avg = sum(self.stats['recent_scores'])/len(self.stats['recent_scores']) if self.stats['recent_scores'] else 0
        table.add_row("Game ID", str(self.stats['game_idx']), "Mode", self.stats['mode'])
        table.add_row("Last Score", f"{self.stats['score']:.1f}", "Best", f"{self.stats['best_score']:.1f}")
        table.add_row("Lines", str(self.stats['lines']), "Avg(20)", f"{avg:.1f}")
        table.add_row("Buffer", f"{self.stats['buffer_size']}", "Loss", f"{self.stats['loss']:.5f}")
        return Panel(table, title="Metrics", border_style="blue")

    def get_renderable(self):
        self.layout["header"].update(Panel(Text("Tetris AI RL", justify="center", style="bold yellow"), style="on blue"))
        self.layout["main"].update(self._generate_table())
        self.layout["footer"].update(Panel("\n".join(self.logs), title="Logs", style="grey70"))
        return self.layout

    def print_plain(self):
        s = self.stats
        print(f"Game {s['game_idx']} | Score {s['score']:.1f} | Loss {s['loss']:.4f} | Buf {s['buffer_size']}")

    def get_context(self):
        if self.use_rich: return Live(self.get_renderable(), refresh_per_second=4, console=self.console)
        return type('Dummy', (object,), {'__enter__': lambda s: None, '__exit__': lambda s,e,v,t: None})()