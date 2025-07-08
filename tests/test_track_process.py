from rich.console import Console
from node.node import Flow
from node.reporters import RichReporter, track


def test_track_inside_node_process():
    flow = Flow(executor="process")

    @flow.node()
    def loop(n: int) -> int:
        total = 0
        for i in track(range(n), description="loop", total=n):
            total += i
        return total

    root = loop(5)
    reporter = RichReporter(console=Console(force_terminal=True))
    result = flow.run(root, reporter=reporter)
    assert result == 10
