from rich.console import Console
from node import Runtime
from node.reporters import RichReporter, track


def test_track_inside_node_process():
    rt = Runtime(executor="process", continue_on_error=False, validate=False)

    @rt.define()
    def loop(n: int) -> int:
        total = 0
        for i in track(range(n), description="loop", total=n):
            total += i
        return total

    root = loop(5)
    reporter = RichReporter(console=Console(force_terminal=True))
    result = rt.run(root, reporter=reporter)
    assert result == 10
