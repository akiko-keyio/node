from node.node import Flow
from node.reporters import RichReporter, track
from node.logger import console


def test_track_inside_node():
    flow = Flow()

    @flow.node()
    def loop(n: int) -> int:
        total = 0
        for i in track(range(n), description="loop", total=n):
            total += i
        return total

    root = loop(5)
    reporter = RichReporter(console=console, force_terminal=True)
    result = flow.run(root, reporter=reporter)
    assert result == 10
