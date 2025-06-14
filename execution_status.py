from node.node import Flow
from node.reporters import RichReporter
import time

flow = Flow()

@flow.node()
def add(x: int, y: int) -> int:
    time.sleep(2)
    return x + y

@flow.node()
def square(x: int) -> int:
    time.sleep(3)
    return x * x

@flow.node()
def inc(x: int) -> int:
    time.sleep(0.2)
    return x + 1

if __name__ == "__main__":
    root = square(add(square(2), square(2)))
    reporter = RichReporter()
    result = flow.run(root, reporter=reporter)
    print("Result:", result)
