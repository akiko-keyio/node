"""Demo: Using track() to avoid Rich Progress conflicts.

This demonstrates the correct way to show progress inside node functions
without conflicting with RichReporter's Live display.
"""

import time
import random
import node
from node import gather, MemoryLRU
from node.reporters import RichReporter, track

# 使用线程执行器，MemoryLRU 确保每次运行从空缓存开始
node.configure(executor="thread", workers=4, cache=MemoryLRU())


@node.define()
def download_batch(batch_id, items):
    """Download items with internal progress tracking.
    
    使用框架的 track() 函数来显示内部循环进度，
    而不是直接使用 rich.progress.Progress，避免与 RichReporter 冲突。
    """
    results = []
    # ✅ 正确方式：使用 node.track()
    for item in track(items, description=f"Batch {batch_id}", total=len(items)):
        time.sleep(random.uniform(0.1, 0.3))  # Simulate download
        results.append(item * 2)
    return results


@node.define()
def process_results(batches):
    """Process all downloaded batches."""
    time.sleep(0.5)
    return sum(sum(batch) for batch in batches)


def main():
    print("=== Demo: track() inside node functions ===\n")
    print("track() integrates with RichReporter to show nested progress.\n")
    
    # Create 3 batches, each with 5 items
    batches = []
    for batch_id in range(3):
        items = list(range(batch_id * 5, (batch_id + 1) * 5))
        batches.append(download_batch(batch_id, items))
    
    root = process_results(gather(batches))
    
    reporter = RichReporter(refresh_per_second=10)
    result = node.run(root, reporter=reporter)
    
    print(f"\nResult: {result}")


if __name__ == "__main__":
    main()
