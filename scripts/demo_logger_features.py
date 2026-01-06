
import time
import random
import node
from node import gather, MemoryLRU
from node.reporters import RichReporter

# 使用线程执行器，MemoryLRU 确保每次运行从空缓存开始
node.configure(executor="thread", workers=4, cache=MemoryLRU())


@node.define()
def download_data(idx):
    """Simulates downloading data."""
    # Random sleep to show progress ⠋ updating
    time.sleep(random.uniform(0.5, 2.0))
    return idx


@node.define()
def process_data(data):
    """Simulates processing data."""
    time.sleep(random.uniform(1.0, 3.0))
    return data * 2


@node.define()
def summarize(results):
    """Simulates summary."""
    time.sleep(1.0)
    return sum(results)


def main():
    print("--- Demo: Cold Run (Showing Aggregation) ---")
    print("Goal: Show 'download_data' and 'process_data' aggregating multiple tasks.")
    
    # Create 8 tasks for download and process
    items = []
    for i in range(8):
        d = download_data(i)
        p = process_data(d)
        items.append(p)
    
    # Final summary
    # Use gather to resolve list of nodes
    root = summarize(gather(items))
    
    reporter = RichReporter(refresh_per_second=10)
    
    # First run: Should show counters like 3/8, spinner, etc.
    node.run(root, reporter=reporter)
    
    print("\n" + "="*30 + "\n")
    
    print("--- Demo: Cached Run (Showing Hidden Tasks) ---")
    print("Goal: Everything should be cached. Output should be empty or minimal.")
    
    # Second run: Should be silent for tasks
    node.run(root, reporter=reporter)
    print("(Run complete - if you saw no task bars above this line, cache hiding works!)")

if __name__ == "__main__":
    main()
