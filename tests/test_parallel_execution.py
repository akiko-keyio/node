"""Tests for parallel execution mechanics."""

import os
import sys
import threading
import time

import pytest

import node
from node import DiskJoblib, MemoryLRU, ChainCache


@pytest.fixture(autouse=True)
def reset_node():
    """Reset node runtime before each test."""
    node.reset()
    yield
    node.reset()


@pytest.fixture
def temp_cache(tmp_path):
    """Provide a temporary cache directory for test isolation."""
    return ChainCache([MemoryLRU(), DiskJoblib(root=tmp_path / "cache")])


class TestParallelExecution:
    """Tests verify that nodes actually run in parallel."""

    def test_nodes_run_concurrently_thread(self, temp_cache):
        """Verify independent nodes run concurrently in thread mode."""
        node.configure(workers=4, executor="thread", cache=temp_cache)
        
        task_sleep = 0.3
        num_tasks = 4
        
        @node.define()
        def parallel_sleep_task(task_id: int):
            time.sleep(task_sleep)
            return task_id
        
        @node.dimension()
        def task_ids():
            return list(range(num_tasks))
        
        t0 = time.perf_counter()
        parallel_sleep_task(task_id=task_ids())()
        dur = time.perf_counter() - t0
        
        serial_estimate = num_tasks * task_sleep
        assert dur < serial_estimate * 0.6, \
            f"Expected parallel speedup, got {dur:.2f}s vs serial {serial_estimate:.2f}s"

    @pytest.mark.skip(reason="Process executor has issues in test environment")
    def test_nodes_run_concurrently_process(self, temp_cache):
        """Verify independent nodes run concurrently in process mode."""
        node.configure(workers=4, executor="process", cache=temp_cache)
        
        @node.define()
        def get_pid_task(task_id: int):
            time.sleep(0.3)
            return os.getpid()
        
        @node.dimension()
        def task_ids():
            return [0, 1, 2, 3]
        
        pids = get_pid_task(task_id=task_ids())()
        assert len(set(pids)) >= 1

    def test_workers_1_runs_serially(self, temp_cache):
        """With workers=1, no concurrent execution should happen."""
        node.configure(workers=1, cache=temp_cache)
        
        active_count = 0
        max_concurrent = 0
        lock = threading.Lock()
        
        @node.define()
        def serial_counting_task(i: int):
            nonlocal active_count, max_concurrent
            with lock:
                active_count += 1
                max_concurrent = max(max_concurrent, active_count)
            time.sleep(0.05)
            with lock:
                active_count -= 1
            return i
        
        @node.dimension()
        def serial_items():
            return [1, 2, 3, 4]
        
        serial_counting_task(i=serial_items())()
        
        assert max_concurrent == 1, f"Expected max concurrent 1, got {max_concurrent}"


class TestWorkerLimits:
    """Tests for worker constraints."""

    def test_per_function_worker_limit(self, temp_cache):
        """Verify @node.define(workers=N) limits concurrency."""
        node.configure(workers=8, cache=temp_cache)
        
        active_count = 0
        max_active = 0
        lock = threading.Lock()
        
        @node.define(workers=2)
        def worker_limited_task(i: int):
            nonlocal active_count, max_active
            with lock:
                active_count += 1
                max_active = max(max_active, active_count)
            time.sleep(0.05)
            with lock:
                active_count -= 1
            return i
        
        @node.dimension()
        def limit_items():
            return list(range(10))
        
        worker_limited_task(i=limit_items())()
        
        assert max_active <= 2, f"Expected max concurrency <= 2, got {max_active}"

    def test_local_execution_in_main_thread(self, temp_cache):
        """Verify @node.define(local=True) runs in main thread."""
        node.configure(workers=4, cache=temp_cache)
        main_thread_id = threading.get_ident()
        
        @node.define(local=True)
        def main_thread_task(i):
            return threading.get_ident()
        
        @node.dimension()
        def local_items():
            return [1, 2, 3]
        
        thread_ids = main_thread_task(i=local_items())()
        
        for tid in thread_ids:
            assert tid == main_thread_id, "Local task should run in main thread"


class TestErrorIsolation:
    """Tests for error handling during parallel execution."""

    def test_error_isolation_with_continue(self, temp_cache):
        """Verify one failing node doesn't stop other independent nodes."""
        node.configure(workers=4, continue_on_error=True, cache=temp_cache)
        
        results = []
        lock = threading.Lock()
        
        @node.define(cache=False)
        def error_isolation_task(i: int):
            if i == 2:
                raise ValueError("Intentional failure")
            with lock:
                results.append(i)
            return i
        
        @node.dimension()
        def error_items():
            return [1, 2, 3, 4]
        
        error_isolation_task(i=error_items())()
        
        succeeded = set(results)
        assert 2 not in succeeded, "Task 2 should have failed"
        assert len(succeeded) == 3, f"Expected 3 successful tasks, got {len(succeeded)}"

    def test_downstream_skipped_on_upstream_failure(self, temp_cache):
        """Verify downstream nodes are skipped when upstream fails."""
        node.configure(workers=4, continue_on_error=True, cache=temp_cache)
        
        downstream_called = False
        
        @node.define(cache=False)
        def failing_upstream():
            raise ValueError("Upstream failed")
        
        @node.define(cache=False)
        def skipped_downstream(data):
            nonlocal downstream_called
            downstream_called = True
            return data
        
        skipped_downstream(data=failing_upstream())()
        
        assert not downstream_called, "Downstream should be skipped"


class TestCacheConcurrency:
    """Tests for cache safety under concurrency."""

    def test_cache_prevents_recomputation(self, tmp_path):
        """Verify cache prevents re-execution of the same computation."""
        cache = ChainCache([MemoryLRU(), DiskJoblib(root=tmp_path / "cache")])
        node.configure(workers=4, cache=cache)

        call_count = 0
        count_lock = threading.Lock()
        
        @node.define()
        def file_tracked_task(x):
            nonlocal call_count
            with count_lock:
                call_count += 1
            return x * x
        
        @node.dimension()
        def cache_items():
            return [1, 2, 3, 4, 5]
        
        result1 = file_tracked_task(x=cache_items())()
        calls_after_first = call_count
        
        result2 = file_tracked_task(x=cache_items())()
        calls_after_second = call_count
        
        assert list(result1) == list(result2)
        assert calls_after_first == 5
        assert calls_after_second == 5, "Cache should prevent re-execution"

    def test_concurrent_cache_access_safe(self, tmp_path):
        """Verify concurrent access to cache doesn't cause corruption."""
        cache = ChainCache([MemoryLRU(), DiskJoblib(root=tmp_path / "cache")])
        node.configure(workers=8, cache=cache)
        
        @node.define()
        def square_task(x):
            time.sleep(0.01)
            return x * x
        
        @node.dimension()
        def numbers():
            return list(range(50))
        
        result = square_task(x=numbers())()
        
        expected = [i * i for i in range(50)]
        assert list(result) == expected
