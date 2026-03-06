"""Tests for parallel context manager."""

import os
import sys
import time

import pytest
from joblib import Parallel, delayed

import node
from node.parallel import parallel_context


@pytest.fixture(autouse=True)
def reset_node():
    """Reset node runtime before each test."""
    node.reset()
    yield
    node.reset()


class TestParallelContext:
    """Tests for parallel_context context manager."""

    def test_no_limit_when_workers_1(self):
        """With workers=1, no limiting should be applied."""
        with parallel_context(workers=1):
            # Should work without any config changes
            result = Parallel(n_jobs=2)(delayed(lambda x: x)(i) for i in range(4))
            assert result == [0, 1, 2, 3]

    def test_no_limit_when_workers_0(self):
        """With workers=0, no limiting should be applied."""
        with parallel_context(workers=0):
            result = Parallel(n_jobs=2)(delayed(lambda x: x)(i) for i in range(4))
            assert result == [0, 1, 2, 3]

    def test_context_manager_basic(self):
        """Verify parallel_context works as context manager."""
        with parallel_context(workers=4):
            result = Parallel(n_jobs=2)(delayed(lambda x: x * 2)(i) for i in range(4))
            assert result == [0, 2, 4, 6]

    def test_nested_context_managers(self):
        """Verify nested parallel_context works correctly."""
        with parallel_context(workers=4):
            with parallel_context(workers=2):
                result = Parallel(n_jobs=2)(delayed(lambda x: x)(i) for i in range(3))
                assert result == [0, 1, 2]


class TestRuntimeIntegration:
    """Tests for parallel_context integration with Runtime."""

    def test_limit_inner_parallelism_default_true(self):
        """limit_inner_parallelism should default to True."""
        node.configure()
        rt = node.get_runtime()
        assert rt.limit_inner_parallelism is True

    def test_limit_inner_parallelism_can_be_disabled(self):
        """limit_inner_parallelism can be set to False."""
        node.configure(limit_inner_parallelism=False)
        rt = node.get_runtime()
        assert rt.limit_inner_parallelism is False


class TestNestedJoblib:
    """Tests for nested joblib parallelism scenarios."""

    def test_node_with_inner_joblib_parallel(self):
        """Test node function that uses joblib.Parallel internally."""
        node.configure(workers=2)
        
        @node.define()
        def compute_with_joblib(n: int) -> list:
            def work(x):
                time.sleep(0.01)
                return x * x
            
            results = Parallel(n_jobs=2)(delayed(work)(i) for i in range(n))
            return results
        
        result = compute_with_joblib(4)()
        assert result == [0, 1, 4, 9]

    def test_multiple_nodes_with_inner_joblib(self):
        """Test multiple concurrent nodes each using joblib.Parallel."""
        node.configure(workers=4)
        
        @node.define()
        def heavy_compute(task_id: int) -> dict:
            def inner_work(x):
                time.sleep(0.01)
                return x + task_id
            
            results = Parallel(n_jobs=2)(delayed(inner_work)(i) for i in range(3))
            return {"task": task_id, "results": results}
        
        @node.dimension()
        def task_ids():
            return [1, 2, 3, 4]
        
        all_results = heavy_compute(task_id=task_ids())()
        
        assert len(all_results) == 4
        assert all_results[0]["task"] == 1
        assert all_results[0]["results"] == [1, 2, 3]

    def test_sklearn_style_njobs(self):
        """Test that sklearn-style n_jobs=-1 works correctly."""
        node.configure(workers=2)
        
        @node.define()
        def sklearn_style_fit(n_samples: int) -> list:
            def process_sample(i):
                return i ** 2
            
            # n_jobs=-1 should be automatically limited
            results = Parallel(n_jobs=-1)(
                delayed(process_sample)(i) for i in range(n_samples)
            )
            return results
        
        result = sklearn_style_fit(5)()
        assert result == [0, 1, 4, 9, 16]


class TestBoundaryConditions:
    """Tests for edge cases and boundary conditions."""

    def test_empty_dimension(self):
        """Empty dimension should return empty result."""
        node.configure(workers=4)
        
        @node.dimension()
        def empty_dim():
            return []
        
        @node.define()
        def process(x):
            return x * 2
        
        result = process(x=empty_dim())()
        assert len(result) == 0

    def test_single_item_dimension(self):
        """Single item dimension should work correctly."""
        node.configure(workers=4)
        
        @node.dimension()
        def single():
            return [42]
        
        @node.define()
        def double(x):
            return x * 2
        
        result = double(x=single())()
        assert len(result) == 1
        assert result[0] == 84

    def test_large_workers_count(self):
        """Large workers count should not crash."""
        node.configure(workers=100)
        
        @node.define()
        def simple(x):
            return x
        
        @node.dimension()
        def items():
            return [1, 2, 3]
        
        result = simple(x=items())()
        assert list(result) == [1, 2, 3]
