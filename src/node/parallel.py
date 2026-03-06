"""Parallel execution context for controlling nested parallelism."""

from __future__ import annotations

from contextlib import contextmanager

__all__ = ["parallel_context"]


@contextmanager
def parallel_context(workers: int):
    """Limit inner BLAS/OpenMP threads when running inside Node's parallel workers.

    When Node runs multiple nodes in parallel (workers > 1), each node's
    internal BLAS/OpenMP threads (used by NumPy, SciPy, sklearn) should be
    limited to prevent thread explosion.

    This context manager uses joblib's `parallel_config` to set
    `inner_max_num_threads=1`, which limits nested threading while
    allowing joblib.Parallel to still use multiple processes.

    Args:
        workers: Number of concurrent Node workers. If <= 1, no limiting
                 is applied.

    Example:
        >>> with parallel_context(workers=4):
        ...     # sklearn's n_jobs still works, but BLAS is single-threaded
        ...     model.fit(X, y)
    """
    if workers <= 1:
        yield
        return

    from joblib import parallel_config

    # inner_max_num_threads=1 limits BLAS/OpenMP inside joblib workers
    # This prevents: 4 Node workers × 16 BLAS threads = 64 threads
    # Instead we get: 4 Node workers × 1 BLAS thread = 4 threads
    # joblib.Parallel can still use n_jobs for process-level parallelism
    # backend='loky' is required for inner_max_num_threads to work
    with parallel_config(backend="loky", inner_max_num_threads=1):
        yield
