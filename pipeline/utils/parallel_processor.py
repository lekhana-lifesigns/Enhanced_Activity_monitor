# pipeline/utils/parallel_processor.py
"""
Parallel Processing Utilities (TODO-051).
Enables parallel execution of independent operations for better performance.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Callable, Any, Optional
import threading

log = logging.getLogger("parallel_processor")


class ParallelProcessor:
    """
    Parallel processor for independent operations.
    TODO-051: Parallel Processing
    """
    
    def __init__(self, max_workers=3, enabled=True):
        """
        Initialize parallel processor.
        
        Args:
            max_workers: Maximum number of parallel workers
            enabled: Whether parallel processing is enabled
        """
        self.max_workers = max_workers
        self.enabled = enabled
        self.executor = None
        
        if self.enabled:
            self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="parallel")
            log.info("Parallel processor initialized with %d workers", max_workers)
        else:
            log.info("Parallel processor disabled (sequential mode)")
    
    def process_parallel(self, tasks: List[tuple]) -> List[Any]:
        """
        Process multiple tasks in parallel.
        
        Args:
            tasks: List of (func, *args, **kwargs) tuples
        
        Returns:
            List of results in same order as tasks
        """
        if not self.enabled or not self.executor:
            # Sequential processing
            return [func(*args, **kwargs) for func, args, kwargs in tasks]
        
        # Parallel processing
        futures = {}
        for i, (func, args, kwargs) in enumerate(tasks):
            future = self.executor.submit(func, *args, **kwargs)
            futures[future] = i
        
        # Collect results in order
        results = [None] * len(tasks)
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                log.warning("Parallel task %d failed: %s", idx, e)
                results[idx] = None
        
        return results
    
    def shutdown(self, wait=True, timeout=5.0):
        """Shutdown the executor."""
        if self.executor:
            self.executor.shutdown(wait=wait, timeout=timeout)
            log.info("Parallel processor shutdown")

