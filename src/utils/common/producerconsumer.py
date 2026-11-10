"""
Producer-consumer pattern utilities for Luigi tasks.

This module provides helper functions and a mixin class to implement
producer-consumer patterns with consistent error handling and resource management.

Structure:
1. Mixin class (ProducerConsumerMixin)
2. Pipeline creation (create_producer_consumer_pipeline)
3. Producer wrapper (create_producer_with_sentinel)
4. Consumer failure record (logged_series_failure)
5. Producer outlet (ProducerOutlet, its stop signal ProducerStopped)
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from queue import Queue, Empty
from typing import Callable, Iterator, Tuple, TypeVar

from .schema import RuntimeConfig

logger = logging.getLogger(__name__)

T = TypeVar('T')


# =============================================================================
# MIXIN CLASS
# =============================================================================

class ProducerConsumerMixin:
    """Mixin class that provides producer-consumer functionality to Luigi tasks; each host task declares the
    runtime section it reads (a SectionParameter of RuntimeConfig)."""

    runtime: RuntimeConfig
    
    def run_producer_consumer(
        self,
        producer_func: Callable[[ProducerOutlet], None],
        consumer_func: Callable[[T], bool],
        queue_type: type = Queue,
        **queue_kwargs
    ) -> Tuple[int, int]:
        """Run a producer-consumer pipeline using the task's configuration.
        
        Args:
            producer_func: Function that produces items
            consumer_func: Function that consumes items, returns success bool
            queue_type: Type of queue to use
            **queue_kwargs: Additional arguments for queue initialization
            
        Returns:
            Tuple of (total_items, successful_items)
        """
        queue_maxsize, queue_get_timeout = self.runtime.queue_maxsize, self.runtime.queue_get_timeout
        
        return create_producer_consumer_pipeline(
            producer_func, consumer_func, queue_type,
            queue_maxsize, queue_get_timeout, **queue_kwargs
        )


# =============================================================================
# PIPELINE CREATION
# =============================================================================

def create_producer_consumer_pipeline(
    producer_func: Callable[[ProducerOutlet], None],
    consumer_func: Callable[[T], bool],
    queue_type: type = Queue,
    queue_maxsize: int = 32,
    queue_get_timeout: int = 20,
    **queue_kwargs
) -> Tuple[int, int]:
    """Create and run a producer-consumer pipeline with common error handling.

    The producer runs in a one-worker thread pool and puts its items through a ProducerOutlet,
    so its failure reaches the caller through its future, re-raised here once, and no thread
    hook reports it. The consume loop ends only on the sentinel: once the producer has ended,
    an item it put before ending is still read, and an empty queue raises RuntimeError (a
    producer that ended without the sentinel). When the loop does not complete (a consumer
    failure, the missing sentinel, an interrupt), the outlet stops: its queued items are
    discarded once, which completes a put blocked on a full queue, and the producer's next put
    raises ProducerStopped; the pipeline then waits for the producer and raises its own failure,
    if any, with the consumer side's exception as its context. Control leaves only once the
    producer has ended.

    Args:
        producer_func: Function that produces items and puts them through the outlet
        consumer_func: Function that consumes items from queue, returns success bool
        queue_type: Type of queue to use (default: Queue)
        queue_maxsize: Maximum size of the queue
        queue_get_timeout: Timeout for queue.get() in seconds
        **queue_kwargs: Additional arguments for queue initialization

    Returns:
        Tuple of (total_items, successful_items)
    """
    queue = queue_type(maxsize=queue_maxsize, **queue_kwargs)
    outlet = ProducerOutlet(queue)
    total_items = 0
    successful_items = 0
    with ThreadPoolExecutor(max_workers=1) as executor:
        producer = None
        completed = False
        try:
            producer = executor.submit(producer_func, outlet)
            while True:
                try:
                    item = queue.get(timeout=queue_get_timeout)
                except Empty:
                    if not producer.done():
                        continue
                    # A done future's every put is complete: an empty queue now means no sentinel will come
                    if queue.empty():
                        raise RuntimeError("the producer ended without putting the sentinel") from None
                    item = queue.get_nowait()

                if item is None:
                    break

                total_items += 1
                if consumer_func(item):
                    successful_items += 1
            completed = True
        finally:
            if not completed:
                outlet.stop()
                if producer is not None:
                    failure = producer.exception()
                    if failure is not None and not isinstance(failure, ProducerStopped):
                        raise failure
        producer.result()
    logger.info(f"Processed {total_items} items, successful: {successful_items}")
    return total_items, successful_items


# =============================================================================
# PRODUCER WRAPPER
# =============================================================================

def create_producer_with_sentinel(
    producer_func: Callable[[], None],
    queue: ProducerOutlet
) -> None:
    """Wrapper for producer functions: the sentinel follows them however they end, their failure propagates.

    Args:
        producer_func: The actual producer function
        queue: The producer's outlet to put items into
    """
    try:
        producer_func()
    finally:
        queue.put(None)


# =============================================================================
# CONSUMER FAILURE RECORD
# =============================================================================

@contextmanager
def logged_series_failure(series_id: str) -> Iterator[None]:
    """A queue consumer's body: a failure is logged once at ERROR, naming the series, then re-raised."""
    try:
        yield
    except Exception as e:
        logger.error(f"Error processing series {series_id}: {e}")
        raise


# =============================================================================
# PRODUCER OUTLET
# =============================================================================

class ProducerOutlet:
    """The producer's only end of the queue: put enqueues until the pipeline stops the outlet."""

    def __init__(self, queue: Queue) -> None:
        self._queue = queue
        self._stopped = False

    def put(self, item: object) -> None:
        """Enqueue the item; once stopped, drop the sentinel (None) and raise ProducerStopped for any other item."""
        if self._stopped:
            if item is None:
                return
            raise ProducerStopped("the pipeline stopped its producer")
        self._queue.put(item)

    def stop(self) -> None:
        """Stop the outlet, then discard the queued items once, which completes a put blocked on a full queue.

        The consumer side is the queue's only getter, so a queue that is not empty always yields an item.
        """
        self._stopped = True
        while not self._queue.empty():
            self._queue.get_nowait()


class ProducerStopped(RuntimeError):
    """Raised by ProducerOutlet.put once the pipeline has stopped the outlet; the pipeline never reports it."""
