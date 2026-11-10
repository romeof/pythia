"""Regression for the consume loop that ends only on the sentinel (G-12, F-30).

One child program runs create_producer_consumer_pipeline with queue_get_timeout 1, a queue_type
whose put sets an event after the sentinel, and the pipeline call timed inside the child by
time.perf_counter (its DURATION line, written in a finally; the monotonic clock's 15.6 ms tick
could read a 3 s call short), so interpreter start and imports never count. Case (a),
the hang test: a producer wrapped by create_producer_with_sentinel sleeps 3 s, then puts two
items. Both liveness predicates are patched for calls from the main thread only,
threading.Thread.is_alive (the thread of the code before G-8) and concurrent.futures.Future.done
(the future after it): the patch waits for the event (at most 5 s), then until the original
predicate reports the producer ended (at most 5 s), writes "PREDICATE ended qsize=<n>", flushed
at once, and returns the original answer. Verdicts: exactly one "PREDICATE ended qsize=3" line
(the race was driven), the call lasts at least 3 s and under 20 s within the 120 s bound, and it
logs "Processed 2 items". The code before G-12 ends its loop on the predicate with both items
and the sentinel still queued, then waits in queue.join() for task_done calls that never come:
the child times out. Case (b): a bare producer that returns without the sentinel: the
RuntimeError's exception line exactly once, and the call lasts under 2 s, one wait (before G-12:
the call returns (0, 0) and the child returns 0). Case (c): a bare producer that raises: the last
traceback block names its error, which no other block names, and follows the interpreter's
context line after the block of the missing-sentinel RuntimeError, and no "Exception in thread"
line (before G-12: the thread hook's line, then the producer's error alone).

Structure:
1. main
2. Cases
3. Helpers
"""
import sys

sys.dont_write_bytecode = True

import testkit

KIT = testkit.preamble(__file__)

from typing import Optional

CHILD_TIMEOUT_S = 120.0
PATCH_WAIT_S = 5.0
POLL_S = 0.05
PRODUCER_SLEEP_S = 3.0
RACE_BOUND_S = 20.0
BARE_BOUND_S = 2.0
PREDICATE_PREFIX = "PREDICATE "
DRIVEN_LINE = "PREDICATE ended qsize=3"
DURATION_PREFIX = "DURATION "
PROCESSED_TWO = "Processed 2 items"
MISSING_SENTINEL_LINE = "RuntimeError: the producer ended without putting the sentinel"
PRODUCER_ERROR = "planted producer error"
PRODUCER_LINE = f"OSError: {PRODUCER_ERROR}"
THREAD_LINE = "Exception in thread"
CHILD_CODE = testkit.CHILD_BOOTSTRAP + (
    "import concurrent.futures, logging, queue, threading, time\n"
    "from src.utils.common.producerconsumer import create_producer_consumer_pipeline, create_producer_with_sentinel\n"
    "logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='%(levelname)s %(name)s: %(message)s')\n"
    "case = sys.argv[4]\n"
    "sentinel_put, queues = threading.Event(), []\n"
    "class SignalledQueue(queue.Queue):\n"
    "    def __init__(self, *args, **kwargs):\n"
    "        super().__init__(*args, **kwargs)\n"
    "        queues.append(self)\n"
    "    def put(self, item, *args, **kwargs):\n"
    "        super().put(item, *args, **kwargs)\n"
    "        if item is None:\n"
    "            sentinel_put.set()\n"
    "def driven(original, ended):\n"
    "    def predicate(self):\n"
    "        if threading.current_thread() is not threading.main_thread():\n"
    "            return original(self)\n"
    f"        sentinel_put.wait({PATCH_WAIT_S})\n"
    f"        deadline = time.monotonic() + {PATCH_WAIT_S}\n"
    "        while not ended(original(self)) and time.monotonic() < deadline:\n"
    f"            time.sleep({POLL_S})\n"
    "        answer = original(self)\n"
    f"        sys.stdout.write('{PREDICATE_PREFIX}' + ('ended' if ended(answer) else 'running') + ' qsize='"
    " + str(queues[-1].qsize()) + '\\n')\n"
    "        sys.stdout.flush()\n"
    "        return answer\n"
    "    return predicate\n"
    "def produce(outlet):\n"
    "    if case == 'race':\n"
    f"        time.sleep({PRODUCER_SLEEP_S})\n"
    "        outlet.put(1)\n"
    "        outlet.put(2)\n"
    "    elif case == 'raising':\n"
    f"        raise OSError('{PRODUCER_ERROR}')\n"
    "producer = produce\n"
    "if case == 'race':\n"
    "    threading.Thread.is_alive = driven(threading.Thread.is_alive, lambda alive: not alive)\n"
    "    concurrent.futures.Future.done = driven(concurrent.futures.Future.done, lambda done: done)\n"
    "    producer = lambda outlet: create_producer_with_sentinel(lambda: produce(outlet), outlet)\n"
    "started = time.perf_counter()\n"
    "try:\n"
    "    create_producer_consumer_pipeline(producer, lambda item: True, queue_type=SignalledQueue, queue_get_timeout=1)\n"
    "finally:\n"
    f"    sys.stdout.write('{DURATION_PREFIX}' + str(time.perf_counter() - started) + '\\n')\n"
    "    sys.stdout.flush()\n")


def main() -> None:
    """The three cases, then the summary (return code 1 on any FAIL)."""
    driven_race()
    bare_producer()
    raising_producer()
    KIT.finish()


# =============================================================================
# CASES
# =============================================================================


def driven_race() -> None:
    """Case (a): the driven race, then both items consumed within the bound."""
    child, lines = pipeline_child("race")
    predicates = [line.rstrip() for line in lines if line.startswith(PREDICATE_PREFIX)]
    KIT.verdict(f"case (a): exactly one '{DRIVEN_LINE}' line (the race was driven)", predicates == [DRIVEN_LINE],
                [DRIVEN_LINE], predicates)
    duration = call_duration(lines)
    KIT.verdict(f"case (a): the call lasts at least {PRODUCER_SLEEP_S:.0f} s and under {RACE_BOUND_S:.0f} s",
                duration is not None and PRODUCER_SLEEP_S <= duration < RACE_BOUND_S,
                f"[{PRODUCER_SLEEP_S}, {RACE_BOUND_S}) s", {"duration_s": duration, "timed_out": child.timed_out})
    processed = [line for line in lines if PROCESSED_TWO in line]
    KIT.verdict(f"case (a): the call logs '{PROCESSED_TWO}'", bool(processed), PROCESSED_TWO,
                processed or {"timed_out": child.timed_out})


def bare_producer() -> None:
    """Case (b): a producer without the sentinel: the RuntimeError once, after one wait."""
    child, lines = pipeline_child("bare")
    raised = [line for line in lines if line.startswith(MISSING_SENTINEL_LINE)]
    KIT.verdict("case (b): the missing-sentinel RuntimeError exactly once", len(raised) == 1, 1,
                {"lines": raised, "return_code": child.return_code})
    duration = call_duration(lines)
    KIT.verdict(f"case (b): the call lasts under {BARE_BOUND_S:.0f} s (one wait)",
                duration is not None and duration < BARE_BOUND_S, f"< {BARE_BOUND_S} s", duration)


def raising_producer() -> None:
    """Case (c): a producer that raises without the sentinel: its error once, the RuntimeError as its context."""
    child, lines = pipeline_child("raising")
    blocks = testkit.traceback_blocks(child.text)
    chained = (len(blocks) >= 2 and blocks[-1] == (PRODUCER_LINE, True) and blocks[-2][0] == MISSING_SENTINEL_LINE
               and sum(block[0] == PRODUCER_LINE for block in blocks) == 1)
    KIT.verdict("case (c): the producer's error raised once, the missing-sentinel RuntimeError as its context", chained,
                [MISSING_SENTINEL_LINE, (PRODUCER_LINE, True)], blocks)
    found = [line for line in lines if THREAD_LINE in line]
    KIT.verdict(f"case (c): no '{THREAD_LINE}' line", not found, "none", found[:3])


# =============================================================================
# HELPERS
# =============================================================================


def pipeline_child(case: str) -> tuple[testkit.Child, list[str]]:
    """The child program for one case, under the 120 s bound, with its capture's plain lines."""
    with testkit.scratch(KIT) as folder:
        child = testkit.run_child(KIT, folder, case, ["-c", CHILD_CODE, *testkit.child_args(KIT), case],
                                  timeout_s=CHILD_TIMEOUT_S)
    return child, testkit.plain_lines(child.text)


def call_duration(lines: list[str]) -> Optional[float]:
    """The pipeline call's duration from the child's DURATION line, None without one."""
    found = [float(line[len(DURATION_PREFIX):]) for line in lines if line.startswith(DURATION_PREFIX)]
    return found[-1] if found else None


if __name__ == "__main__":
    main()
