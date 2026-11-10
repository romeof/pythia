"""Regression for the loud per-series failure of the Trigger task (HYP-05, loud-failure half).

Case (a): --task trigger on a six-series realistic copy of the real input with
runtime.queue_maxsize 1 in its dev copy and a planted OSError in every
BaseDetection.analyze_series call, under a 120 s timeout: the planted error is logged, no
[Errno 2] line, and the run returns non-zero within the timeout. The bound of 1 keeps the producer
blocked in put when the consumer fails, so an end that waited for the producer would hang (a
timeout counts as FAIL). (The planted failure replaces an omitted trigger.replay.window_type,
which the schema default now fills, Q11.) Case (b): a child runs create_producer_consumer_pipeline
with queue_maxsize 1, a producer (wrapped by create_producer_with_sentinel) that puts two items
and raises, and a consumer that waits up to five seconds for the producer thread on each item and
then raises: the producer's error is raised once in the propagated traceback, with the consumer's
as its context (testkit.traceback_blocks: the last block names the producer's error and follows
the interpreter's context line, the block before it names the consumer's), no "Error in
producer" record, no "Exception in thread" line, the producer thread has ended when control
leaves the pipeline, then a non-zero return code (G-8). The code before G-8 reports the producer's
failure through its wrapper's record and the thread hook and raises the consumer's error alone;
a pipeline that lets the consumer's failure leave while the producer waits in put leaves that
thread alive.

Structure:
1. main
2. Cases
"""
import sys

sys.dont_write_bytecode = True

import testkit

KIT = testkit.preamble(__file__)

PLANTED_TARGET = "src.utils.trigger.detection:BaseDetection.analyze_series"
ERRNO_LINE = "[Errno 2]"
CASE_A_TIMEOUT_S = 120.0
CASE_B_TIMEOUT_S = 120.0
PRODUCER_ERROR = "planted producer error"
CONSUMER_ERROR = "planted consumer error"
PRODUCER_LINE = f"OSError: {PRODUCER_ERROR}"
CONSUMER_LINE = f"OSError: {CONSUMER_ERROR}"
WRAPPER_RECORD = "Error in producer"
THREAD_LINE = "Exception in thread"
ALIVE_PREFIX = "PRODUCER ALIVE "
PIPELINE_CODE = testkit.CHILD_BOOTSTRAP + (
    "import logging, threading\n"
    "from src.utils.common.producerconsumer import create_producer_consumer_pipeline, create_producer_with_sentinel\n"
    "logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='%(levelname)s %(name)s: %(message)s')\n"
    "producers = []\n"
    "def produce(queue):\n"
    "    queue.put(1)\n"
    "    queue.put(2)\n"
    f"    raise OSError('{PRODUCER_ERROR}')\n"
    "def consume(item):\n"
    "    producers.append(next(thread for thread in threading.enumerate() if thread is not threading.main_thread()))\n"
    "    producers[-1].join(timeout=5.0)\n"
    f"    raise OSError('{CONSUMER_ERROR}')\n"
    "try:\n"
    "    create_producer_consumer_pipeline(lambda queue: create_producer_with_sentinel(lambda: produce(queue), queue),\n"
    "                                      consume, queue_maxsize=1, queue_get_timeout=1)\n"
    "finally:\n"
    f"    sys.stdout.write('{ALIVE_PREFIX}' + str(producers[-1].is_alive()) + '\\n')\n")


def main() -> None:
    """Both cases, then the summary (return code 1 on any FAIL)."""
    planted_consumer_failure()
    blocked_producer()
    KIT.finish()


# =============================================================================
# CASES
# =============================================================================


def planted_consumer_failure() -> None:
    """Case (a): the planted consumer error, no input error, a non-zero return code within the timeout."""
    with testkit.scratch(KIT) as folder:
        config_file = testkit.config_copy(KIT, folder, strategy={"trigger.replay.mode": "realistic"},
                                          dev={"runtime.queue_maxsize": 1})
        child = testkit.planted_run(KIT, folder, "planted", config_file, target=PLANTED_TARGET, task="trigger",
                                    timeout_s=CASE_A_TIMEOUT_S)
        lines = testkit.plain_lines(child.text)
        KIT.verdict("case (a): the planted consumer error is logged", any(testkit.PLANTED_ERROR in line for line in lines),
                    testkit.PLANTED_ERROR, sum(testkit.PLANTED_ERROR in line for line in lines))
        KIT.verdict("case (a): the input was read (no [Errno 2])", not any(ERRNO_LINE in line for line in lines), "none",
                    [line for line in lines if ERRNO_LINE in line][:2])
        KIT.verdict("case (a): return != 0 within the timeout", child.return_code not in (0, None), "return != 0",
                    {"return_code": child.return_code, "timed_out": child.timed_out})


def blocked_producer() -> None:
    """Case (b): the producer's error raised once with the consumer's as its context, neither the wrapper's record nor
    the thread hook's line, no producer thread left, then a non-zero return code."""
    with testkit.scratch(KIT) as folder:
        child = testkit.run_child(KIT, folder, "blocked", ["-c", PIPELINE_CODE, *testkit.child_args(KIT)],
                                  timeout_s=CASE_B_TIMEOUT_S)
        lines = testkit.plain_lines(child.text)
        blocks = testkit.traceback_blocks(child.text)
        chained = (len(blocks) >= 2 and blocks[-1] == (PRODUCER_LINE, True) and blocks[-2][0] == CONSUMER_LINE
                   and sum(block[0] == PRODUCER_LINE for block in blocks) == 1)
        KIT.verdict("case (b): the producer's error raised once, the consumer's as its context", chained,
                    [CONSUMER_LINE, (PRODUCER_LINE, True)], blocks)
        for name, fragment in (("wrapper's record", WRAPPER_RECORD), ("thread hook's line", THREAD_LINE)):
            found = [line for line in lines if fragment in line]
            KIT.verdict(f"case (b): no '{fragment}' line (the {name})", not found, "none", found[:3])
        alive = [line[len(ALIVE_PREFIX):] for line in lines if line.startswith(ALIVE_PREFIX)]
        KIT.verdict("case (b): the producer thread has ended when control leaves the pipeline", alive == ["False"],
                    ["False"], alive)
        KIT.verdict("case (b): return != 0", child.return_code not in (0, None), "return != 0",
                    {"return_code": child.return_code, "timed_out": child.timed_out})


if __name__ == "__main__":
    main()
