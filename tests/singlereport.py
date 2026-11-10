"""Regression for one report per failure and the producer's stop however the pipeline ends (G-8, G-10).

Case (a): a testkit.planted_run child, --task trigger on a one-series realistic copy with a
planted OSError in Read.iter_series (the Trigger producer): exactly one ERROR record names the
planted error (the FAILURE record), no "Error in producer", "Exception in thread" or
"Error opening HDF5 store" line, exactly two traceback lines (the FAILURE record's and the uncaught
RuntimeError's), return != 0. Case (b): the same with the planted OSError in BaseDetection.analyze_series (a
consumer): exactly two ERROR records name the planted error (the series record, then the FAILURE
record), no thread, wrapper or store line, one FAILURE record, return != 0. Case (c): a child runs
create_producer_consumer_pipeline with queue_maxsize 1, a wrapped producer that puts and counts
50 items and a consumer that raises KeyboardInterrupt on its first item, under a 60 s bound: the
child ends within the bound, the producer thread (found from the consumer) has ended when control
leaves the pipeline, the producer put fewer than 50 items, and the exception that leaves the
child is the consumer's KeyboardInterrupt (the exception line of testkit.traceback_blocks' last
item), with no line naming ProducerStopped, the stop's own signal. The child's finally flushes
stdout after its status lines, since the capture merges its block-buffered stdout with its
stderr. The return code is only required != 0: an uncaught KeyboardInterrupt ends the process the
way a Ctrl-C does. Case (d): a testkit.planted_run child as in (a), with pandas' HDFStore.put
planted to raise for a key with the prefix series_, so the Trigger consumer's first store write
(/series_0) fails inside put_dataframe_normalized: exactly three ERROR records name the planted error (the
write record naming its key, the series record, then the FAILURE record), return != 0. The code
before G-8 reports a producer failure through its wrapper, the thread hook, the store line and
the FAILURE record, and leaves its daemon producer blocked in put after an interrupt; the code
before G-10 adds the store line to a failed store write.

Structure:
1. main
2. Cases
3. Helpers
"""
import sys

sys.dont_write_bytecode = True

import testkit

KIT = testkit.preamble(__file__)

PRODUCER_TARGET = "src.utils.preprocess.read:Read.iter_series"
CONSUMER_TARGET = "src.utils.trigger.detection:BaseDetection.analyze_series"
WRITE_TARGET = "pandas.io.pytables:HDFStore.put"
SERIES_KEY_PREFIX = "series_"
WRAPPER_RECORD = "Error in producer"
THREAD_LINE = "Exception in thread"
STORE_LINE = "Error opening HDF5 store"
# put_dataframe_normalized's record for the frame of series 0, the one series ONE_SERIES keeps
WRITE_RECORD = "Error storing DataFrame at key '/series_0':"
STOP_SIGNAL = "ProducerStopped"
EXPECTED_TRACEBACKS = 2
ITEMS = 50
CASE_C_TIMEOUT_S = 60.0
CONSUMER_INTERRUPT = "planted consumer interrupt"
INTERRUPT_LINE = f"KeyboardInterrupt: {CONSUMER_INTERRUPT}"
ALIVE_PREFIX = "PRODUCER ALIVE "
PUTS_PREFIX = "PUTS "
PIPELINE_CODE = testkit.CHILD_BOOTSTRAP + (
    "import threading\n"
    "from src.utils.common.producerconsumer import create_producer_consumer_pipeline, create_producer_with_sentinel\n"
    "puts, producers = [], []\n"
    "def produce(queue):\n"
    f"    for item in range({ITEMS}):\n"
    "        queue.put(item)\n"
    "        puts.append(item)\n"
    "def consume(item):\n"
    "    producers.append(next(thread for thread in threading.enumerate() if thread is not threading.main_thread()))\n"
    f"    raise KeyboardInterrupt('{CONSUMER_INTERRUPT}')\n"
    "try:\n"
    "    create_producer_consumer_pipeline(lambda queue: create_producer_with_sentinel(lambda: produce(queue), queue),\n"
    "                                      consume, queue_maxsize=1, queue_get_timeout=1)\n"
    "finally:\n"
    f"    sys.stdout.write('{ALIVE_PREFIX}' + str(producers[-1].is_alive()) + '\\n')\n"
    f"    sys.stdout.write('{PUTS_PREFIX}' + str(len(puts)) + '\\n')\n"
    "    sys.stdout.flush()\n")


def main() -> None:
    """The four cases, then the summary (return code 1 on any FAIL)."""
    producer_failure()
    consumer_failure()
    interrupted_pipeline()
    store_write_failure()
    KIT.finish()


# =============================================================================
# CASES
# =============================================================================


def producer_failure() -> None:
    """Case (a): one ERROR record names the producer's planted error, no duplicate line, two tracebacks, return != 0."""
    with testkit.scratch(KIT) as folder:
        config_file = testkit.config_copy(KIT, folder, strategy=dict(testkit.ONE_SERIES))
        child = testkit.planted_run(KIT, folder, "producer", config_file, target=PRODUCER_TARGET, task="trigger")
        lines = testkit.plain_lines(child.text)
        records = [line for line in lines if line.startswith(testkit.ERROR_PREFIX) and testkit.PLANTED_ERROR in line]
        KIT.verdict("case (a): exactly one ERROR record names the planted error (the FAILURE record)",
                    len(records) == 1 and testkit.HANDLER_PREFIX in records[0], 1, records)
        duplicate_lines(lines, "case (a)", (WRAPPER_RECORD, THREAD_LINE, STORE_LINE))
        tracebacks = [line for line in lines if line.rstrip() == testkit.TRACEBACK_LINE]
        KIT.verdict(f"case (a): exactly {EXPECTED_TRACEBACKS} traceback lines (the FAILURE record's and the uncaught one)",
                    len(tracebacks) == EXPECTED_TRACEBACKS, EXPECTED_TRACEBACKS, len(tracebacks))
        KIT.verdict("case (a): return != 0", child.return_code not in (0, None), "return != 0",
                    {"return_code": child.return_code, "timed_out": child.timed_out})


def consumer_failure() -> None:
    """Case (b): a consumer's planted error: series and FAILURE records, no thread, wrapper or store line, return != 0."""
    with testkit.scratch(KIT) as folder:
        config_file = testkit.config_copy(KIT, folder, strategy=dict(testkit.ONE_SERIES))
        child = testkit.planted_run(KIT, folder, "consumer", config_file, target=CONSUMER_TARGET, task="trigger")
        lines = testkit.plain_lines(child.text)
        error_records = [line for line in lines if line.startswith(testkit.ERROR_PREFIX) and testkit.PLANTED_ERROR in line]
        KIT.verdict("case (b): exactly two ERROR records name the planted error (the series record, then the FAILURE record)",
                    len(error_records) == 2 and testkit.SERIES_RECORD in error_records[0]
                    and testkit.HANDLER_PREFIX in error_records[1], 2, error_records)
        duplicate_lines(lines, "case (b)", (THREAD_LINE, WRAPPER_RECORD, STORE_LINE))
        records = [line for line in lines if testkit.HANDLER_PREFIX in line]
        KIT.verdict(f"case (b): one '{testkit.HANDLER_PREFIX}' record names the planted error",
                    len(records) == 1 and testkit.PLANTED_ERROR in records[0], 1, records)
        KIT.verdict("case (b): return != 0", child.return_code not in (0, None), "return != 0",
                    {"return_code": child.return_code, "timed_out": child.timed_out})


def interrupted_pipeline() -> None:
    """Case (c): an interrupt in the consumer: the child ends in time, its producer ended early, the interrupt leaves."""
    with testkit.scratch(KIT) as folder:
        child = testkit.run_child(KIT, folder, "interrupted", ["-c", PIPELINE_CODE, *testkit.child_args(KIT)],
                                  timeout_s=CASE_C_TIMEOUT_S)
        lines = testkit.plain_lines(child.text)
        KIT.verdict(f"case (c): the child ends within {CASE_C_TIMEOUT_S:.0f} s, return != 0",
                    not child.timed_out and child.return_code not in (0, None), "ended, return != 0",
                    {"return_code": child.return_code, "timed_out": child.timed_out})
        alive = [line[len(ALIVE_PREFIX):] for line in lines if line.startswith(ALIVE_PREFIX)]
        KIT.verdict("case (c): the producer thread has ended when control leaves the pipeline", alive == ["False"],
                    ["False"], alive)
        puts = [int(line[len(PUTS_PREFIX):]) for line in lines if line.startswith(PUTS_PREFIX)]
        KIT.verdict(f"case (c): the producer put fewer than {ITEMS} items", len(puts) == 1 and puts[0] < ITEMS,
                    f"< {ITEMS}", puts)
        blocks = testkit.traceback_blocks(child.text)
        KIT.verdict("case (c): the consumer's KeyboardInterrupt leaves the child", bool(blocks) and blocks[-1][0] == INTERRUPT_LINE,
                    INTERRUPT_LINE, blocks[-1:])
        signals = [line for line in lines if STOP_SIGNAL in line]
        KIT.verdict(f"case (c): no line names {STOP_SIGNAL}", not signals, "none", signals[:3])


def store_write_failure() -> None:
    """Case (d): a consumer's failed store write: write, series and FAILURE records name the error, return != 0."""
    with testkit.scratch(KIT) as folder:
        config_file = testkit.config_copy(KIT, folder, strategy=dict(testkit.ONE_SERIES))
        child = testkit.planted_run(KIT, folder, "write", config_file, target=WRITE_TARGET, task="trigger",
                                    key_prefix=SERIES_KEY_PREFIX)
        error_records = [line for line in testkit.plain_lines(child.text)
                         if line.startswith(testkit.ERROR_PREFIX) and testkit.PLANTED_ERROR in line]
        KIT.verdict("case (d): exactly three ERROR records name the planted error (the write record naming its key, "
                    "the series record, then the FAILURE record)",
                    len(error_records) == 3 and WRITE_RECORD in error_records[0]
                    and testkit.SERIES_RECORD in error_records[1] and testkit.HANDLER_PREFIX in error_records[2],
                    3, error_records)
        KIT.verdict("case (d): return != 0", child.return_code not in (0, None), "return != 0",
                    {"return_code": child.return_code, "timed_out": child.timed_out})


# =============================================================================
# HELPERS
# =============================================================================


def duplicate_lines(lines: list[str], case: str, fragments: tuple) -> None:
    """One verdict per fragment of a duplicate report: no line holds it."""
    for fragment in fragments:
        found = [line for line in lines if fragment in line]
        KIT.verdict(f"{case}: no '{fragment}' line", not found, "none", found[:3])


if __name__ == "__main__":
    main()
