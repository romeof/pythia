"""Regression for the reference keys iterated in sorted sequence (G-14, F-18).

On a two-series realistic copy with read.reference_series ["0", "1"]: a first --task all, then
store children launched one seed at a time from seed 1 up to seed 32 (testkit.run_child's
hashseed), each logging list(set(store.keys())) of that run's signal store, the set production
iterates; the launches stop at the first seed whose /reference_series_* keys come out in another
sequence than an earlier seed's, and that pair is recorded with both sequences (none by seed 32:
FAIL, the case would be vacuous). Then --task all under each seed of the pair
(testkit.run_pipeline's hashseed): /gradient_reference_variation and /gradient_terminal_attenuations
equal row by row. The code before G-14 iterates the key set, so under the pair the two tables
come out as a row permutation.

Structure:
1. main
2. Helpers
"""
import sys

sys.dont_write_bytecode = True

import testkit

KIT = testkit.preamble(__file__)

import json
from pathlib import Path
from typing import Optional

import pandas as pd

TWO_SERIES = {"trigger.replay.mode": "realistic", "read.skip_series": ["2", "3", "4", "5"],
              "read.reference_series": ["0", "1"]}
SEEDS = range(1, 33)
REFERENCE_PREFIX = "/reference_series_"
SEQUENCE_PREFIX = "SEQUENCE "
TABLES = ("/gradient_reference_variation", "/gradient_terminal_attenuations")
KEYS_CODE = testkit.CHILD_BOOTSTRAP + (
    "\nimport json\n"
    "import pandas as pd\n"
    "with pd.HDFStore(sys.argv[4], mode='r') as store:\n"
    "    keys = list(set(store.keys()))\n"
    f"sys.stdout.write('{SEQUENCE_PREFIX}' + json.dumps(keys) + '\\n')\n")


def main() -> None:
    """The first run, the seed pair, the two seeded runs' tables, then the summary (return code 1 on any FAIL)."""
    with testkit.scratch(KIT) as folder:
        config_file = testkit.config_copy(KIT, folder, strategy=dict(TWO_SERIES))
        first = testkit.run_pipeline(KIT, folder, "first", config_file, ["all"])
        KIT.verdict("the first --task all returns 0", first.return_code == 0, 0, first.return_code)
        pair = seed_pair(folder, testkit.output_file(config_file, "signal"))
        if pair is not None:
            tables = [seeded_tables(folder, config_file, seed) for seed in pair]
            for key in TABLES:
                KIT.verdict(f"{key} equal row by row under seeds {pair[0]} and {pair[1]}",
                            tables[0][key].equals(tables[1][key]), "equal",
                            {seed: found[key].iloc[:, 0].drop_duplicates().tolist() for seed, found in zip(pair, tables)})
    KIT.finish()


# =============================================================================
# HELPERS
# =============================================================================


def seed_pair(folder: Path, signal_file: Path) -> Optional[tuple[int, int]]:
    """Seeds from 1 up, one store child each: the first seed whose reference keys come out in another sequence than
    an earlier seed's, with that earlier seed; None (a FAIL verdict) when no seed up to 32 does."""
    sequences: dict[int, list[str]] = {}
    for seed in SEEDS:
        child = testkit.run_child(KIT, folder, f"keys{seed}", ["-c", KEYS_CODE, *testkit.child_args(KIT),
                                                                 str(signal_file)], hashseed=seed)
        logged = [json.loads(line[len(SEQUENCE_PREFIX):]) for line in testkit.plain_lines(child.text)
                  if line.startswith(SEQUENCE_PREFIX)]
        references = [key for key in (logged[0] if logged else []) if key.startswith(REFERENCE_PREFIX)]
        earlier = next((other for other, seen in sequences.items() if seen != references), None)
        sequences[seed] = references
        if earlier is not None:
            KIT.verdict(f"seeds {earlier} and {seed} iterate the signal store's reference keys in two sequences",
                        len(sequences[earlier]) == len(references) == 2, "two reference keys in two sequences",
                        {earlier: sequences[earlier], seed: references})
            return earlier, seed
    KIT.verdict(f"a seed among {SEEDS.start}..{SEEDS.stop - 1} iterates the reference keys in another sequence", False,
                "a seed pair", sequences)
    return None


def seeded_tables(folder: Path, config_file: Path, seed: int) -> dict[str, pd.DataFrame]:
    """--task all under the seed, then the two tables of its performance store."""
    child = testkit.run_pipeline(KIT, folder, f"seed{seed}", config_file, ["all"], hashseed=seed)
    KIT.verdict(f"--task all under seed {seed} returns 0", child.return_code == 0, 0, child.return_code)
    with pd.HDFStore(str(testkit.output_file(config_file, "performance")), mode="r") as store:
        return {key: store.get(key) for key in TABLES}


if __name__ == "__main__":
    main()
