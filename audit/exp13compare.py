"""EXP-13: value-by-value comparer of two output trees (CMP-1..3, EXP-07, EXP-22).

Per .h5 file: key sets, then per key the storer format, the type, the index (Index.equals
and its dtype), the columns, the dtypes per column label, DataFrame.equals (NaN-aware) and
the axis names that equals ignores, with float differences characterised by their maximum
absolute and relative difference; other files are byte-compared; git_info.txt is excepted.
Every difference is reported; nothing raises on a difference.
--rename PATH applies a name map (harness.rename_map) to the left tree in memory before each
comparison: its store keys, the frame columns its drop_columns lists (dropped before any
other change to a frame), column labels, the string cells of the columns the map names and
the keys of each parsed .yaml file, which is then compared parsed (one yaml difference per
file naming each differing key path); .png bytes are skipped; every difference names the
left side as written and as mapped.
--self-test plants one difference of every kind into a scratch copy under data\\exp\\13
(the columns kind twice: a column on one side only and a reversed column sequence) and
requires each to be reported alone, an identical copy to report none, and two frames
with a repeated column label to compare without raising; its rename part plants a neutral
map's pairs on every occurrence in a second copy (plant_renamed_key, plant_renamed_column,
plant_renamed_value, plant_renamed_yaml_key) and drops its dropped column from every frame
but the first that holds it (plant_dropped_column), with a negative control per kind
(plant_unmapped_cell, plant_unmapped_yaml_key, plant_removed_png; the frame that keeps the
dropped column), and checks each at its site.

Structure:
1. main
2. Self-test: the planted differences, then the rename part
3. Tree and file comparison
4. Frame comparison
"""
import sys

sys.dont_write_bytecode = True

import os

sys.path.insert(0, os.environ["PYTHIA_AUDIT_HARNESS"])
import harness

CTX = harness.preamble(__file__, imports_src=False)

import argparse
import dataclasses
import json
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

EXCEPTED_FILES = ("git_info.txt",)
H5_SUFFIX = ".h5"
YAML_SUFFIX = ".yaml"
PNG_SUFFIX = ".png"
TRIPLE_LIMIT = 10
EXPERIMENT = "13"
PLANT_FLOAT = ("realistic/performance/0.h5", "/total_variation", "total_variation")
PLANT_DTYPE = ("ideal/trigger/0.h5", "/triggers_0", "trigger_start_values")
PLANT_DROP = ("ideal/signal/0.h5", "/strategy_name")
PLANT_PNG = "ideal/performance/plots/gradient/gradient.png"
PLANT_REMOVED = "ideal/trigger/0_preprocess_report.txt"
PLANT_FORMAT = ("realistic/performance/0.h5", "/total_variation_attenuated")
PLANT_TYPE = ("ideal/performance/0.h5", "/gradient_terminal_attenuations", "amount")
PLANT_INDEX_DTYPE = ("ideal/trigger/0.h5", "/triggers_1")
PLANT_ONE_SIDED = ("ideal/trigger/0.h5", "/triggers_2", "is_true")
PLANT_REVERSED = ("ideal/trigger/0.h5", "/triggers_3")
PLANT_NAMES = ("ideal/trigger/0.h5", "/triggers_4", "planted")
OTHER_FORMAT = {"fixed": "table", "table": "fixed"}
REPEATED_LABEL = "repeated"
# the rename part: a map of neutral pairs over names the reference tree holds, a column of every /series_ frame to
# drop, and one negative control per kind
RENAME_PAIRS = {"keys": {"/strategy_name": "/planted_name"}, "columns": {"trials_to_true": "planted_column"},
                "values": {"name": {"acr": "planted_value"}}, "yaml_keys": {"strategy": "planted_section"},
                "drop_columns": ["turning_points"]}
UNMAPPED_CELL = ("ideal/performance/0.h5", "/gradient_metrics_table", "metric_key")
UNMAPPED_YAML = ("dev.yaml", ("trigger", "planted_section", "plot_debug"), "planted_debug")
UNMAPPED_YAML_WRITTEN = "trigger.strategy.plot_debug"
YAML_PLANT_SITE = "extrema.yaml"
REMOVED_PNG = "realistic/performance/plots/gradient/gradient.png"
RENAME_MAP_FOLDER = "rename_map"
RENAME_MAP_FILE = "map.json"


@dataclasses.dataclass(frozen=True)
class Difference:
    """One reported difference between the two trees."""

    kind: str
    path: str
    key: str
    detail: str


@dataclasses.dataclass
class Tally:
    """Compared items, for the non-vacuity check."""

    files: int = 0
    h5_files: int = 0
    keys: int = 0
    h5_bytes_differ: int = 0
    png_skipped: int = 0


def main() -> None:
    """Compare two trees, under a name map with --rename, or run the self-test against a source tree."""
    args = _parse_args()
    if args.self_test is not None:
        self_test(args.self_test)
    else:
        mapping = harness.rename_map(args.rename) if args.rename is not None else None
        differences, tally = compare_trees(args.left, args.right, h5_only=args.h5_only, mapping=mapping)
        report(args.left, args.right, differences, tally)
    CTX.summary()


def _parse_args() -> argparse.Namespace:
    """Two trees (with --h5-only and --rename PATH), or --self-test SOURCE."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("left", nargs="?", type=Path)
    parser.add_argument("right", nargs="?", type=Path)
    parser.add_argument("--h5-only", action="store_true")
    parser.add_argument("--rename", type=Path, metavar="PATH")
    parser.add_argument("--self-test", type=Path, metavar="SOURCE")
    args = parser.parse_args()
    if args.self_test is None and (args.left is None or args.right is None):
        parser.error("give two trees or --self-test SOURCE")
    return args


def report(left: Path, right: Path, differences: list[Difference], tally: Tally) -> None:
    """Log every difference and the counts; the comparison verdict is identical or not."""
    for difference in differences:
        CTX.log.info("DIFF %s %s %s: %s", difference.kind, difference.path, difference.key, difference.detail)
    CTX.log.info("COMPARED %s vs %s: %d files (%d .h5, %d keys, %d .png skipped), %d .h5 files with differing bytes, "
                 "%d differences", left, right, tally.files, tally.h5_files, tally.keys, tally.png_skipped,
                 tally.h5_bytes_differ, len(differences))
    CTX.verdict("non-vacuity", tally.files > 0 and (tally.h5_files == 0 or tally.keys > 0), "items compared > 0",
                dataclasses.asdict(tally))
    CTX.verdict("identical value by value", not differences, "no difference", f"{len(differences)} differences")


# =============================================================================
# SELF-TEST
# =============================================================================


def self_test(source: Path) -> None:
    """Plant one difference of every kind, each in its own file or key; frames with a repeated column label; then the
    rename part."""
    folder = CTX.exp_dir(EXPERIMENT)
    left, right = _scratch_pair(folder, source, "left", "right")
    identical, tally = compare_trees(left, right, h5_only=False)
    CTX.verdict("self-test: identical copies", not identical and tally.keys > 0, "no difference, keys compared",
                f"{len(identical)} differences, {tally.keys} keys")
    params = _params_dir(right)
    plant_float(params)
    plant_dtype(params)
    plant_drop(params)
    plant_png(params)
    plant_removed(params)
    plant_format(params)
    plant_type(params)
    plant_index_dtype(params)
    plant_one_sided(params)
    plant_reversed(params)
    plant_names(params)
    found, _ = compare_trees(left, right, h5_only=False)
    expected = {  # name: (kind, file, key, detail fragment) of the one difference each plant must cause at its site
        "flipped float": ("values", PLANT_FLOAT[0], PLANT_FLOAT[1], PLANT_FLOAT[2]),
        "changed dtype": ("dtypes", PLANT_DTYPE[0], PLANT_DTYPE[1], PLANT_DTYPE[2]),
        "dropped key": ("h5-keys", PLANT_DROP[0], "", PLANT_DROP[1]),
        "changed png byte": ("bytes", PLANT_PNG, "", ""),
        "removed file": ("file-set", PLANT_REMOVED, "", ""),
        "other storer format": ("format", *PLANT_FORMAT, ""),
        "frame stored as a Series": ("type", *PLANT_TYPE[:2], ""),
        "index dtype only": ("index", *PLANT_INDEX_DTYPE, ""),
        "column on one side only": ("columns", *PLANT_ONE_SIDED),
        "reversed column sequence": ("columns", *PLANT_REVERSED, ""),
        "index name": ("names", *PLANT_NAMES),
    }
    for name, (kind, file, key, fragment) in expected.items():
        at_site = [difference for difference in found if difference.path.endswith(file) and difference.key == key]
        CTX.verdict(f"self-test: {name} reported alone",
                    [difference.kind for difference in at_site] == [kind] and fragment in at_site[0].detail, [kind],
                    [dataclasses.astuple(difference) for difference in at_site])
    CTX.verdict("self-test: nothing else reported", len(found) == len(expected), len(expected),
                [dataclasses.astuple(difference) for difference in found])
    repeated = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], columns=[REPEATED_LABEL, REPEATED_LABEL])
    changed = repeated.copy()
    changed.iat[0, 1] = 5.0
    kinds = [kind for kind, _ in compare_frames(repeated, changed)]
    CTX.verdict("self-test: a repeated column label compares without raising", kinds == ["values"], ["values"], kinds)
    rename_self_test(folder, source)


def _scratch_pair(folder: Path, source: Path, *names: str) -> tuple[Path, ...]:
    """Fresh copies of the source tree under the experiment folder; an existing one is refused."""
    trees = tuple(folder / name for name in names)
    CTX.guard(*trees)
    for tree in trees:
        if tree.exists():
            harness.refuse(CTX.log, "scratch", f"{tree} already exists")
        shutil.copytree(source, tree)
    return trees


def rename_self_test(folder: Path, source: Path) -> None:
    """RENAME_PAIRS planted on every occurrence in a right copy: no difference under the map at each pair's sites and at
    least one without it; its dropped column gone from every frame but the first that holds it: no difference under the
    map at those sites and one columns difference at each without it; under the map, exactly one difference for each
    negative control, the flipped png skipped."""
    left, right = _scratch_pair(folder, source, "rename_left", "rename_right")
    map_folder = folder / RENAME_MAP_FOLDER
    CTX.guard(map_folder)
    if map_folder.exists():
        harness.refuse(CTX.log, "scratch", f"{map_folder} already exists")
    map_folder.mkdir()
    with open(map_folder / RENAME_MAP_FILE, "x", encoding="utf-8") as sink:
        json.dump(RENAME_PAIRS, sink)
    mapping = harness.rename_map(map_folder / RENAME_MAP_FILE)
    params, prefix = _params_dir(right), _params_dir(right).relative_to(right).as_posix() + "/"
    (old_key, new_key), = RENAME_PAIRS["keys"].items()
    (old_column, new_column), = RENAME_PAIRS["columns"].items()
    (value_column, pairs), = RENAME_PAIRS["values"].items()
    (old_value, new_value), = pairs.items()
    (old_yaml, new_yaml), = RENAME_PAIRS["yaml_keys"].items()
    dropped, = RENAME_PAIRS["drop_columns"]
    key_files = {prefix + rel for rel in plant_renamed_key(params, old_key, new_key)}
    column_sites = {(prefix + rel, key) for rel, key in plant_renamed_column(params, old_column, new_column)}
    value_sites = {(prefix + rel, key) for rel, key in plant_renamed_value(params, value_column, old_value, new_value)}
    yaml_files = {prefix + rel for rel in plant_renamed_yaml_key(params, old_yaml, new_yaml)}
    drop_sites = [(prefix + rel, key) for rel, key in plant_dropped_column(params, dropped)]
    plant_unmapped_cell(params, old_value, new_value)
    plant_unmapped_yaml_key(params)
    plant_png(params)
    plant_removed_png(params)
    found, _ = compare_trees(left, right, h5_only=False, mapping=mapping)
    plain, _ = compare_trees(left, right, h5_only=False)
    sites = {  # kind: (the planted sites, the difference kind each must cause without the map, the predicate of a site)
        "key": (key_files, "h5-keys", lambda item: item.path in key_files and (item.kind == "h5-keys" or new_key in item.key)),
        "column": (column_sites, "columns", lambda item: (item.path, item.key) in column_sites),
        "value": (value_sites, "values", lambda item: (item.path, item.key) in value_sites),
        "yaml key": (yaml_files, "bytes", lambda item: item.path == prefix + YAML_PLANT_SITE),
    }
    for kind, (planted, plain_kind, at_site) in sites.items():
        mapped = [dataclasses.astuple(item) for item in found if at_site(item)]
        unmapped = sorted({item.kind for item in plain if at_site(item)})
        CTX.verdict(f"self-test rename: {kind} pair planted and no difference at its sites under the map",
                    bool(planted) and not mapped, "planted, none", {"planted": sorted(planted), "differences": mapped})
        CTX.verdict(f"self-test rename: {kind} pair reported without the map", plain_kind in unmapped, plain_kind, unmapped)
    kept_site, dropped_sites = drop_sites[:1], set(drop_sites[1:])
    remaining = [dataclasses.astuple(item) for item in found if (item.path, item.key) in dropped_sites]
    CTX.verdict("self-test rename: dropped column planted and no difference at its sites under the map",
                bool(dropped_sites) and not remaining, "planted, none",
                {"planted": sorted(dropped_sites), "differences": remaining})
    per_site = {f"{path} {key}": [item.kind for item in plain if (item.path, item.key) == (path, key)]
                for path, key in sorted(dropped_sites)}
    CTX.verdict("self-test rename: dropped column one columns difference at each site without the map",
                bool(per_site) and all(kinds == ["columns"] for kinds in per_site.values()), "columns at each site", per_site)
    kept = [item.kind for item in found if (item.path, item.key) in kept_site]
    CTX.verdict("self-test rename: a frame still holding the dropped column is one columns difference under the map",
                kept == ["columns"], ["columns"], {"site": kept_site, "differences": kept})
    cell = [item for item in found if (item.path, item.key) == (prefix + UNMAPPED_CELL[0], UNMAPPED_CELL[1])]
    triple = repr((old_value, old_value, new_value))
    CTX.verdict("self-test rename: an unmapped column's cell equal to a mapped value is left and reported, its triple named",
                [item.kind for item in cell] == ["values"] and triple in cell[0].detail, f"one values difference naming {triple}",
                [dataclasses.astuple(item) for item in cell])
    written_path, mapped_path = UNMAPPED_YAML_WRITTEN, ".".join(UNMAPPED_YAML[1])
    document = [item for item in found if item.path == prefix + UNMAPPED_YAML[0]]
    CTX.verdict("self-test rename: an unmapped YAML key renamed on the right is reported in both spellings",
                [item.kind for item in document] == ["yaml"] and written_path in document[0].detail
                and mapped_path in document[0].detail, f"one yaml difference naming {written_path} and {mapped_path}",
                [dataclasses.astuple(item) for item in document])
    flipped = [dataclasses.astuple(item) for item in found if item.path == prefix + PLANT_PNG]
    CTX.verdict("self-test rename: a flipped png byte is skipped under the map", not flipped, "no difference", flipped)
    removed = [item.kind for item in found if item.path == prefix + REMOVED_PNG]
    CTX.verdict("self-test rename: a png removed from the right copy is one file-set difference", removed == ["file-set"],
                ["file-set"], removed)
    CTX.verdict("self-test rename: nothing else reported under the map", len(found) == 4, 4,
                [dataclasses.astuple(item) for item in found])


def plant_renamed_key(params: Path, old: str, new: str) -> list[str]:
    """Store the key under its new name in every store that holds it, format kept; the stores changed."""
    changed = []
    for path in sorted(params.rglob(f"*{H5_SUFFIX}")):
        with pd.HDFStore(path, mode="a") as store:
            if old not in store.keys():
                continue
            fmt = store.get_storer(old).format_type
            stored = store.get(old)
            store.remove(old)
            store.put(new, stored, format=fmt)
        changed.append(path.relative_to(params).as_posix())
    return changed


def plant_renamed_column(params: Path, old: str, new: str) -> list[tuple[str, str]]:
    """Rename the column label in every frame that holds it; the (store, key) sites changed."""
    sites = _frames_holding(params, old)
    for rel, key in sites:
        _rewrite(params / rel, key, lambda frame: frame.rename(columns={old: new}))
    return sites


def plant_renamed_value(params: Path, column: str, old: str, new: str) -> list[tuple[str, str]]:
    """Rename the value in every cell of the column, in every frame that holds the column; the (store, key) sites."""
    sites = _frames_holding(params, column)
    for rel, key in sites:
        _rewrite(params / rel, key, lambda frame: frame.assign(**{column: frame[column].replace({old: new})}))
    return sites


def _frames_holding(params: Path, column: str) -> list[tuple[str, str]]:
    """(store, key) of every stored frame with the column label."""
    sites = []
    for path in sorted(params.rglob(f"*{H5_SUFFIX}")):
        with pd.HDFStore(path, mode="r") as store:
            sites += [(path.relative_to(params).as_posix(), key) for key in sorted(store.keys())
                      if isinstance(stored := store.get(key), pd.DataFrame) and column in stored.columns]
    return sites


def plant_renamed_yaml_key(params: Path, old: str, new: str) -> list[str]:
    """Rename the key at every depth of every .yaml file that holds it (parsed and dumped with yaml); the files changed."""
    changed = []
    for path in sorted(params.rglob(f"*{YAML_SUFFIX}")):
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        renamed = _yaml_key_renamed(document, old, new)
        if renamed != document:
            path.write_text(yaml.safe_dump(renamed, sort_keys=False), encoding="utf-8")
            changed.append(path.relative_to(params).as_posix())
    return changed


def _yaml_key_renamed(document, old: str, new: str):
    """A parsed document with one key renamed in every mapping at every depth."""
    if isinstance(document, list):
        return [_yaml_key_renamed(item, old, new) for item in document]
    if not isinstance(document, dict):
        return document
    return {new if key == old else key: _yaml_key_renamed(value, old, new) for key, value in document.items()}


def plant_dropped_column(params: Path, column: str) -> list[tuple[str, str]]:
    """Drop the column from every frame that holds it but the first in sorted sequence; the (store, key) sites, the
    kept one first."""
    sites = _frames_holding(params, column)
    for rel, key in sites[1:]:
        _rewrite(params / rel, key, lambda frame: frame.drop(columns=column))
    return sites


def plant_unmapped_cell(params: Path, old: str, new: str) -> None:
    """Rename the first cell equal to the mapped value in a column the map does not name."""
    path, key, column = UNMAPPED_CELL

    def renamed_first(frame: pd.DataFrame) -> pd.DataFrame:
        values = frame[column].copy()
        values.iat[int(np.flatnonzero(values.to_numpy() == old)[0])] = new
        return frame.assign(**{column: values})

    _rewrite(params / path, key, renamed_first)


def plant_unmapped_yaml_key(params: Path) -> None:
    """Rename, on the right only, one key the map does not name, below the renamed key."""
    rel, (*parents, leaf), new = UNMAPPED_YAML
    path = params / rel
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    node = document
    for part in parents:
        node = node[part]
    node[new] = node.pop(leaf)
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")


def plant_removed_png(params: Path) -> None:
    """Remove one png file."""
    (params / REMOVED_PNG).unlink()


def _params_dir(tree: Path) -> Path:
    """The strategy-parameters folder of a copied output tree."""
    return tree / harness.REFERENCE_PARAMS.relative_to(harness.DATA_OUTPUT_DIR)


def plant_float(params: Path) -> None:
    """Change one float value of a fixed-format table."""
    path, key, column = PLANT_FLOAT
    _rewrite(params / path, key, lambda frame: frame.assign(**{column: _bumped(frame[column])}))


def plant_dtype(params: Path) -> None:
    """Change one column's dtype (float32 -> float64), values unchanged."""
    path, key, column = PLANT_DTYPE
    _rewrite(params / path, key, lambda frame: frame.assign(**{column: frame[column].astype("float64")}))


def plant_drop(params: Path) -> None:
    """Drop one key."""
    path, key = PLANT_DROP
    with pd.HDFStore(params / path, mode="a") as store:
        store.remove(key)


def plant_png(params: Path) -> None:
    """Flip one byte in the middle of a png."""
    path = params / PLANT_PNG
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0xFF
    path.write_bytes(bytes(data))


def plant_removed(params: Path) -> None:
    """Remove one file that is not a store."""
    (params / PLANT_REMOVED).unlink()


def plant_format(params: Path) -> None:
    """Store one key in the other storer format, content unchanged."""
    path, key = PLANT_FORMAT
    _rewrite(params / path, key, lambda frame: frame, other_format=True)


def plant_type(params: Path) -> None:
    """Store one column of a frame in the frame's place, as a Series."""
    path, key, column = PLANT_TYPE
    _rewrite(params / path, key, lambda frame: frame[column])


def plant_index_dtype(params: Path) -> None:
    """Change the index dtype (int64 -> float64) with equal values, which Index.equals ignores."""
    path, key = PLANT_INDEX_DTYPE
    _rewrite(params / path, key, lambda frame: frame.set_axis(frame.index.astype("float64")))


def plant_one_sided(params: Path) -> None:
    """Drop one column, so it exists on one side only."""
    path, key, column = PLANT_ONE_SIDED
    _rewrite(params / path, key, lambda frame: frame.drop(columns=column))


def plant_reversed(params: Path) -> None:
    """Reverse the column sequence, labels and dtypes unchanged."""
    path, key = PLANT_REVERSED
    _rewrite(params / path, key, lambda frame: frame[frame.columns[::-1]])


def plant_names(params: Path) -> None:
    """Name the index, which Index.equals and DataFrame.equals ignore."""
    path, key, name = PLANT_NAMES
    _rewrite(params / path, key, lambda frame: frame.rename_axis(name))


def _bumped(column: pd.Series) -> pd.Series:
    """The column with its middle value increased by one, dtype kept."""
    values = column.to_numpy(copy=True)
    values[len(values) // 2] = values[len(values) // 2] + values.dtype.type(1)
    return pd.Series(values, index=column.index, name=column.name)


def _rewrite(path: Path, key: str, change, *, other_format: bool = False) -> None:
    """Replace one key with a changed object, in its storer format or, with other_format, in the other one."""
    with pd.HDFStore(path, mode="a") as store:
        fmt = store.get_storer(key).format_type
        stored = change(store.get(key))
        store.put(key, stored, format=OTHER_FORMAT[fmt] if other_format else fmt)


# =============================================================================
# TREE AND FILE COMPARISON
# =============================================================================


def compare_trees(left: Path, right: Path, *, h5_only: bool, mapping: Optional[dict] = None) -> tuple[list[Difference], Tally]:
    """Every difference between two trees (relative paths compared); under a name map the left tree loses the map's
    dropped columns and is renamed in memory first, .yaml files compare parsed and .png bytes are skipped."""
    files_left, files_right = _files(left), _files(right)
    differences, tally = [], Tally()
    for rel in sorted(files_left | files_right):
        if h5_only and not rel.endswith(H5_SUFFIX):
            continue
        if Path(rel).name in EXCEPTED_FILES:
            CTX.log.info("EXCEPTED %s", rel)
            continue
        if rel not in files_right or rel not in files_left:
            differences.append(Difference("file-set", rel, "", "only in " + ("left" if rel in files_left else "right")))
            continue
        tally.files += 1
        if rel.endswith(H5_SUFFIX):
            tally.h5_files += 1
            differences += compare_h5(left / rel, right / rel, rel, tally, mapping)
            if harness.file_md5(left / rel) != harness.file_md5(right / rel):
                tally.h5_bytes_differ += 1
        elif mapping is not None and rel.endswith(PNG_SUFFIX):
            CTX.log.info("SKIPPED %s", rel)
            tally.png_skipped += 1
        elif mapping is not None and rel.endswith(YAML_SUFFIX):
            differences += compare_yaml(left / rel, right / rel, rel, mapping)
        elif (left / rel).read_bytes() != (right / rel).read_bytes():
            differences.append(Difference("bytes", rel, "", "byte content differs"))
    return differences, tally


def _files(root: Path) -> set[str]:
    """Relative POSIX paths of every file under a tree."""
    return {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}


def compare_h5(left: Path, right: Path, rel: str, tally: Tally, mapping: Optional[dict]) -> list[Difference]:
    """Key sets, then per common key format, type, index, columns, dtypes and values; under a name map the left
    store's keys are renamed first, each left object loses the map's dropped columns before its rename, and each left
    key is named as written and as mapped."""
    differences = []
    with pd.HDFStore(left, mode="r") as store_left, pd.HDFStore(right, mode="r") as store_right:
        written = (harness.renamed_keys(sorted(store_left.keys()), mapping) if mapping is not None
                   else {key: key for key in store_left.keys()})
        keys_left, keys_right = set(written), set(store_right.keys())
        if keys_left != keys_right:
            differences.append(Difference("h5-keys", rel, "", f"only left {[_spellings(written[key], key) for key in sorted(keys_left - keys_right)]}, "
                                                              f"only right {sorted(keys_right - keys_left)}"))
        for key in sorted(keys_left & keys_right):
            tally.keys += 1
            format_left = store_left.get_storer(written[key]).format_type
            format_right = store_right.get_storer(key).format_type
            label = _spellings(written[key], key)
            if format_left != format_right:
                differences.append(Difference("format", rel, label, f"{format_left} vs {format_right}"))
            stored = store_left.get(written[key])
            kept = harness.kept_frame(stored, mapping) if mapping is not None else stored
            renamed = harness.renamed_frame(kept, mapping) if mapping is not None else kept
            differences += [Difference(kind, rel, label, detail) for kind, detail
                            in compare_frames(renamed, store_right.get(key), kept if mapping is not None else None)]
    return differences


def compare_yaml(left: Path, right: Path, rel: str, mapping: dict) -> list[Difference]:
    """Both files parsed, the left renamed by the map: one difference naming every differing key path, the left's as
    written and as mapped."""
    written = yaml.safe_load(left.read_text(encoding="utf-8"))
    document = yaml.safe_load(right.read_text(encoding="utf-8"))
    leaves_left = _yaml_leaves(written, harness.renamed_document(written, mapping))
    leaves_right = _yaml_leaves(document, document)
    differing = []
    for path in sorted(leaves_left.keys() | leaves_right.keys()):
        if path not in leaves_left:
            differing.append(f"{path}: only right")
            continue
        spelled, value = _spellings(leaves_left[path][0], path), leaves_left[path][1]
        if path not in leaves_right:
            differing.append(f"{spelled}: only left")
        elif value != leaves_right[path][1] or type(value) is not type(leaves_right[path][1]):
            differing.append(f"{spelled}: {value!r} vs {leaves_right[path][1]!r}")
    return [Difference("yaml", rel, "", "; ".join(differing))] if differing else []


def _yaml_leaves(written, mapped, written_path: str = "", mapped_path: str = "") -> dict[str, tuple[str, object]]:
    """{mapped key path: (key path as written, mapped value)} of a parsed document and its renamed copy, walked in step
    (a rename keeps each mapping's key sequence); every value but a non-empty mapping is a leaf."""
    if not (isinstance(written, dict) and written):
        return {mapped_path: (written_path, mapped)}
    leaves = {}
    for (key, value), (new, renamed) in zip(written.items(), mapped.items()):
        leaves.update(_yaml_leaves(value, renamed, _joined(written_path, key), _joined(mapped_path, new)))
    return leaves


def _joined(path: str, key: object) -> str:
    """A dotted key path extended by one key."""
    return f"{path}.{key}" if path else str(key)


def _spellings(written: object, mapped: object) -> str:
    """A name as written and, when the map renamed it, as mapped."""
    return str(mapped) if written == mapped else f"{written} (mapped {mapped})"


# =============================================================================
# FRAME COMPARISON
# =============================================================================


def compare_frames(left, right, written=None) -> list[tuple[str, str]]:
    """(kind, detail) of each difference between two stored objects; values only when index, columns and dtypes agree,
    the axis names that equals ignores in any case. written is the left object as stored, less the map's dropped
    columns, when a name map renamed it into left: every label and string cell a difference names then shows both
    spellings."""
    if type(left) is not type(right):
        return [("type", f"{type(left).__name__} vs {type(right).__name__}")]
    found = []
    left_index, right_index = _index_dtypes(left.index), _index_dtypes(right.index)
    if not left.index.equals(right.index) or left_index != right_index:
        found.append(("index", f"{len(left.index)} vs {len(right.index)} rows, dtypes {left_index} vs {right_index}"))
    labels = _written_labels(left, written)
    if isinstance(left, pd.DataFrame) and not left.columns.equals(right.columns):
        shown = list(left.columns) if written is None else [_spellings(labels[label], label) for label in left.columns]
        found.append(("columns", f"{shown} vs {list(right.columns)}"))
    left_dtypes, right_dtypes = _dtype_texts(left), _dtype_texts(right)
    changed = {_spellings(labels.get(label, label), label): f"{left_dtypes[label]} vs {right_dtypes[label]}"
               for label in sorted(left_dtypes.keys() & right_dtypes.keys(), key=str) if left_dtypes[label] != right_dtypes[label]}
    if changed:
        found.append(("dtypes", str(changed)))
    if not found and not left.equals(right):
        found.append(("values", _value_differences(left, right, written)))
    left_names, right_names = _axis_names(left), _axis_names(right)
    if left_names != right_names:
        shown = left_names if written is None else _spellings(_axis_names(written), left_names)
        found.append(("names", f"{shown} vs {right_names}"))
    return found


def _written_labels(left, written) -> dict:
    """Column label -> the label as written before a name map renamed it; each label maps to itself without a map (a
    Series has the one empty label of _dtype_texts, its name compared with the axis names)."""
    if isinstance(left, pd.Series):
        return {"": ""}
    return dict(zip(left.columns, left.columns if written is None else written.columns))


def _index_dtypes(index: pd.Index) -> str:
    """The index dtype, or the dtype of each level of a MultiIndex, as text."""
    return str([str(dtype) for dtype in index.dtypes]) if isinstance(index, pd.MultiIndex) else str(index.dtype)


def _dtype_texts(stored) -> dict:
    """Column label -> the dtypes of the columns with that label, as text (a label can repeat); a Series is one column
    under the empty label, its name compared with the axis names."""
    if isinstance(stored, pd.Series):
        return {"": str(stored.dtype)}
    texts = {}
    for label, dtype in zip(stored.columns, stored.dtypes):
        texts[label] = f"{texts[label]}, {dtype}" if label in texts else str(dtype)
    return texts


def _axis_names(stored) -> dict:
    """The names equals ignores: of the index levels, of the column levels, and a Series' own name."""
    names = {"index": list(stored.index.names)}
    if isinstance(stored, pd.DataFrame):
        names["columns"] = list(stored.columns.names)
    else:
        names["name"] = stored.name
    return names


def _value_differences(left, right, written=None) -> str:
    """Per column, taken by its number so a repeated label is compared too: maximum absolute and relative difference
    for numbers, differing cells otherwise; under a name map each column is named in both spellings and other cells
    by their first distinct (left as written, mapped left, right) triples."""
    source = left if written is None else written
    pairs = ([(_spellings(source.columns[number], label), left.iloc[:, number], right.iloc[:, number], source.iloc[:, number])
              for number, label in enumerate(left.columns)]
             if isinstance(left, pd.DataFrame) else [(_spellings(source.name, left.name), left, right, source)])
    parts = []
    for name, column_left, column_right, column_written in pairs:
        if column_left.equals(column_right):
            continue
        if pd.api.types.is_numeric_dtype(column_left) and pd.api.types.is_numeric_dtype(column_right):
            a = column_left.to_numpy(dtype="float64")
            b = column_right.to_numpy(dtype="float64")
            nan_mismatch = int(np.sum(np.isnan(a) != np.isnan(b)))
            both = ~np.isnan(a) & ~np.isnan(b)
            absolute = np.abs(a[both] - b[both])
            relative = absolute / np.maximum(np.abs(a[both]), np.finfo("float64").tiny)
            parts.append(f"{name}: {int(np.sum(absolute > 0))} cells differ, max abs {absolute.max(initial=0.0):.6g}, "
                         f"max rel {relative.max(initial=0.0):.6g}, NaN mismatches {nan_mismatch}")
        else:
            differs = ((column_left != column_right) & ~(column_left.isna() & column_right.isna())).to_numpy()
            text = f"{name}: {int(differs.sum())} cells differ"
            if written is not None:
                triples = _first_triples(zip(column_written.to_numpy()[differs], column_left.to_numpy()[differs],
                                             column_right.to_numpy()[differs]))
                text += f"; first (left, mapped left, right) {triples}"
            parts.append(text)
    return "; ".join(parts) or "objects differ (equals False) with equal columns"


def _first_triples(triples) -> list[tuple]:
    """The first TRIPLE_LIMIT distinct triples, in the sequence of their first occurrence (NaN equal to NaN)."""
    found, seen = [], set()
    for triple in triples:
        if repr(triple) not in seen:
            seen.add(repr(triple))
            found.append(triple)
            if len(found) == TRIPLE_LIMIT:
                break
    return found


if __name__ == "__main__":
    main()
