"""EXP-23: structural checks of the audit report (acceptance checks A1, A3, A8, A9).

Checks: headings A-H in order; the ten fields of every finding, each with a value, and
one finding per ID; findings sorted by severity; every tracked src path plus pipeline.py,
tests/*.py, workflows/*.yaml, scripts/*.py and every file the audit added listed in B with
a depth, the tracked files absent from the commit --base names marked as added (without
--base none is, after one NOTICE line); the AMBIGUOUS Q-row count equal to the count of E
questions labelled Q<n>; every finding below Confirmed holding the evidence that would
settle it and titled with the report title of the --rules folder's rule data
(harness.RULE_DATA_FILE; without it the title check is skipped after one NOTICE line). The
practical-impact ranking within one severity is not machine-readable (a reviewer check).
--self-test requires a valid skeleton to pass and eight mutated skeletons (seven without
the report title) to be rejected.

Structure:
1. main
2. Self-test
3. Report checks
4. Expected paths
"""
import sys

sys.dont_write_bytecode = True

import os

sys.path.insert(0, os.environ["PYTHIA_AUDIT_HARNESS"])
import harness

CTX = harness.preamble(__file__, imports_src=False)

import argparse
import re
from pathlib import Path
from typing import Optional

SECTIONS = {
    "A": "Executive summary", "B": "Scope and methodology", "C": "Findings", "D": "Missing tests",
    "E": "Inconsistencies and design questions", "F": "Verification results",
    "G": "Recommended action plan", "H": "Worktree and changes",
}
HEADING = re.compile(r"^## ([A-H])\. ")
FINDING = re.compile(r"^### (F-\d{2})\b(.*)$")
FIELD = re.compile(r"^- \*\*([^*]+):\*\* ?(.*)$")
FIELDS = ("ID", "Severity", "Confidence", "Category", "Location", "Intended behaviour",
          "Observed behaviour", "Evidence", "Impact", "Recommended follow-up")
SEVERITIES = ("Critical", "High", "Medium", "Low", "Informational")
CONFIRMED = "Confirmed"
SETTLING = "Evidence that would settle it"
NO_FINDINGS = "No findings"
PLANTED_ITEM = "not a defect: planted item"
PATH_ROW = re.compile(r"^\| `([^`]+)` \| (TRACED|READ|SKIMMED)\b(.*)$")
ADDED_MARK = "added by the audit"
Q_ROW = re.compile(r"^\| Q(\d+) \| AMBIGUOUS \|")
E_QUESTION = re.compile(r"^### Q(\d+)\b")


def main() -> None:
    """Check the report, or run the self-test; the title check only with the rule data of --rules."""
    args = _parse_args()
    data = CTX.rule_data(args.rules)
    title = None if data is None else data["report_title"]
    expected, added = expected_paths(args.base)
    if args.self_test:
        self_test(expected, added, title)
    else:
        problems = check_report(args.report.read_text(encoding="utf-8"), expected, added, title)
        CTX.log.info("checked %s (MD5 %s) against %d expected paths", args.report, harness.file_md5(args.report), len(expected))
        for check, found in problems.items():
            CTX.verdict(check, not found, "no problem", found[:20])
    CTX.summary()


def _parse_args() -> argparse.Namespace:
    """A report path, or --self-test; optional --rules, the folder of the rule data that gives the report title, and
    --base, the commit against which the audit's added files are found."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("report", nargs="?", type=Path)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--rules", type=Path)
    parser.add_argument("--base")
    args = parser.parse_args()
    if not args.self_test and args.report is None:
        parser.error("give a report path or --self-test")
    return args


# =============================================================================
# SELF-TEST
# =============================================================================


def self_test(expected: list[str], added: set[str], title: Optional[str]) -> None:
    """A valid skeleton passes; each planted defect is rejected by its own check (the untitled finding only with a
    title)."""
    valid = skeleton(expected, added, title)
    problems = check_report(valid, expected, added, title)
    CTX.verdict("self-test: valid skeleton passes", not any(problems.values()), "no problem",
                {check: found for check, found in problems.items() if found})
    mutations = {
        "missing heading": ("A1 headings", valid.replace("## D. Missing tests\n", "")),
        "nine-field finding": ("A1 fields", valid.replace("- **Impact:** planted impact one\n", "", 1)),
        "field without a value": ("A1 fields", valid.replace("- **Impact:** planted impact one\n", "- **Impact:**\n", 1)),
        "repeated finding ID": ("A1 fields", valid.replace("### F-02 - ", "### F-01 - ", 1).replace("- **ID:** F-02", "- **ID:** F-01", 1)),
        "unsorted severities": ("A1 severity sequence", _swap_findings(valid)),
        "missing path": ("A3 coverage", valid.replace(f"| `{expected[0]}` |", "| `removed` |", 1)),
        "Q-count off by one": ("A8 questions", valid.replace("### Q2 - planted question two\n", "", 1)),
    }
    if title is not None:
        mutations["below-Confirmed without the report title"] = ("A9 unconfirmed", valid.replace(
            f"### F-02 - {title} - {PLANTED_ITEM}", f"### F-02 - {PLANTED_ITEM}"))
    for name, (check, text) in mutations.items():
        found = check_report(text, expected, added, title)
        CTX.verdict(f"self-test: {name} rejected by {check}", bool(found[check]), "rejected", found[check][:3])


def skeleton(expected: list[str], added: set[str], title: Optional[str]) -> str:
    """A minimal report that satisfies every check; its below-Confirmed finding carries the title when one is given."""
    rows = [f"| `{path}` | TRACED | {ADDED_MARK if path in added else 'planted'} |" for path in expected]
    finding_one = _finding("F-01", "planted defect", "High", CONFIRMED, "one", "planted follow-up")
    finding_two = _finding("F-02", PLANTED_ITEM if title is None else f"{title} - {PLANTED_ITEM}", "Medium", "High", "two",
                           f"{SETTLING}: a planted experiment")
    parts = [
        "# Report", "", "## A. Executive summary", "planted", "", "## B. Scope and methodology",
        "| Path | Depth | Notes |", "|---|---|---|", *rows, "", "| Q | Tag | Two readings |", "|---|---|---|",
        "| Q1 | AMBIGUOUS | one |", "| Q2 | AMBIGUOUS | two |", "", "## C. Findings", "", finding_one, finding_two,
        "## D. Missing tests", "", "## E. Inconsistencies and design questions", "",
        "### Q1 - planted question one", "", "### Q2 - planted question two", "", "### E-1 - planted item", "",
        "## F. Verification results", "", "## G. Recommended action plan", "", "## H. Worktree and changes", "",
    ]
    return "\n".join(parts) + "\n"


def _finding(identifier: str, title: str, severity: str, confidence: str, word: str, follow_up: str) -> str:
    """One finding block with its ten fields."""
    values = {"ID": identifier, "Severity": severity, "Confidence": confidence, "Category": "robustness",
              "Location": "`src/x.py:1`", "Intended behaviour": f"planted intent {word}",
              "Observed behaviour": f"planted observation {word}", "Evidence": f"planted evidence {word}",
              "Impact": f"planted impact {word}", "Recommended follow-up": follow_up}
    return "\n".join([f"### {identifier} - {title}", *[f"- **{label}:** {values[label]}" for label in FIELDS], ""]) + "\n"


def _swap_findings(text: str) -> str:
    """Put the Medium finding before the High one."""
    first_start = text.index("### F-01")
    second_start = text.index("### F-02")
    end = text.index("## D. Missing tests")
    return text[:first_start] + text[second_start:end] + text[first_start:second_start] + text[end:]


# =============================================================================
# REPORT CHECKS
# =============================================================================


def check_report(text: str, expected: list[str], added: set[str], title: Optional[str]) -> dict[str, list[str]]:
    """Problems found by each check group (A9's title condition only with a title)."""
    lines = text.splitlines()
    sections = _sections(lines)
    findings = _findings(sections.get("C", []))
    names = [name for name, _ in findings]
    return {
        "A1 headings": _heading_problems(lines),
        "A1 fields": [f"{name}: missing or empty {missing}" for name, fields in findings
                      if (missing := [label for label in FIELDS if not fields.get(label, "").strip()])]
                     + [f"{name}: ID field {fields.get('ID')!r}" for name, fields in findings if fields.get("ID") != name]
                     + [f"{name}: {names.count(name)} findings with this ID" for name in sorted(set(names)) if names.count(name) > 1],
        "A1 severity sequence": _severity_problems(findings),
        "A3 coverage": _coverage_problems(sections.get("B", []), expected, added),
        "A8 questions": _question_problems(sections.get("B", []), sections.get("E", [])),
        "A9 unconfirmed": _unconfirmed_problems(findings, title),
        "non-vacuity": _vacuity_problems(sections, findings, expected),
    }


def _sections(lines: list[str]) -> dict[str, list[str]]:
    """Lines of each lettered section."""
    found, current = {}, None
    for line in lines:
        match = HEADING.match(line)
        if match:
            current = match.group(1)
            found[current] = []
        elif current is not None:
            found[current].append(line)
    return found


def _heading_problems(lines: list[str]) -> list[str]:
    """Headings A-H, each once, in order, with their names."""
    headings = [line for line in lines if HEADING.match(line)]
    letters = [HEADING.match(line).group(1) for line in headings]
    problems = [] if letters == list(SECTIONS) else [f"headings {letters}, expected {list(SECTIONS)}"]
    problems += [f"heading {line!r} lacks its name" for line in headings
                 if SECTIONS[HEADING.match(line).group(1)] not in line]
    return problems


def _findings(lines: list[str]) -> list[tuple[str, dict]]:
    """(ID with its heading text, fields) of every finding in section C."""
    findings = []
    for line in lines:
        match = FINDING.match(line)
        if match:
            findings.append((match.group(1), {"__title__": match.group(2)}))
            continue
        field = FIELD.match(line)
        if field and findings and field.group(1) in FIELDS:
            findings[-1][1][field.group(1)] = field.group(2)
    return findings


def _severity_problems(findings: list[tuple[str, dict]]) -> list[str]:
    """Known severities, from most to least severe."""
    ranks, problems = [], []
    for name, fields in findings:
        severity = fields.get("Severity", "").split(" ")[0]
        if severity not in SEVERITIES:
            problems.append(f"{name}: unknown severity {severity!r}")
            continue
        ranks.append((SEVERITIES.index(severity), name))
    problems += [f"{later} ({SEVERITIES[rank_later]}) after {earlier} ({SEVERITIES[rank_earlier]})"
                 for (rank_earlier, earlier), (rank_later, later) in zip(ranks, ranks[1:]) if rank_later < rank_earlier]
    return problems


def _coverage_problems(section_b: list[str], expected: list[str], added: set[str]) -> list[str]:
    """Every expected path on its own B row with a depth; added files marked as such."""
    rows = {}
    for line in section_b:
        match = PATH_ROW.match(line)
        if match:
            rows[match.group(1)] = match.group(3)
    problems = [f"missing {path}" for path in expected if path not in rows]
    problems += [f"{path} not marked '{ADDED_MARK}'" for path in sorted(added) if path in rows and ADDED_MARK not in rows[path]]
    return problems


def _question_problems(section_b: list[str], section_e: list[str]) -> list[str]:
    """count(AMBIGUOUS Q-rows) == count(E questions Q<n>), with the same numbers."""
    rows = sorted(int(match.group(1)) for line in section_b if (match := Q_ROW.match(line)))
    questions = sorted(int(match.group(1)) for line in section_e if (match := E_QUESTION.match(line)))
    return [] if rows == questions else [f"Q-rows {rows} vs E questions {questions}"]


def _unconfirmed_problems(findings: list[tuple[str, dict]], title: Optional[str]) -> list[str]:
    """Below Confirmed: the report title in the heading (when one is given) and the settling evidence in the
    follow-up."""
    problems = []
    for name, fields in findings:
        if fields.get("Confidence", "").split(" ")[0] == CONFIRMED:
            continue
        if title is not None and title not in fields["__title__"]:
            problems.append(f"{name}: below Confirmed but not titled '{title}'")
        if SETTLING not in fields.get("Recommended follow-up", ""):
            problems.append(f"{name}: follow-up lacks '{SETTLING}'")
    return problems


def _vacuity_problems(sections: dict, findings: list, expected: list[str]) -> list[str]:
    """Findings parsed (or none stated), B rows at least the expected paths, Q-rows present."""
    problems = []
    if not findings and not any(NO_FINDINGS in line for line in sections.get("C", [])):
        problems.append("no finding parsed and section C does not state none")
    rows = sum(bool(PATH_ROW.match(line)) for line in sections.get("B", []))
    if rows < len(expected):
        problems.append(f"{rows} B path rows for {len(expected)} expected paths")
    if not any(Q_ROW.match(line) for line in sections.get("B", [])):
        problems.append("no AMBIGUOUS Q-row in B")
    return problems


# =============================================================================
# EXPECTED PATHS
# =============================================================================


def expected_paths(base: Optional[str]) -> tuple[list[str], set[str]]:
    """Tracked src files, pipeline.py, tests/*.py, workflows/*.yaml, scripts/*.py and audit/*.py, and those the audit
    added: the ones absent from the commit base; without a base none, after one NOTICE line."""
    _, out = harness.run_helper(["git", "ls-files", "-z"], cwd=harness.REAL_WORKTREE, log=CTX.log)
    tracked = [path for path in out.split("\0") if path]
    patterns = (re.compile(r"^src/"), re.compile(r"^pipeline\.py$"), re.compile(r"^tests/[^/]+\.py$"),
                re.compile(r"^workflows/[^/]+\.yaml$"), re.compile(r"^scripts/[^/]+\.py$"), re.compile(r"^audit/[^/]+\.py$"))
    expected = sorted(path for path in tracked if any(pattern.match(path) for pattern in patterns))
    added: set[str] = set()
    if base is None:
        CTX.log.info("%sno --base commit: the check of the files added by the audit is skipped", harness.NOTICE_PREFIX)
    else:
        _, listed = harness.run_helper(["git", "ls-tree", "-r", "-z", "--name-only", base], cwd=harness.REAL_WORKTREE, log=CTX.log)
        added = set(expected) - {path for path in listed.split("\0") if path}
    CTX.log.info("expected paths: %d (%d added by the audit)", len(expected), len(added))
    return expected, added


if __name__ == "__main__":
    main()
