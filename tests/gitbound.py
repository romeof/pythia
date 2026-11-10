"""Regression for the bounded git calls and their loud failure (G-11, F-15).

Cases (a), (b), (d), (e) and (f) share one child program, which begins with
testkit.ANSI_CHILD_BOOTSTRAP (harmless under UTF-8, required by the ANSI child): before
pipeline is imported it patches subprocess.run for calls whose first argument is a list
starting with git, and for (f) tempfile.TemporaryFile for calls with mode "w+"; every other
call delegates to the original, untouched and unrecorded. Its argv names the case. Each case
runs --task trigger on its own one-series realistic copy; the git failure lies outside Luigi,
so each child is judged by its own verdicts.
(a) git raises OSError: an ERROR record names git rev-parse HEAD and the planted message,
    return != 0, no provenance file, the trigger 0.h5 present (before G-11: the WARNING "Could not
    get git information", return code 0, the YAML copies present).
(b) run records its timeout and stdout keywords and raises TimeoutExpired: timeout 60.0 and a
    file-like stdout recorded, and the ERROR record names the command and the expiry (before:
    timeout None, stdout PIPE).
(c) a positive control, nothing patched: _get_git_output(["git", "rev-parse", "HEAD"]) equals
    the 40-hex id of the test's own bounded temporary-file call, and a healthy --task trigger
    leaves a git_info.txt that reads as UTF-8 and whose Commit Hash line holds that id (both
    trees).
(d) git remote get-url origin replaced by a program that returns 2 (git's status for a missing
    remote), run by the real subprocess.run: an ERROR record names the command, return != 0, no
    provenance file, the trigger 0.h5 present (before: the WARNING, return code 0, the YAML copies and a
    partial git_info.txt).
(e) an ANSI child (-X utf8=0) with every git command replaced by a program writing
    b"a\\r\\nb\\rc\\xc3\\xa9\\r\\n": the case first requires that the child's code page decodes
    b"c\\xc3\\xa9" to something other than the UTF-8 text (else it cannot bite and fails as
    "could not run"), then return code 0 and git_info.txt's bytes holding exactly
    b"Commit Hash: a\\r\\nb\\r\\nc\\xc3\\xa9\\r\\nBranch:" (before: the code page's decode written
    as UTF-8).
(f) tempfile.TemporaryFile raises a planted FileNotFoundError: return != 0, the planted message,
    no record holding "git command", no provenance file, the trigger 0.h5 present (before: no
    temporary file is used, return code 0, every provenance file present).

Structure:
1. main
2. Cases
3. Helpers
"""
import sys

sys.dont_write_bytecode = True

import testkit

KIT = testkit.preamble(__file__)

import re
import subprocess
import tempfile
from pathlib import Path

from src.utils.common.outputs import _get_git_output

GIT_FILE = "git_info.txt"
PROVENANCE_FILES = (testkit.STRATEGY_FILE, testkit.DEV_FILE, GIT_FILE)
GIT_RECORD = "git command"
HEAD_COMMAND = ["git", "rev-parse", "HEAD"]
REMOTE_COMMAND = "git remote get-url origin"
BOUND_S = 60.0
RECORDED_PREFIX = "RECORDED "
BITES_PREFIX = "BITES "
COMMIT_PREFIX = "Commit Hash: "
COMMIT_ID = re.compile(r"[0-9a-f]{40}")
TRANSLATED = b"Commit Hash: a\r\nb\r\nc\xc3\xa9\r\nBranch:"
EXPIRY = "timed out"
GIT_CHILD = testkit.ANSI_CHILD_BOOTSTRAP + r'''
import subprocess, tempfile
case, strategy = sys.argv[4], sys.argv[5]
original_run, original_temporary = subprocess.run, tempfile.TemporaryFile
sys.stdout.write('BITES ' + str(b'c\xc3\xa9'.decode(locale.getpreferredencoding(False), 'replace') != 'c\xe9') + '\n')
sys.stdout.flush()
def described(stdout):
    if stdout == subprocess.PIPE:
        return 'pipe'
    if stdout == subprocess.DEVNULL:
        return 'devnull'
    return 'file' if hasattr(stdout, 'fileno') else repr(stdout)
def run(*args, **kwargs):
    command = args[0] if args else kwargs.get('args')
    if not (isinstance(command, list) and command and command[0] == 'git'):
        return original_run(*args, **kwargs)
    if case == 'raise':
        raise OSError(testkit.PLANTED_ERROR)
    if case == 'expire':
        sys.stdout.write('RECORDED ' + repr(kwargs.get('timeout')) + ' ' + described(kwargs.get('stdout')) + '\n')
        sys.stdout.flush()
        raise subprocess.TimeoutExpired(command, kwargs.get('timeout'))
    if case == 'status' and command[1:] == ['remote', 'get-url', 'origin']:
        command = [sys.executable, '-c', 'raise SystemExit(2)']
    if case == 'bytes':
        command = [sys.executable, '-c', "import sys; sys.stdout.buffer.write(b'a\\r\\nb\\rc\\xc3\\xa9\\r\\n')"]
    return original_run(command, *args[1:], **kwargs)
def temporary(*args, **kwargs):
    if case == 'tempfile' and kwargs.get('mode') == 'w+':
        raise FileNotFoundError(testkit.PLANTED_ERROR)
    return original_temporary(*args, **kwargs)
subprocess.run, tempfile.TemporaryFile = run, temporary
import pipeline
sys.argv = ['pipeline.py', '--task', 'trigger', '--strategy', strategy, '--profile', testkit.DEV_PROFILE]
pipeline.main()
'''


def main() -> None:
    """The six cases, then the summary (return code 1 on any FAIL)."""
    raised_git()
    expired_git()
    positive_control()
    missing_remote()
    ansi_capture()
    unusable_temporary_folder()
    KIT.finish()


# =============================================================================
# CASES
# =============================================================================


def raised_git() -> None:
    """Case (a): a git call that raises fails the run after the trigger output, with no provenance file."""
    with testkit.scratch(KIT) as folder:
        config_file, child, lines = git_child(folder, "raise")
        records = error_records(lines, " ".join(HEAD_COMMAND))
        KIT.verdict(f"case (a): an ERROR record names {' '.join(HEAD_COMMAND)} and the planted message",
                    any(testkit.PLANTED_ERROR in line for line in records), 1, records[:2])
        failed_after_trigger(config_file, child, "case (a)")


def expired_git() -> None:
    """Case (b): the call passes the 60 s bound and a file as stdout; an expiry is logged naming the command."""
    with testkit.scratch(KIT) as folder:
        _, child, lines = git_child(folder, "expire")
        recorded = [line[len(RECORDED_PREFIX):].split() for line in lines if line.startswith(RECORDED_PREFIX)]
        KIT.verdict(f"case (b): the git call passes timeout {BOUND_S} and a file-like stdout",
                    bool(recorded) and recorded[0] == [repr(BOUND_S), "file"], [repr(BOUND_S), "file"], recorded[:1])
        records = [line for line in error_records(lines, " ".join(HEAD_COMMAND)) if EXPIRY in line]
        KIT.verdict(f"case (b): an ERROR record names {' '.join(HEAD_COMMAND)} and the expiry", len(records) == 1, 1,
                    records)


def positive_control() -> None:
    """Case (c): the capture works: the in-process id and a healthy run's git_info.txt equal git's own answer."""
    head = head_id()
    found = _get_git_output(list(HEAD_COMMAND))
    KIT.verdict("case (c): _get_git_output(['git', 'rev-parse', 'HEAD']) equals the test's own 40-hex id",
                bool(COMMIT_ID.fullmatch(head)) and found == head, head, found)
    with testkit.scratch(KIT) as folder:
        config_file = testkit.config_copy(KIT, folder, strategy=dict(testkit.ONE_SERIES))
        child = testkit.run_pipeline(KIT, folder, "healthy", config_file, ["trigger"])
        raw = git_info(config_file).read_bytes() if git_info(config_file).is_file() else b""
        utf8 = testkit.raised(lambda: raw.decode("utf-8")) is None
        commit = [line for line in raw.decode("utf-8", "replace").splitlines() if line.startswith(COMMIT_PREFIX)]
        KIT.verdict("case (c): a healthy --task trigger writes a UTF-8 git_info.txt whose Commit Hash line holds the id",
                    child.return_code == 0 and utf8 and commit == [f"{COMMIT_PREFIX}{head}"],
                    {"return_code": 0, "utf-8": True, "commit": [f"{COMMIT_PREFIX}{head}"]},
                    {"return_code": child.return_code, "utf-8": utf8, "commit": commit})


def missing_remote() -> None:
    """Case (d): a git call that returns 2 fails the run after the trigger output, with no provenance file."""
    with testkit.scratch(KIT) as folder:
        config_file, child, lines = git_child(folder, "status")
        records = error_records(lines, REMOTE_COMMAND)
        KIT.verdict(f"case (d): an ERROR record names {REMOTE_COMMAND}", len(records) == 1, 1, records)
        failed_after_trigger(config_file, child, "case (d)")


def ansi_capture() -> None:
    """Case (e): under the ANSI code page git's UTF-8 output is decoded as UTF-8 and its line ends are translated."""
    with testkit.scratch(KIT) as folder:
        config_file, child, lines = git_child(folder, "bytes", ansi=True)
        bites = [line[len(BITES_PREFIX):].strip() for line in lines if line.startswith(BITES_PREFIX)]
        KIT.verdict("case (e): the child's code page decodes the UTF-8 bytes otherwise (else the case could not run)",
                    bites == ["True"], ["True"], bites or "could not run")
        raw = git_info(config_file).read_bytes() if git_info(config_file).is_file() else b""
        KIT.verdict("case (e): return code 0 and git_info.txt holds git's UTF-8 text with every line end translated",
                    child.return_code == 0 and TRANSLATED in raw, {"return_code": 0, "bytes": TRANSLATED},
                    {"return_code": child.return_code, "bytes": raw[raw.find(COMMIT_PREFIX.encode("ascii")):][:40]})


def unusable_temporary_folder() -> None:
    """Case (f): a temporary file that cannot be created fails the run with its own error, outside the git handler."""
    with testkit.scratch(KIT) as folder:
        config_file, child, lines = git_child(folder, "tempfile")
        planted = [line for line in lines if testkit.PLANTED_ERROR in line]
        KIT.verdict("case (f): the planted FileNotFoundError leaves the run", bool(planted), "present", planted[-1:])
        records = [line for line in lines if GIT_RECORD in line]
        KIT.verdict(f"case (f): no record holds '{GIT_RECORD}'", not records, "none", records[:2])
        failed_after_trigger(config_file, child, "case (f)")


# =============================================================================
# HELPERS
# =============================================================================


def git_child(folder: Path, case: str, *, ansi: bool = False) -> tuple[Path, testkit.Child, list[str]]:
    """The shared child program for one case on a one-series realistic copy in the folder."""
    config_file = testkit.config_copy(KIT, folder, strategy=dict(testkit.ONE_SERIES))
    child = testkit.run_child(KIT, folder, case, ["-c", GIT_CHILD, *testkit.child_args(KIT), case,
                                                  testkit.strategy_argument(config_file)],
                              timeout_s=testkit.PIPELINE_TIMEOUT_S, ansi=ansi)
    return config_file, child, testkit.plain_lines(child.text)


def error_records(lines: list[str], command: str) -> list[str]:
    """The ERROR records that name the command."""
    return [line for line in lines if line.startswith(testkit.ERROR_PREFIX) and command in line]


def failed_after_trigger(config_file: Path, child: testkit.Child, case: str) -> None:
    """return != 0, no provenance file, the trigger output present."""
    folder = git_info(config_file).parent
    present = [name for name in PROVENANCE_FILES if (folder / name).is_file()]
    trigger = testkit.output_file(config_file, "trigger").is_file()
    KIT.verdict(f"{case}: return != 0, no provenance file, the trigger 0.h5 present",
                child.return_code not in (0, None) and not present and trigger,
                {"return_code": "!= 0", "provenance": [], "trigger 0.h5": True},
                {"return_code": child.return_code, "timed_out": child.timed_out, "provenance": present, "trigger 0.h5": trigger})


def git_info(config_file: Path) -> Path:
    """The copy's git_info.txt in its strategy folder."""
    return testkit.output_file(config_file, "trigger").parents[2] / GIT_FILE


def head_id() -> str:
    """git rev-parse HEAD of the repository the run records, through the test's own bounded temporary-file call."""
    with tempfile.TemporaryFile() as out:
        subprocess.run(HEAD_COMMAND, cwd=Path.cwd(), stdin=subprocess.DEVNULL, stdout=out, check=True,
                       timeout=testkit.HELPER_BOUND_S)
        out.seek(0)
        return out.read().decode("ascii").strip()


if __name__ == "__main__":
    main()
