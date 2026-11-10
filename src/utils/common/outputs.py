"""
Output paths, provenance files and task output folders of a run.

Structure:
1. Output path generation (get_output_file_name, the hierarchical output path, and _get_base_output_dir, its
   shared components; copy_config_to_output, remove_provenance_files and _provenance_paths, the strategy folder's
   provenance files; _get_strategy_output_dir; _get_git_output, git in the repository holding this source)
2. Task output management (clean_task_output_directory, remove_task_output_directory)
"""

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Tuple, Union

from .config import profile_config_file
from .constants import GIT_INFO_FILE, GIT_TIMEOUT_S, PROFILES, REPORT_ENCODING
from .schema import ReadConfig, TriggerConfig

logger = logging.getLogger(__name__)

# The repository holding this source (src/utils/common/outputs.py -> parents[3]); git provenance runs here (E-5).
SOURCE_ROOT = Path(__file__).resolve().parents[3]


# =============================================================================
# OUTPUT PATH GENERATION
# =============================================================================


def get_output_file_name(read: ReadConfig, trigger: TriggerConfig, task: str) -> Path:
    """Generate output file path from the read and trigger sections with hierarchical structure."""
    data_root, raw_path, dataset_name, strategy_type, strategy_parameters = (
        _get_base_output_dir(read, trigger)
    )
    replay_mode = trigger.replay.mode
    return (
        data_root
        / dataset_name
        / strategy_type
        / strategy_parameters
        / replay_mode
        / task
        / raw_path.name
    )


def _get_base_output_dir(
    read: ReadConfig, trigger: TriggerConfig,
) -> Tuple[Path, Path, str, str, str]:
    """Return shared path components: (data_root, raw_path, dataset_name, strategy_type, strategy_parameters).

    data_root is <read.output_root>/<date>, where <date> is the name of the folder two levels above raw_data;
    strategy_parameters names the trigger section's gate items and strategy parameters.
    """
    from .strategyname import get_strategy_parameters

    raw_path = Path(read.raw_data)
    output_root = Path(read.output_root)
    strategy_type = trigger.strategy.strategy_name
    strategy_parameters = get_strategy_parameters(trigger)
    dataset_name = raw_path.parents[1].name
    data_root = output_root / raw_path.parents[2].name
    return data_root, raw_path, dataset_name, strategy_type, strategy_parameters


def copy_config_to_output(config_file: str, profile: str, read: ReadConfig, trigger: TriggerConfig) -> None:
    """Copy config files and git commit info to the strategy/parameters folder.

    The git text is gathered first, so a failed git call leaves no provenance file. Then writes three of the
    _provenance_paths items: the strategy YAML, the run's profile YAML that load_config read and git_info.txt. A
    copy that does not complete removes every provenance file, so they describe only a completed run.
    """
    git_text = (
        "Git Information\n" + "=" * 50 + "\n\n"
        + f"Commit Hash: {_get_git_output(['git', 'rev-parse', 'HEAD'])}\n"
        + f"Branch: {_get_git_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])}\n"
        + f"Commit Date: {_get_git_output(['git', 'log', '-1', '--pretty=%ci'])}\n"
        + f"Repository URL: {_get_git_output(['git', 'remote', 'get-url', 'origin'])}\n\n"
        + "Commit Message:\n" + "-" * 30 + "\n"
        + _get_git_output(["git", "log", "-1", "--pretty=%B"])
    )
    paths = _provenance_paths(config_file, read, trigger)
    _get_strategy_output_dir(read, trigger).mkdir(parents=True, exist_ok=True)

    profile_file = profile_config_file(config_file, profile)
    completed = False
    try:
        shutil.copy2(config_file, paths[Path(config_file).name])
        logger.info(f"Analysis config copied to: {paths[Path(config_file).name]}")
        shutil.copy2(profile_file, paths[profile_file.name])
        logger.info(f"Profile config copied to: {paths[profile_file.name]}")
        paths[GIT_INFO_FILE].write_text(git_text, encoding=REPORT_ENCODING)
        logger.info(f"Git info copied to: {paths[GIT_INFO_FILE]}")
        completed = True
    finally:
        if not completed:
            remove_provenance_files(config_file, read, trigger)


def remove_provenance_files(config_file: str, read: ReadConfig, trigger: TriggerConfig) -> None:
    """Remove the strategy folder's provenance files; an absent file is skipped, any other failure raises."""
    for path in _provenance_paths(config_file, read, trigger).values():
        path.unlink(missing_ok=True)


def _provenance_paths(config_file: str, read: ReadConfig, trigger: TriggerConfig) -> Dict[str, Path]:
    """The provenance files in the strategy folder, keyed by file name: the strategy YAML copy (its own name),
    one copy per profile of PROFILES (a run writes its own; the removal takes every one, so no copy of an earlier
    run's profile stands beside this run's) and git_info.txt."""
    strategy_dir = _get_strategy_output_dir(read, trigger)
    names = (Path(config_file).name, *(profile_config_file(config_file, profile).name for profile in PROFILES),
             GIT_INFO_FILE)
    return {name: strategy_dir / name for name in names}


def _get_strategy_output_dir(read: ReadConfig, trigger: TriggerConfig) -> Path:
    """Get the strategy output directory path (mode-independent, above mode subfolder)."""
    data_root, _, dataset_name, strategy_type, strategy_parameters = (
        _get_base_output_dir(read, trigger)
    )
    return data_root / dataset_name / strategy_type / strategy_parameters


def _get_git_output(cmd: list) -> str:
    """Execute git command in SOURCE_ROOT and return stripped output, bounded by GIT_TIMEOUT_S.

    It runs in the tree holding this source, never the working directory, so git_info.txt names the code that
    ran even when the run starts in another tree (E-5).

    A failure (an expiry, a non-zero return code, a missing git, output that is not UTF-8) is logged at ERROR naming the
    command, then raised. The output goes to a text-mode temporary file, never a pipe: its read decodes strictly
    as UTF-8 in this thread and translates line ends, and the wait after an expiry's kill joins no pipe reader.
    The child writes through a duplicate of the file's handle, which shares the file offset, so the read starts
    after seek(0); the file goes with its last handle. A temporary folder that cannot hold the file raises its own
    error, outside the handler, before any git process exists.
    """
    with tempfile.TemporaryFile(mode="w+", encoding=REPORT_ENCODING) as out:
        try:
            subprocess.run(cmd, cwd=SOURCE_ROOT, stdin=subprocess.DEVNULL, stdout=out, check=True, timeout=GIT_TIMEOUT_S)
            out.seek(0)
            return out.read().strip()
        except (subprocess.SubprocessError, OSError, UnicodeDecodeError) as exc:
            logger.error("git command %s failed: %s", " ".join(cmd), exc)
            raise


# =============================================================================
# TASK OUTPUT MANAGEMENT
# =============================================================================


def clean_task_output_directory(target_file: str) -> Path:
    """Clean previous task outputs and prepare output directory.

    A folder that cannot be removed raises, so a stale target never stands in for the
    task's output.

    Args:
        target_file: Path to the target output file

    Returns:
        Path to the output file with parent directory created
    """
    remove_task_output_directory(target_file)

    output_file = Path(target_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    return output_file


def remove_task_output_directory(target_file: Union[str, Path]) -> None:
    """Remove the folder of a task's output file when it exists, logging the removal; a folder that cannot be
    removed raises."""
    target_dir = Path(target_file).parent
    if target_dir.exists():
        shutil.rmtree(target_dir)
        logger.info(f"✓ Cleaned previous outputs from {target_dir}")
