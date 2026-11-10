"""
Replot the gradient figure of one or more Performance outputs.

Each performance HDF5 file named on the command line is replotted to the path replot_gradient_from_hdf5 derives
beside it (<its folder>/plots/gradient/gradient_hd5.png), so two inputs never write one file. Every path is checked
before the first replot: missing ones raise one FileNotFoundError that names them, and nothing is written.

Usage (from the repo root):
    python scripts/gradientreplot.py <performance.h5> [<performance.h5> ...]
"""

import argparse
import logging
import sys
from pathlib import Path

# The repository root, so `import src...` works when run as a script.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logger = logging.getLogger(__name__)


def main() -> int:
    """Parse the paths, refuse missing ones, then replot each."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    paths = [Path(path) for path in _parse_arguments().paths]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Performance files not found: {', '.join(missing)}")

    from src.utils.performance.replot import replot_gradient_from_hdf5

    for path in paths:
        logger.info(f"Replotting from: {path}")
        replot_gradient_from_hdf5(path)
    return 0


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("paths", nargs="+", help="Performance HDF5 files to replot.")
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(main())
