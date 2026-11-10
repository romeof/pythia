"""
Common plotting utilities shared across performance visualization modules.

Provides consistent styling, legend handling, and figure utilities for
both perseries and grouping performance plots.

Structure:
1. Style dataclass
2. PlotBase class (constants & style, rendering helpers, output & figures)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

from pathlib import Path


@dataclass(frozen=True)
class Style:
    color: str
    linestyle: str
    label: str


class PlotBase:
    """Small collection of helpers shared by performance plots.

    Focused on:
      - consistent color/linestyle mapping for (plus/minus, true/false)
      - legend de-duplication
      - monospace text block placement
    """

    # =========================================================================
    # CONSTANTS & STYLE
    # =========================================================================

    LABEL_PLUS_TRUE = "Plus True"
    LABEL_PLUS_FALSE = "Plus False"
    LABEL_MINUS_TRUE = "Minus True"
    LABEL_MINUS_FALSE = "Minus False"

    # Class variable used by style_for() and add_legend_once()
    legend_sequence = (LABEL_PLUS_TRUE, LABEL_PLUS_FALSE, LABEL_MINUS_TRUE, LABEL_MINUS_FALSE)

    @staticmethod
    def style_for(is_plus: bool, is_true: bool) -> Style:
        if is_plus and is_true:
            return Style("blue", "-", PlotBase.LABEL_PLUS_TRUE)
        if is_plus and not is_true:
            return Style("blue", ":", PlotBase.LABEL_PLUS_FALSE)
        if (not is_plus) and is_true:
            return Style("red", "-", PlotBase.LABEL_MINUS_TRUE)
        return Style("red", ":", PlotBase.LABEL_MINUS_FALSE)

    # =========================================================================
    # RENDERING HELPERS
    # =========================================================================

    @staticmethod
    def add_legend_once(ax, handles_map: Dict[str, any]) -> None:
        # sort legend by canonical sequence
        handles = [handles_map[lbl] for lbl in PlotBase.legend_sequence if lbl in handles_map]
        labels = [lbl for lbl in PlotBase.legend_sequence if lbl in handles_map]
        if handles:
            ax.legend(handles, labels, loc="upper left")

    @staticmethod
    def place_monospace_text(fig, left_ax, text: str, y: float = 0.045) -> None:
        left = left_ax.get_position().x0
        fig.text(left, y, text, family="monospace", fontsize=8, va="bottom", ha="left")

    # =========================================================================
    # OUTPUT & FIGURES
    # =========================================================================

    @staticmethod
    def plots_dir_from_output(output_file: Path) -> Path:
        plots_dir = Path(output_file).parent / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        return plots_dir

    @staticmethod
    def should_emit_signal_plot(
        series_id: str,
        *,
        emit_signal_plots: bool,
        whitelist: Optional[Iterable[str]] = None,
        plotted_so_far: int = 0,
        max_signal_plots: int = 0,
    ) -> bool:
        if not emit_signal_plots:
            return False
        allowlist = set(map(str, whitelist or []))
        if allowlist and str(series_id) not in allowlist:
            return False
        if not allowlist and plotted_so_far >= int(max_signal_plots):
            return False
        return True
