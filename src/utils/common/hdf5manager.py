"""
Centralized HDF5 operations manager for consistent and efficient file handling.

This module provides a unified interface for all HDF5 operations across the codebase,
reducing code duplication and improving performance through proper resource management.

Structure:
1. HDF5Manager class: context management, the write operation (put_dataframe_normalized), iteration, read
   operations (get_dataframe variants), key operations, normalization, helpers
2. Global instance
3. Module-level stream helpers (stream_trigger_series)
"""

import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Set, Tuple

import pandas as pd

from .constants import (
    HDF5_REFERENCE_PREFIX,
    HDF5_SERIES_PREFIX,
    HDF5_TRIGGERS_PREFIX,
    TEMP_STORE_SUFFIX,
)

logger = logging.getLogger(__name__)


class HDF5Manager:
    """Centralized manager for HDF5 operations with optimized settings."""

    DEFAULT_COMPRESSION = "blosc:zstd"
    DEFAULT_COMPRESSION_LEVEL = 5

    # =========================================================================
    # CONTEXT MANAGEMENT
    # =========================================================================

    @contextmanager
    def open_store(self, file_path: Path, mode: str = "r"):
        """Context manager for HDF5 store operations with automatic cleanup.

        Mode "w" creates the file's folder when it is missing, so a task's run() needs no prepared
        folder; it writes <stem>.tmp.h5 beside the file and moves it onto the file only after the body
        completed, so an interrupted write never leaves a partial file at the path; on an exception the
        temporary file is removed. Only a failure to create the store is logged, then raised; an exception
        of the body propagates unlogged, with its own context.
        """
        target = Path(file_path)
        written = target.with_name(f"{target.stem}{TEMP_STORE_SUFFIX}") if mode == "w" else target
        store = None
        completed = False
        try:
            try:
                if mode == "w":
                    target.parent.mkdir(parents=True, exist_ok=True)
                store = pd.HDFStore(str(written), mode=mode)
            except Exception as e:
                logger.error(f"Error opening HDF5 store {file_path}: {e}")
                raise
            yield store
            completed = True
        finally:
            if store is not None:
                store.close()
            if written != target:
                if completed:
                    os.replace(written, target)
                else:
                    written.unlink(missing_ok=True)

    # =========================================================================
    # WRITE OPERATION
    # =========================================================================

    def put_dataframe_normalized(
        self,
        store: pd.HDFStore,
        key: str,
        df: pd.DataFrame,
        format: str,
        schema: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Normalize the DataFrame for HDF5 and store it at the normalized key in the caller's format; only a
        "table" is compressed, with DEFAULT_COMPRESSION at DEFAULT_COMPRESSION_LEVEL. A normalization error
        propagates unlogged; a failed store write is logged, naming the key, then raised."""
        df_norm = self.normalize_for_hdf5(df, schema=schema)
        normalized_key = self._normalize_key(key)
        compression = (
            {"complib": self.DEFAULT_COMPRESSION, "complevel": self.DEFAULT_COMPRESSION_LEVEL}
            if format == "table"
            else {}
        )
        try:
            store.put(normalized_key, df_norm, format=format, **compression)
        except Exception as e:
            logger.error(f"Error storing DataFrame at key '{normalized_key}': {e}")
            raise

    # =========================================================================
    # ITERATION
    # =========================================================================

    def iter_series(self, store: pd.HDFStore) -> Iterator[Tuple[str, pd.DataFrame]]:
        """Iterate non-reference series from an HDF5 store.

        Only iterates keys starting with '/series_'. Skips reference keys,
        trigger keys, and any other non-series keys.

        Yields:
            Tuple of (series_id, DataFrame) for each valid series
        """
        for series_id, key in self.iter_series_keys(store):
            df = self.get_dataframe_fast(store, key)
            if len(df) == 0:
                continue
            yield series_id, df

    def iter_series_keys(self, store: pd.HDFStore) -> Iterator[Tuple[str, str]]:
        """Iterate (series_id, key) over the '/series_' keys, reference keys excluded, without reading a frame.

        These are the keys iter_series reads; it also skips a frame without rows.
        """
        prefix = "/" + HDF5_SERIES_PREFIX
        ref_prefix = "/" + HDF5_REFERENCE_PREFIX
        keys_set = self.get_keys_set(store)
        for key in sorted(keys_set):
            if not key.startswith(prefix):
                continue
            if key.startswith(ref_prefix):
                continue
            yield key[len(prefix):], key

    # =========================================================================
    # READ OPERATIONS
    # =========================================================================

    def get_dataframe(self, store: pd.HDFStore, key: str) -> Optional[pd.DataFrame]:
        """Retrieve DataFrame from HDF5 store with key existence check (None for an absent key; a read error raises)."""
        normalized_key = self._normalize_key(key)
        if normalized_key in store.keys():
            df = store.get(normalized_key)
            return df
        else:
            logger.warning(f"Key '{normalized_key}' not found in HDF5 store")
            return None

    def get_dataframe_fast(
        self, store: pd.HDFStore, key: str
    ) -> pd.DataFrame:
        """Retrieve DataFrame without key existence check (for validated keys; a read error raises)."""
        normalized_key = self._normalize_key(key)
        df = store.get(normalized_key)
        return df

    # =========================================================================
    # KEY OPERATIONS
    # =========================================================================

    def get_keys_set(self, store: pd.HDFStore) -> Set[str]:
        """Get all keys as a set for O(1) lookups (a read error raises)."""
        keys_set = set(store.keys())
        return keys_set

    # =========================================================================
    # NORMALIZATION
    # =========================================================================

    def normalize_for_hdf5(
        self, df: pd.DataFrame, schema: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """Normalize a DataFrame to be HDF5-friendly and compact."""
        if df is None or len(df) == 0:
            return df

        # Fast path: skip copy when no schema and no categoricals
        has_categoricals = any(
            getattr(df[col].dtype, "name", "") == "category" for col in df.columns
        )
        if not schema and not has_categoricals:
            return df

        out = df.copy()
        if has_categoricals:
            out = self._convert_categoricals(out)
        return self._apply_schema(out, schema or {})

    def _convert_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert categorical columns to string."""
        for col in list(df.columns):
            if getattr(df[col].dtype, "name", "") == "category":
                df[col] = df[col].astype(str)
        return df

    def _apply_schema(self, df: pd.DataFrame, schema: Dict[str, Any]) -> pd.DataFrame:
        """Apply schema transformations to DataFrame."""
        # Drop columns
        for col in schema.get("drop", []):
            if col in df.columns:
                df = df.drop(columns=[col])

        # Float32 conversion
        for col in schema.get("float32", []):
            if col in df.columns:
                df[col] = df[col].astype("float32")

        # Bool conversion
        for col in schema.get("bool", []):
            if col in df.columns:
                df[col] = df[col].astype("bool")

        # Int32 conversion (with fillna=-1)
        for col in schema.get("int32", []):
            if col in df.columns:
                if df[col].isna().any():
                    df[col] = df[col].fillna(-1)
                df[col] = df[col].astype("int32")

        # String conversion
        for col in schema.get("str", []):
            if col in df.columns:
                if df[col].isna().any():
                    df[col] = df[col].fillna("")
                df[col] = df[col].astype(str)

        return df

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _normalize_key(self, key: str) -> str:
        """Normalize key to start with '/'."""
        return key if key.startswith("/") else f"/{key}"


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

hdf5_manager = HDF5Manager()


# =============================================================================
# MODULE-LEVEL STREAM HELPERS
# =============================================================================


def stream_trigger_series(
    store: pd.HDFStore,
) -> Iterator[Tuple[str, pd.DataFrame, Optional[pd.DataFrame]]]:
    """Iterate (series_id, series_df, triggers_df) from a trigger HDF5 store.

    Pairs each /series_{id} key with its corresponding /triggers_{id} key.
    Yields None for triggers_df when no triggers key exists for a given series.

    Args:
        store: HDF5 store in read mode

    Yields:
        Tuple of (series_id, series_df, triggers_df) for each series
    """
    triggers_prefix = "/" + HDF5_TRIGGERS_PREFIX
    triggers_keys_set = {
        k for k in hdf5_manager.get_keys_set(store) if k.startswith(triggers_prefix)
    }
    for series_id, series_df in hdf5_manager.iter_series(store):
        triggers_key = f"/{HDF5_TRIGGERS_PREFIX}{series_id}"
        triggers_df = None
        if triggers_key in triggers_keys_set:
            triggers_df = hdf5_manager.get_dataframe_fast(store, triggers_key)
        yield series_id, series_df, triggers_df
