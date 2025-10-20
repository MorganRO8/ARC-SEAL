"""Helpers for persisting intermediate training artifacts to disk."""
from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from typing import Any, Dict, Optional

import numpy as np


class PhaseJSONEncoder(json.JSONEncoder):
    """JSON encoder that gracefully handles NumPy scalars and arrays."""

    def default(self, obj: Any):  # type: ignore[override]
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def save_phase1_results(
    path: str,
    task_configs: Dict[str, Any],
    *,
    metadata: Optional[Dict[str, Any]] = None,
    ensure_fsync: bool = True,
) -> None:
    """Write Phase 1 outputs to ``path`` and optionally attach ``metadata``."""

    payload: Dict[str, Any] = {
        "task_configs": task_configs,
        "metadata": metadata or {},
        "saved_at": datetime.now(UTC).isoformat(),
    }

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, cls=PhaseJSONEncoder)
        handle.flush()
        if ensure_fsync:
            os.fsync(handle.fileno())


def load_phase1_results(path: str) -> Dict[str, Any]:
    """Load previously persisted Phase 1 outputs from ``path``."""

    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)
