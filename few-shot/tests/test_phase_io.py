from pathlib import Path

import numpy as np

from arclib.phase_io import load_phase1_results, save_phase1_results


def test_phase1_results_roundtrip(tmp_path):
    output_path = Path(tmp_path) / "phase1.json"
    sample_data = {
        "task_001": [
            {
                "config": {"training": {"num_train_epochs": 1}},
                "prompt": "example prompt",
                "response": "{}",
                "token_ids": [1, 2, 3],
                "array_payload": np.array([[1, 0], [0, 1]]),
            }
        ]
    }
    metadata = {"code_mode": True, "include_spatial_grid_views": False}

    save_phase1_results(str(output_path), sample_data, metadata=metadata)

    assert output_path.exists()
    loaded = load_phase1_results(str(output_path))

    assert loaded["task_configs"] == {
        "task_001": [
            {
                "config": {"training": {"num_train_epochs": 1}},
                "prompt": "example prompt",
                "response": "{}",
                "token_ids": [1, 2, 3],
                "array_payload": [[1, 0], [0, 1]],
            }
        ]
    }
    assert loaded["metadata"] == metadata
    assert "saved_at" in loaded
