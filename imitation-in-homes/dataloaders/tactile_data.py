import pickle
from pathlib import Path
from typing import Union
import json

import numpy as np
import torch

from dataloaders.utils import TrajectorySlice

TACTILE_PKL_FILENAME = "tactile_data.pkl"


class TactileDataLoader:
    def __init__(
        self,
        tactile_data_path: Union[str, Path],
        subtract_tactile_baseline: bool = True,
    ):
        self.tactile_data_path = Path(tactile_data_path)
        self._load_tactile_data()
        self._baseline = np.median(self._tactile_data[:3], axis=0, keepdims=True)
        self.base_data_path = self.tactile_data_path.parent

        with open(self.base_data_path / "sensor_stats.json") as json_data:
            self.sensor_stats = json.load(json_data)
            json_data.close()
        if subtract_tactile_baseline:
            self._tactile_data -= self._baseline

    def _load_tactile_data(self):
        tactile_pkl_path = self.tactile_data_path / TACTILE_PKL_FILENAME
        # Load the tactile data from the pickle file.

        with tactile_pkl_path.open("rb") as f:
            tactile_ts, tactile_data = pickle.load(f)

        self._tactile_data = np.array(tactile_data, dtype=np.float32)

        self._len = len(tactile_data)

    def __len__(self):
        return self._len

    def get_batch(self, indices: Union[np.ndarray, TrajectorySlice]):
        if isinstance(indices, TrajectorySlice):
            indices = np.arange(
                indices.start_index, indices.end_index, indices.skip + 1
            )

        indices = np.array(indices)
        n = len(indices)
        assert np.all(indices >= 0)

        normalized_tactile_data = (
            self._tactile_data[indices] - self.sensor_stats["shift"]
        ) / self.sensor_stats["scale"]
        
        return normalized_tactile_data.astype(np.float32)