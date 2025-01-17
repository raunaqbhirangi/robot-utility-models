from dataloaders.extended_action_dataset import ExtendedActionDataset

import numpy as np
import torch


class ExtendedActionTactileDataset(ExtendedActionDataset):
    def __init__(
        self,
        config,
        num_extra_actions=0,
        only_action_return=False,
        return_tactile=False,
        *args,
        **kwargs,
    ):
        super().__init__(config, num_extra_actions, only_action_return, *args, **kwargs)

    def __getitem__(self, index):
        self._total_calls += 1
        # If index is float64, cast to int.
        if isinstance(index, np.float64):
            index = index.astype(np.int64)

        trajectory_slice = self._subslices[index]
        is_padding, actions = self._get_action_slice(trajectory_slice)
        if self._only_action_return:
            return is_padding, actions  # dummy value for obs
        return_frames = (self._get_video_frames(trajectory_slice),)
        # frames = self._get_video_frames(trajectory_slice)
        if self._data_config.use_depth:  # TODO: here make cleaner?
            depths = self._get_depth_frames(trajectory_slice)
            return_frames += (depths,)
        if self._data_config.use_tactile:
            tactile = self._get_tactile_frames(trajectory_slice)
            return_frames += (tactile,)
        return *return_frames, is_padding, actions
