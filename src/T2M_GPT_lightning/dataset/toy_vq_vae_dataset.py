from typing import Literal

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from capstone_utils.skeleton_utils.skeleton import SUPPORT_DATA_SPECES, Skeleton


class ToyDataset(Dataset):
    def __init__(
        self,
        data_path: str | None = None,
        data_tensor_path: str | None = None,
        joint_size: int = 150,
        normalise: bool = True,
        is_data_has_timestamp: bool = False,
        frame_size: int | None = None,
        window_size: int = -1,
        data_spec: Literal["all", "face", "body", "hand", "core", "rel_hand"] = "all",
    ) -> None:
        """
        Load the toy dataset from the given path.

        Args:
        - joint_size (int): The number of joints in the skeleton.
        - frame_size (int): The number of frames in the skeleton. If None, the frames will not be truncated or zero-padded.
        - window_size (int): The number of frames in the window. If -1, the window size will not be used.
        - dataset_size (int): The number of samples in the dataset.
        - normalise (bool): Whether to normalise the data or not.
        - is_data_has_timestamp (bool): Whether the data has a timestamp or not.
        - data_path (str): The path to the data file.
        - data_tensor_path (str): The path to the data tensor file.
        - data_spec (str): The type of data to load. Can be "all", "face", "body", "hand", "core", or "rel_hand".
        """
        # Set attributes
        self.joint_size = joint_size
        self.frame_size = frame_size
        self.window_size = window_size
        self.normalise = normalise
        self.is_data_has_timestamp = is_data_has_timestamp
        if data_spec not in SUPPORT_DATA_SPECES:
            raise ValueError(f"data_spec must be one of {SUPPORT_DATA_SPECES}, but got {data_spec}")
        self.data_spec = data_spec

        self.min_value = float("inf")
        self.max_value = -float("inf")

        if data_path is None and data_tensor_path is None:
            raise ValueError("Either data_path or data_tensor_path must be provided.")
        if data_tensor_path is not None:
            # Load the data from the tensor file
            self.data = torch.load(data_tensor_path)
            # Calculate the min and max values
            self.min_value = min([data.min() for data in self.data])
            self.max_value = max([data.max() for data in self.data])
        else:
            self.data = []
            with open(data_path, "r") as f:
                # for line in f:
                for line in tqdm(f, desc="Loading data", unit="line"):
                    line = line.strip().split(" ")
                    line = [float(val) for val in line]
                    line = torch.tensor(line).reshape(-1, self.joint_size + self.is_data_has_timestamp)[
                        :, : -1 if self.is_data_has_timestamp else len(line)
                    ]
                    self.data.append(line)
                    # Calculate the min and max values
                    self.min_value = min(self.min_value, line.min())
                    self.max_value = max(self.max_value, line.max())

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        data = self.data[idx]
        if self.window_size != -1:
            start_index = torch.randint(0, data.shape[0] - self.window_size, (1,)).item()
            end_index = start_index + self.window_size
            data = data[start_index:end_index]
        if self.normalise:
            data = (data - self.min_value) / (self.max_value - self.min_value)

        data = Skeleton(data, joint_size=self.joint_size)
        return data.get_by_data_spec(self.data_spec)

    def get_full_sequences_by_idx(self, idx: int, unnorm: bool = False) -> torch.Tensor:
        """
        This method is used to get all the sequences from the given index.
        Note: this is ignore the window size

        Args:
        - idx (int): The index of the data

        Returns:
        - data (torch.Tensor): The data tensor
        """
        data = self.data[idx]
        if self.normalise and not unnorm:
            data = (data - self.min_value) / (self.max_value - self.min_value)
        return data
