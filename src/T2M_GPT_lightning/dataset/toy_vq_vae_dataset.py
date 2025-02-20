import torch
from torch.utils.data import Dataset


class ToyDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        joint_size: int = 150,
        normalise: bool = True,
        is_data_has_timestamp: bool = False,
        frame_size: int | None = None,
        window_size: int = -1,
    ) -> None:
        """
        Load the toy dataset from the given path.

        Args:
        - joint_size (int): The number of joints in the skeleton.
        - frame_size (int): The number of frames in the skeleton. If None, the frames will not be truncated or zero-padded.
        - window_size (int): The number of frames in the window. If -1, the window size will not be used.
        - dataset_size (int): The number of samples in the dataset.
        """
        # Set attributes
        self.data_path = data_path
        self.joint_size = joint_size
        self.frame_size = frame_size
        self.window_size = window_size

        self.min_value = float("inf")
        self.max_value = -float("inf")

        self.raw_data = []

        self.normalise = normalise

        self.is_data_has_timestamp = is_data_has_timestamp

        with open(data_path, "r") as f:
            self.data = f.readlines()
            self.data = [line.strip().split(" ") for line in self.data]
            self.raw_data.append(self.data.copy())
            self.data = [[float(val) for val in line] for line in self.data]
            self.data = [
                torch.tensor(line).reshape(-1, self.joint_size + self.is_data_has_timestamp)[
                    :, : -1 if self.is_data_has_timestamp else len(line)
                ]
                for line in self.data
            ]
            # Calculate the min and max values
            for line in self.data:
                self.min_value = min(self.min_value, line.min())
                self.max_value = max(self.max_value, line.max())

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        if self.window_size != -1:
            start_index = torch.randint(0, self.data[idx].shape[0] - self.window_size, (1,)).item()
            end_index = start_index + self.window_size
        else:
            start_index = 0
            end_index = self.data[idx].shape[0]
        data = self.data[idx][start_index:end_index]
        if self.normalise:
            data = (data - self.min_value) / (self.max_value - self.min_value)
        return data

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


if __name__ == "__main__":
    dataset = ToyDataset(data_path="./data/toy_data/train.skels", joint_size=150, frame_size=64, window_size=16)
    print(len(dataset.raw_data[0][0]) / (150 + 1))
    print(dataset.min_value, dataset.max_value)
