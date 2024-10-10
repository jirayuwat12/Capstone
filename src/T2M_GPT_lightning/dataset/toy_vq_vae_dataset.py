import torch
from torch.utils.data import Dataset


class ToyDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        joint_size: int = 150,
        frame_size: int | None = None,
        window_size: int | None = None,
    ) -> None:
        """
        Load the toy dataset from the given path.

        Args:
        - joint_size (int): The number of joints in the skeleton.
        - frame_size (int): The number of frames in the skeleton. If None, the frames will not be truncated or zero-padded.
        - window_size (int): The number of frames in the window. If None, the window will not be applied.
        - dataset_size (int): The number of samples in the dataset.
        """
        # Set attributes
        self.data_path = data_path
        self.joint_size = joint_size
        self.frame_size = frame_size
        self.window_size = window_size

        self.min_value = float("inf")
        self.max_value = -float("inf")

        with open(data_path, "r") as f:
            self.data = f.readlines()
            self.data = [line.strip().split(" ") for line in self.data]
            self.data = [[float(val) for val in line] for line in self.data]
            self.data = [torch.tensor(line).reshape(-1, 151)[:, :-1] for line in self.data]
            # Calculate the min and max values
            for line in self.data:
                self.min_value = min(self.min_value, line.min())
                self.max_value = max(self.max_value, line.max())

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        start_index = torch.randint(0, self.data[idx].shape[0] - self.window_size, (1,)).item()
        end_index = start_index + self.window_size
        data = self.data[idx][start_index:end_index]
        data = (data - self.min_value) / (self.max_value - self.min_value)
        return data


if __name__ == "__main__":
    dataset = ToyDataset(data_path="./data/toy_data/train.skels", joint_size=150, frame_size=64, window_size=16)
    print(dataset[0].shape)
