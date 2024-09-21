import torch
from torch.utils.data import Dataset


class ToyDataset(Dataset):
    def __init__(self, skels_data_path: str, joint_size: int = 150, frame_size: int | None = None, window_size: int | None = None) -> None:
        """
        Load the toy dataset from the given path.

        Args:
        - skels_data_path (str): The path to the skeleton data.
        - joint_size (int): The number of joints in the skeleton.
        - frame_size (int): The number of frames in the skeleton. If None, the frames will not be truncated or zero-padded.
        - window_size (int): The number of frames in the window. If None, the window will not be applied.
        """
        # Set the paths
        self.skels_data_path = skels_data_path

        # Set attributes
        self.joint_size = joint_size
        self.frame_size = frame_size
        self.window_size = window_size

        # Read the data
        with open(self.skels_data_path, "r") as f:
            self.skels_lines = f.readlines()

        # Create the skeleton frames for each skeleton line
        self.skels_frames = []
        for skels_line in self.skels_lines:
            skels_frame = torch.tensor(
                [
                    list(map(float, skels_line.strip().split(" ")))
                    for skels_line in skels_line.split("\t")
                ]
            )
            self.skels_frames.append(skels_frame)
        
        # Convert the list of skeleton frames to a tensor
        for skels_index in range(len(self.skels_frames)):
            # Convert the list of skeleton frames to a tensor and drop counter
            temp_tensor = self.skels_frames[skels_index].clone().reshape(-1)
            self.skels_frames[skels_index] = []
            for i in range(0, temp_tensor.shape[0], joint_size+1):
                self.skels_frames[skels_index].append(temp_tensor[i:i + joint_size])
            self.skels_frames[skels_index] = torch.stack(self.skels_frames[skels_index], dim=0)
            
            # Truncate or zero-pad the frames if needed
            if frame_size is None:
                continue
            elif self.skels_frames[skels_index].shape[0] < frame_size:
                self.skels_frames[skels_index] = torch.cat(
                    [
                        self.skels_frames[skels_index],
                        torch.zeros(frame_size - self.skels_frames[skels_index].shape[0], joint_size)
                    ]
                )
            else:
                self.skels_frames[skels_index] = self.skels_frames[skels_index][:frame_size]

    def __len__(self) -> int:
        return len(self.skels_frames)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        skels = self.skels_frames[idx].clone()
        if self.window_size is None:
            return skels, skels
        start_index = torch.randint(0, skels.shape[0] - self.window_size, (1,)).item()
        skels = skels[start_index:start_index + self.window_size]

        # Test with 1s vector
        # skels = torch.ones_like(skels)

        return skels, skels

if __name__ == "__main__":
    dataset = ToyDataset(skels_data_path="/Users/jirayuwat/Desktop/Capstone/data/toy_data/dev.skels", frame_size=196, window_size=64)
    skels = dataset[0][0]
    print(skels.shape)
    print(skels)
    
    # Test equality
    for i in range(len(dataset)):
        print(f'Index: {i}\t{dataset[i][0].shape}\t{dataset[i][1].shape}')
        assert torch.equal(dataset[i][0], dataset[i][1])
    
    # Test length
    assert len(dataset) == 5