import torch
import torch.nn as nn


def create_resnet1d_block(
    in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int, dilation: int
) -> nn.Sequential:
    """
    Create a ResNet1D block

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Kernel size
        stride (int): Stride
        padding (int): Padding
        dilation (int): Dilation

    Returns:
        nn.Sequential: ResNet1D block
    """
    return nn.Sequential(
        nn.ReLU(),
        nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation),
        nn.ReLU(),
        nn.Conv1d(out_channels, out_channels, 1, 1),
    )


class Resnet1D(nn.Module):
    def __init__(self, emb_dim: int = 256) -> None:
        super(Resnet1D, self).__init__()
        self.block1 = create_resnet1d_block(emb_dim, emb_dim, 3, 1, 9, 9)
        self.block2 = create_resnet1d_block(emb_dim, emb_dim, 3, 1, 3, 3)
        self.block3 = create_resnet1d_block(emb_dim, emb_dim, 3, 1, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x (torch.Tensor): Input tensor shape (B, T, X)
                B: Batch size
                T: Sequence length
                X: Feature dimension

        Returns:
            torch.Tensor: Output tensor shape (B, T, X)
        """
        first_block_in = x
        first_block_out = self.block1(first_block_in)

        second_block_in = first_block_out + first_block_in
        second_block_out = self.block2(second_block_in)

        third_block_in = second_block_out + second_block_in
        third_block_out = self.block3(third_block_in)

        return third_block_out
