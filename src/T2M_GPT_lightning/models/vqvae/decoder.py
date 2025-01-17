import torch
import torch.nn as nn
from T2M_GPT_lightning.models.vqvae.resnet import Resnet1D


class Decoder(nn.Module):
    def __init__(self, L: int, out_dim: int = 150, emb_dim: int = 256) -> None:
        """
        The decoder architecture is implemented respecting the T2M-GPT architecture

        Args:
            L (int): Number of layers
            out_dim (int): Output dimension
            emb_dim (int): Embedding
        """
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv1d(emb_dim, out_dim, 3, 1, 1)
        self.relu = nn.ReLU()
        self.res_block = nn.Sequential()
        for i in range(L):
            self.res_block.add_module(f"res_block_conv_{i}", nn.Conv1d(emb_dim, emb_dim, 3, 1, 1))
            self.res_block.add_module(f"res_block_resnet_{i}", Resnet1D(emb_dim=emb_dim))
            self.res_block.add_module(f"res_block_upsample_{i}", nn.Upsample(scale_factor=2, mode="nearest"))
            self.res_block.add_module(f"res_block_conv2_{i}", nn.Conv1d(emb_dim, emb_dim, 3, 1, 1))

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """
        Initialize the weights of the model

        Description:
            - the weights are initialized using kaiming_normal_
            - the bias are initialized to zeros
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode the input tensor x

        Args:
            x (torch.Tensor): Input tensor shape (B, embed_dim, T/l)
                B: Batch size
                embed_dim: Embedding dimension
                T/l: Sequence length which is sampled

        Returns:
            x_decoded (torch.Tensor): Decoded tensor shape (B, T, X)
        """
        # Reshape input tensor to (B, embed_dim, T/l)
        x_in = x.permute(0, 2, 1)
        # Pass through the resnet block
        x_all_in_resblock = self.res_block(x_in)
        x_first_conv = self.conv1(x_all_in_resblock)
        x_first_conv = self.relu(x_first_conv)
        # Reshape the output tensor to (B, T, X)
        x_out = x_first_conv.permute(0, 2, 1)

        return x_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model

        Args:
            x (torch.Tensor): Input tensor shape (B, embed_dim, T/l)
                B: Batch size
                embed_dim: Embedding dimension
                T/l: Sequence length which is sampled
        """
        return self.decode(x)


if __name__ == "__main__":
    model = Decoder()
    import torchsummary

    torchsummary.summary(model, (512, 16))
