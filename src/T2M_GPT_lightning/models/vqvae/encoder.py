import torch
import torch.nn as nn

from T2M_GPT_lightning.models.vqvae.resnet import Resnet1D


class Encoder(nn.Module):
    def __init__(self, L: int, in_dim: int = 150, emb_dim: int = 256) -> None:
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv1d(in_dim, emb_dim, 3, 1, 1)
        self.relu = nn.ReLU()
        self.res_block = nn.Sequential()
        for i in range(L):
            self.res_block.add_module(f"res_block_conv_{i}", nn.Conv1d(emb_dim, emb_dim, 4, 2, 1))
            self.res_block.add_module(f"res_block_resnet_{i}", Resnet1D(emb_dim=emb_dim))

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode the input tensor x

        Args:
            x (torch.Tensor): Input tensor shape (B, T, X)
                B: Batch size
                T: Sequence length
                X: Feature dimension
        """
        x_in = x.permute(0, 2, 1)
        x_first_conv = self.conv1(x_in)
        x_first_conv = self.relu(x_first_conv)
        x_all_in_resblock = self.res_block(x_first_conv)
        return x_all_in_resblock

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)


if __name__ == "__main__":
    import torchsummary

    model = Encoder()
    torchsummary.summary(model, (64, 150))
