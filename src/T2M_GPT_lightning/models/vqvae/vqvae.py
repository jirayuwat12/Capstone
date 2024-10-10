import torch
import torch.nn as nn
from lightning.pytorch import LightningModule

from T2M_GPT_lightning.models.vqvae.decoder import Decoder
from T2M_GPT_lightning.models.vqvae.encoder import Encoder
from T2M_GPT_lightning.models.vqvae.quantizer import Quantizer


class VQVAEModel(LightningModule):
    def __init__(self, learning_rate: int = 1e-5, L: int = 1) -> None:
        """
        Initialize the VQVAE model

        Args:
            learning_rate (int): Learning rate for the optimizer
        """
        super(VQVAEModel, self).__init__()
        self.encoder = Encoder(L=L)
        self.decoder = Decoder(L=L)
        self.quantizer = Quantizer()

        # Hyperparameters
        self.learning_rate = learning_rate

    def reconstruct(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Reconstruct the input tensor x

        Args:
            x (torch.Tensor): Input tensor shape (B, T, X)
                B: Batch size
                T: Sequence length
                X: Feature dimension

        Returns:
            x_hat (torch.Tensor): Reconstructed tensor shape (B, T, X)
            loss (torch.Tensor): Commitment loss + Embedding loss
            encoding_indices (torch.Tensor): Codebook indices shape (B, T/l, 1)
        """
        x_encoded = self.encoder.encode(x)
        x_quantized, loss, encoding_indices = self.quantizer.quantize(x_encoded)
        x_hat = self.decoder.decode(x_quantized)

        return x_hat, loss, encoding_indices

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.reconstruct(x)

    def compute_codebook_indices(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the codebook indices

        Args:
            x (torch.Tensor): Input tensor shape (B, T, X)
                B: Batch size
                T: Sequence length
                X: Feature dimension

        Returns:
            indices (torch.Tensor): Codebook indices shape (B, T/l, 1)
                l: Sequence length which is sampled (=L^2)
        """
        x_encoded = self.encoder.encode(x)
        _, _, indices = self.quantizer.quantize(x_encoded)
        return indices

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x = batch
        x_hat, vae_loss, indices = self.reconstruct(x)

        loss_reconstruction = nn.MSELoss()(x_hat, x)
        loss = loss_reconstruction + vae_loss

        self.log("train_loss", loss, prog_bar=True)
        if batch_idx == 0:
            print("x", x)
            print("x_hat", x_hat)
            print("vae_loss", vae_loss)
            print("reconstruction_loss", loss_reconstruction)
            print("indices", ",".join([str(i) for i in indices.flatten().tolist()]))
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x = batch
        x_hat, vae_loss, _ = self.reconstruct(x)
        loss_reconstruction = nn.MSELoss()(x_hat, x)
        loss = loss_reconstruction + vae_loss
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        if self.learning_rate:
            return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        else:
            return torch.optim.Adam(self.parameters())


if __name__ == "__main__":
    model = VQVAEModel()

    x = torch.randn(2, 64, 150)
    x_hat, commitment_loss, perplexity = model.reconstruct(x)
    print("x_hat", x_hat)
    print("commitment_loss", commitment_loss)
    print("perplexity", perplexity)
    print(x_hat.shape, commitment_loss, perplexity)
