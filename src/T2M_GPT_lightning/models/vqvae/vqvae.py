import torch
import torch.nn as nn
from lightning.pytorch import LightningModule

from T2M_GPT_lightning.models.vqvae.decoder import Decoder
from T2M_GPT_lightning.models.vqvae.encoder import Encoder
from T2M_GPT_lightning.models.vqvae.quantizer import Quantizer


class VQVAEModel(LightningModule):
    def __init__(
        self,
        learning_rate: int = 1e-5,
        L: int = 1,
        codebook_size: int = 32,
        embedding_dim: int = 256,
        skels_dim: int = 150,
        is_train: bool = True,
        quantizer_decay: float = 0.99,
        is_focus_hand_mode: bool = False,
        ratio_for_hand: float = 0.5,
    ) -> None:
        """
        Initialize the VQVAE model

        Args:
            learning_rate (int): Learning rate for the optimizer
        """
        super(VQVAEModel, self).__init__()
        self.encoder = Encoder(L=L, in_dim=skels_dim, emb_dim=embedding_dim)
        self.decoder = Decoder(L=L, emb_dim=embedding_dim, out_dim=skels_dim)
        self.quantizer = Quantizer(codebook_size=codebook_size, decay=quantizer_decay, codebook_dim=embedding_dim)
        if not is_train:
            self.eval()

        self.l = 2**L

        # Training var
        self.is_focus_hand_mode = is_focus_hand_mode
        self.ratio_for_hand = ratio_for_hand

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
        x_hat, vae_loss, _ = self.reconstruct(x)

        loss_reconstruction = self.compute_reconstruction_loss(x, x_hat, is_focus_hand_mode=self.is_focus_hand_mode)
        loss = loss_reconstruction + vae_loss

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_rec_loss", loss_reconstruction, prog_bar=True)
        self.log("train_vae_loss", vae_loss, prog_bar=True)

        return loss
    
    def compute_reconstruction_loss(self, x: torch.Tensor, x_hat: torch.Tensor, start_of_hand_index: int = 8*3, end_of_hand_index: int = 50*3, is_focus_hand_mode: bool = True) -> torch.Tensor:
        """
        Compute the reconstruction loss

        Args:
            x (torch.Tensor): Input tensor shape (B, T, X)
                B: Batch size
                T: Sequence length
                X: Feature dimension
            x_hat (torch.Tensor): Reconstructed tensor shape (B, T, X)
            is_focus_hand_mode (bool): If focus hand mode

        Returns:
            loss (torch.Tensor): Loss
        """
        if not is_focus_hand_mode:
            loss_reconstruction = nn.SmoothL1Loss()(x_hat, x)
            loss_velocities = nn.SmoothL1Loss()(x_hat[:, 1:] - x_hat[:, :-1], x[:, 1:] - x[:, :-1])
        else:
            x_hand = x[:, :, start_of_hand_index:end_of_hand_index]
            x_hat_hand = x_hat[:, :, start_of_hand_index:end_of_hand_index]
            x_non_hand = torch.cat([x[:, :, :start_of_hand_index], x[:, :, end_of_hand_index:]], dim=-1)
            x_hat_non_hand = torch.cat([x_hat[:, :, :start_of_hand_index], x_hat[:, :, end_of_hand_index:]], dim=-1)

            loss_reconstruction_hand = nn.SmoothL1Loss()(x_hat_hand, x_hand)
            loss_velocities_hand = nn.SmoothL1Loss()(x_hat_hand[:, 1:] - x_hat_hand[:, :-1], x_hand[:, 1:] - x_hand[:, :-1])
            loss_reconstruction_non_hand = nn.SmoothL1Loss()(x_hat_non_hand, x_non_hand)
            loss_velocities_non_hand = nn.SmoothL1Loss()(x_hat_non_hand[:, 1:] - x_hat_non_hand[:, :-1], x_non_hand[:, 1:] - x_non_hand[:, :-1])
    
            loss_reconstruction = loss_reconstruction_hand * self.ratio_for_hand + loss_reconstruction_non_hand * (1 - self.ratio_for_hand)
            loss_velocities = loss_velocities_hand * self.ratio_for_hand + loss_velocities_non_hand * (1 - self.ratio_for_hand)

        return loss_reconstruction + loss_velocities


    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x = batch
        x_hat, vae_loss, _ = self.reconstruct(x)
        loss_reconstruction = self.compute_reconstruction_loss(x, x_hat, is_focus_hand_mode=self.is_focus_hand_mode)
        loss = loss_reconstruction + vae_loss

        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        if self.learning_rate:
            return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        else:
            return torch.optim.Adam(self.parameters())

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Decode the indices to the original tensor

        Args:
            indices (torch.Tensor): Codebook indices shape (B, T/l, 1)
                l: Sequence length which is sampled (=L^2)

        Returns:
            x_hat (torch.Tensor): Reconstructed tensor shape (B, T, X)
        """
        dequantized = self.quantizer.dequantize(indices)
        x_hat = self.decoder.decode(dequantized)

        return x_hat


if __name__ == "__main__":
    model = VQVAEModel()

    x = torch.randn(2, 64, 150)
    x_hat, commitment_loss, perplexity = model.reconstruct(x)
    print("x_hat", x_hat)
    print("commitment_loss", commitment_loss)
    print("perplexity", perplexity)
    print(x_hat.shape, commitment_loss, perplexity)
