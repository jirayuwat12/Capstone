from typing import Any, Literal

import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
from torch.optim.lr_scheduler import LambdaLR

from capstone_utils.skeleton_utils.bmc_loss import calculate_bmc_loss
from T2M_GPT_lightning.models.vqvae.decoder import Decoder
from T2M_GPT_lightning.models.vqvae.encoder import Encoder
from T2M_GPT_lightning.models.vqvae.quantizer import Quantizer


class VQVAEModel(LightningModule):
    def __init__(
        self,
        learning_rate_scheduler: Literal["static", "lambda", "reduce_on_plateau"] = "static",
        learning_rate: Any = 1e-5,
        L: int = 1,
        codebook_size: int = 32,
        embedding_dim: int = 256,
        skels_dim: int = 150,
        is_train: bool = True,
        quantizer_decay: float = 0.99,
        is_focus_hand_mode: bool = False,
        ratio_for_hand: float = 0.5,
        betas: tuple[float, float] = (0.9, 0.99),
        minibatch_count_to_reset: int = 0,
        bmc_loss_multiplier: float = 0,
        except_r: bool = False,
        activation_type: Literal["linear", "relu", "tanh"] = "linear",
    ) -> None:
        """
        Initialize the VQVAE model

        Args:
            learning_rate_scheduler (str): Learning rate scheduler name
            learning_rate (Any): Depending on the learning rate scheduler, it can be a float or Any
            L (int): Number of levels in the VQVAE
            codebook_size (int): Size of the codebook
            embedding_dim (int): Dimension of the embedding
            skels_dim (int): Dimension of the input skeleton data
            is_train (bool): If the model is in training mode
            quantizer_decay (float): Decay rate for the quantizer
            is_focus_hand_mode (bool): If focus hand mode is enabled
            ratio_for_hand (float): Ratio for hand focus mode
            betas (tuple[float, float]): Betas for the Adam optimizer
            minibatch_count_to_reset (int): Number of minibatches before resetting the codebook vectors
            is_train (bool): If the model is in training mode
            bmc_loss_multiplier (float): Multiplier for BMC loss
            except_r (bool): If True, skip the reconstruction of the right hand
            activation_type (Literal["linear", "relu", "tanh"]): Activation function type
        """
        super(VQVAEModel, self).__init__()
        self.save_hyperparameters()
        self.encoder = Encoder(L=L, in_dim=skels_dim, emb_dim=embedding_dim)
        self.decoder = Decoder(L=L, emb_dim=embedding_dim, out_dim=skels_dim, activation_type=activation_type)
        self.quantizer = Quantizer(
            codebook_size=codebook_size,
            decay=quantizer_decay,
            codebook_dim=embedding_dim,
            minibatch_count_to_reset=minibatch_count_to_reset,
        )
        if not is_train:
            self.eval()

        self.l = 2**L

        # Training var
        self.is_focus_hand_mode = is_focus_hand_mode
        self.ratio_for_hand = ratio_for_hand

        # Hyperparameters
        self.learning_rate_scheduler = learning_rate_scheduler
        self.learning_rate = learning_rate
        self.betas = betas
        self.bmc_loss_multiplier = bmc_loss_multiplier
        self.except_r = except_r

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

        if self.except_r:
            x_hat[:, ::3] = x[:, ::3]

        loss_reconstruction = self.compute_reconstruction_loss(x, x_hat, is_focus_hand_mode=self.is_focus_hand_mode)

        loss_bmc_loss = 0
        if self.bmc_loss_multiplier > 0:
            # acc_bmc_loss = 0
            bmc_loss = calculate_bmc_loss(x, x_hat, x.shape[0])
            loss_bmc_loss += bmc_loss * self.bmc_loss_multiplier

        loss = loss_reconstruction + vae_loss + loss_bmc_loss

        self.log("train_loss", loss.detach(), prog_bar=True)
        self.log("train_rec_loss", loss_reconstruction.detach(), prog_bar=True)
        self.log("train_vae_loss", vae_loss.detach(), prog_bar=True)
        if self.bmc_loss_multiplier > 0:
            self.log("train_bmc_loss", loss_bmc_loss.detach(), prog_bar=True)

        # Log the learning rate
        self.log("learning_rate", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True)

        return loss

    def compute_reconstruction_loss(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        start_of_hand_index: int = 8 * 3,
        end_of_hand_index: int = 50 * 3,
        is_focus_hand_mode: bool = True,
    ) -> torch.Tensor:
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
        if x_hat.shape[1] < x.shape[1]:
            x = x[:, : x_hat.shape[1], :]
        if not is_focus_hand_mode:
            loss_reconstruction = nn.SmoothL1Loss()(x_hat, x)
            loss_velocities = nn.SmoothL1Loss()(x_hat[:, 1:] - x_hat[:, :-1], x[:, 1:] - x[:, :-1])
        else:
            x_hand = x[:, :, start_of_hand_index:end_of_hand_index]
            x_hat_hand = x_hat[:, :, start_of_hand_index:end_of_hand_index]
            x_non_hand = torch.cat([x[:, :, :start_of_hand_index], x[:, :, end_of_hand_index:]], dim=-1)
            x_hat_non_hand = torch.cat([x_hat[:, :, :start_of_hand_index], x_hat[:, :, end_of_hand_index:]], dim=-1)

            loss_reconstruction_hand = nn.SmoothL1Loss()(x_hat_hand, x_hand)
            loss_velocities_hand = nn.SmoothL1Loss()(
                x_hat_hand[:, 1:] - x_hat_hand[:, :-1], x_hand[:, 1:] - x_hand[:, :-1]
            )
            loss_reconstruction_non_hand = nn.SmoothL1Loss()(x_hat_non_hand, x_non_hand)
            loss_velocities_non_hand = nn.SmoothL1Loss()(
                x_hat_non_hand[:, 1:] - x_hat_non_hand[:, :-1], x_non_hand[:, 1:] - x_non_hand[:, :-1]
            )

            loss_reconstruction = loss_reconstruction_hand * self.ratio_for_hand + loss_reconstruction_non_hand * (
                1 - self.ratio_for_hand
            )
            loss_velocities = loss_velocities_hand * self.ratio_for_hand + loss_velocities_non_hand * (
                1 - self.ratio_for_hand
            )

        return loss_reconstruction + loss_velocities

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x = batch
        x_hat, vae_loss, _ = self.reconstruct(x)
        loss_reconstruction = self.compute_reconstruction_loss(x, x_hat, is_focus_hand_mode=self.is_focus_hand_mode)
        loss = loss_reconstruction + vae_loss

        self.log("val_loss", loss.detach(), prog_bar=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optim = torch.optim.Adam(self.parameters(), betas=self.betas)
        if self.learning_rate_scheduler == "static":
            optim = torch.optim.Adam(self.parameters(), betas=self.betas, lr=self.learning_rate)
            return optim

        elif self.learning_rate_scheduler == "lambda":
            if not isinstance(self.learning_rate, list):
                raise ValueError("learning_rate should be a list for lambda")

            def lr_lambda(epoch: int) -> float:
                for min_epoch, max_epoch, lr in self.learning_rate:
                    if min_epoch <= epoch <= max_epoch:
                        return lr
                return self.learning_rate[-1][-1]

            scheduler = LambdaLR(optim, lr_lambda)

        elif self.learning_rate_scheduler == "reduce_on_plateau":
            if not isinstance(self.learning_rate, dict):
                raise ValueError("learning_rate should be a dict for reduce_on_plateau")
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optim,
                mode="min",
                patience=self.learning_rate["patience"] if "patience" in self.learning_rate else 5,
                factor=self.learning_rate["factor"] if "factor" in self.learning_rate else 0.5,
            )

        else:
            raise ValueError(f"Unknown learning rate scheduler: {self.learning_rate_scheduler}")

        return {
            "optimizer": optim,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }

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
