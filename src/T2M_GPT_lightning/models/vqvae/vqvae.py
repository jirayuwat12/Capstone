from typing import Any

import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
from torch.optim.lr_scheduler import LambdaLR

from T2M_GPT_lightning.models.vqvae.decoder import Decoder
from T2M_GPT_lightning.models.vqvae.encoder import Encoder
from T2M_GPT_lightning.models.vqvae.quantizer import Quantizer


class VQVAEModel(LightningModule):
    def __init__(
        self,
        learning_rate_scheduler: str = "static",
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
        loss_bone_length_multiplier: float = 0.0,
        minibatch_count_to_reset: int = 256
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
            loss_bone_length_multiplier (float): Multiplier for the bone length loss
        """
        super(VQVAEModel, self).__init__()
        self.save_hyperparameters()
        self.encoder = Encoder(L=L, in_dim=skels_dim, emb_dim=embedding_dim)
        self.decoder = Decoder(L=L, emb_dim=embedding_dim, out_dim=skels_dim)
        self.quantizer = Quantizer(codebook_size=codebook_size, decay=quantizer_decay, codebook_dim=embedding_dim, minibatch_count_to_reset=minibatch_count_to_reset)
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
        self.loss_bone_length_multiplier = loss_bone_length_multiplier

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
        loss_bone_length = self.bone_length_loss(x, x_hat)
        loss = loss_reconstruction + vae_loss + loss_bone_length * self.loss_bone_length_multiplier

        self.log("train_loss", loss.detach(), prog_bar=True)
        self.log("train_rec_loss", loss_reconstruction.detach(), prog_bar=True)
        self.log("train_vae_loss", vae_loss.detach(), prog_bar=True)
        self.log("train_bone_length_loss", loss_bone_length.detach(), prog_bar=True)

        # Log the learning rate
        self.log("learning_rate", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True)

        return loss

    def bone_length_loss(self, x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        """
        Compute the reconstruction loss

        Args:
            x (torch.Tensor): Input tensor shape (B, T, X)
                B: Batch size
                T: Sequence length
                X: Feature dimension
            x_hat (torch.Tensor): Reconstructed tensor shape (B, T, X)
    
                    B: Batch size
                    T: Sequence length
                    X: Feature dimension

        Returns:
            loss (torch.Tensor): Loss
        """
        x = x.view(x.shape[0], x.shape[1], -1, 3)
        x_hat = x_hat.view(x_hat.shape[0], x_hat.shape[1], -1, 3)

        connection_pairs = [

        # LEFT SIDE
        # upper arm (shoulder -> elbow)
        (532, 534),
        # lower arm (elbow -> wrist)
        (534, 536),
        # hand (wrist -> palm)
        (536, 538),
        (538, 540),
        (540, 542),
        (542, 536),
        # left hand
        # thumb
        (499, 500),
        (500, 501),
        (501, 502),
        (502, 503),
        # index
        (504, 505),
        (505, 506),
        (506, 507),
        # middle
        (508, 509),
        (509, 510),
        (510, 511),
        # ring finger
        (512, 513),
        (513, 514),
        (514, 515),
        # little finger
        (516, 517),
        (517, 518),
        (518, 519),


        # RIGHT SIDE
        # upper arm (shoulder -> elbow)
        (531, 533),
        # lower arm (elbow -> wrist)
        (533, 535),
        # hand (wrist -> palm)
        (535, 537),
        (537, 539),
        (539, 541),
        (541, 535),
        # right hand
        # thumb
        (478, 479),
        (479, 480),
        (480, 481),
        (481, 482),
        # index
        (483, 484),
        (484, 485),
        (485, 486),
        # middle
        (487, 488),
        (488, 489),
        (489, 490),
        # ring finger
        (491, 492),
        (492, 493),
        (493, 494),
        # little finger
        (495, 496),
        (496, 497),
        (497, 498),
        


        ]
        bone_length_x = []
        bone_length_x_hat = []
        for p1 , p2 in connection_pairs:
            v1 = x[:, :, p1, :]
            v2 = x[:, :, p2, :]

            v1_hat = x_hat[:, :, p1, :]
            v2_hat = x_hat[:, :, p2, :]
        
            length_x = torch.norm(v1 - v2, dim=-1)
            length_x_hat = torch.norm(v1_hat - v2_hat, dim=-1)

            bone_length_x.append(length_x)
            bone_length_x_hat.append(length_x_hat)

        bone_lengths_x = torch.stack(bone_length_x, dim=-1)
        bone_lengths_x_hat = torch.stack(bone_length_x_hat, dim=-1)

        bone_lengths_x = torch.mean(bone_lengths_x, dim=0) 
        bone_lengths_x_hat = torch.mean(bone_lengths_x_hat, dim=0)

        xq1 = torch.quantile(bone_lengths_x, 0.25 , dim=0)
        xq3 = torch.quantile(bone_lengths_x, 0.75 , dim=0)

        # Compute IQR
        iqr = xq3 - xq1

        # Compute outlier bounds
        lower_bound = xq1 - 1.5 * iqr
        upper_bound = xq3 + 1.5 * iqr

        mask = (bone_lengths_x_hat < lower_bound) | (bone_lengths_x_hat > upper_bound)
        x_hat_outlier = torch.where(mask, bone_lengths_x_hat, torch.tensor(float('nan')).to(bone_lengths_x_hat.device)) 
        
        return (torch.norm((torch.nansum(x_hat_outlier))/x_hat_outlier.shape[0])**0.5) 


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


if __name__ == "__main__":
    model = VQVAEModel()

    x = torch.randn(2, 64, 150)
    x_hat, commitment_loss, perplexity = model.reconstruct(x)
    print("x_hat", x_hat)
    print("commitment_loss", commitment_loss)
    print("perplexity", perplexity)
    print(x_hat.shape, commitment_loss, perplexity)
