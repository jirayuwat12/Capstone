import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
from lightning.pytorch.core.optimizer import LightningOptimizer
from torch.optim.optimizer import Optimizer

from T2M_GPT.models.t2m_trans import Text2Motion_Transformer


class Text2MotionTransformerWrapper(LightningModule):
    def __init__(
        self,
        num_vq: int,
        embed_dim: int,
        clip_dim: int,
        block_size: int,
        num_layers: int,
        n_head: int,
        drop_out_rate: float,
        fc_rate: int,
        learning_rate: float,
    ):
        super().__init__()
        self.trans_encoder = Text2Motion_Transformer(
            num_vq=num_vq,
            embed_dim=embed_dim,
            clip_dim=clip_dim,
            block_size=block_size,
            num_layers=num_layers,
            n_head=n_head,
            drop_out_rate=drop_out_rate,
            fc_rate=fc_rate,
        )
        self.number_of_codebooks = num_vq
        self.total_tokens = num_vq + 1
        self.learning_rate = learning_rate

    def sample(self, clip_feature: torch.Tensor, if_categorial: bool = False) -> torch.Tensor:
        """
        Sample from the model

        Args:
            clip_feature: Clip feature
            if_categorial: If categorial

        Returns:
            xs: Sampled tensor
        """
        return self.trans_encoder.sample(clip_feature, if_categorial)

    def forward(self, idxs: torch.Tensor, clip_feature: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model

        Args:
            idxs: Post tokenized indices
            clip_feature: Clip feature

        Returns:
            logits: Logits
        """
        return self.trans_encoder(idxs, clip_feature)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        text_features, skels_indices = batch
        cls_pred = self.forward(skels_indices, text_features)
        cls_pred = cls_pred.contiguous()

        all_loss = []
        for batch in range(len(skels_indices)):
            print("batch:", batch)
            print("\tcls_pred:", cls_pred[batch].shape)
            print("\tskels_indices:", skels_indices[batch].shape)
            loss_cls = nn.CrossEntropyLoss()(
                cls_pred[batch, :-1, :].view(-1, self.total_tokens), skels_indices[batch].view(-1)
            )
            all_loss.append(loss_cls)

        loss_cls = torch.stack(all_loss).mean()
        self.log("train_loss", loss_cls, prog_bar=True)

        return loss_cls

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate, betas=(0.5, 0.9), eps=1e-8)

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        text_features, skels_indices = batch
        cls_pred = self.forward(skels_indices, text_features)  # Shape: (batch_size, seq_len, num_classes)

        # Reshape for loss computation
        logits = cls_pred.view(-1, self.total_tokens)[:-1, :]  # Shape: (batch_size * seq_len, num_classes)
        targets = skels_indices.view(-1)  # Shape: (batch_size * seq_len)

        # Compute loss
        loss_cls = nn.CrossEntropyLoss()(logits, targets)

        # Log the loss (optional)
        self.log("val_loss", loss_cls, prog_bar=True)

        return loss_cls