import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
from torch.optim.lr_scheduler import LambdaLR

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
        optimizer_betas: tuple = (0.5, 0.99),
        optimizer_eps: float = 1e-8,
    ) -> None:
        """
        Initialize the Text2MotionTransformerWrapper

        :param num_vq: Number of VQ tokens (Codebook size)
        :param embed_dim: Embedding dimension
        :param clip_dim: Dimension of the CLIP feature
        :param block_size: Block size for the transformer
        :param num_layers: Number of transformer layers
        :param n_head: Number of attention heads
        :param drop_out_rate: Dropout rate
        :param fc_rate: Feedforward layer size
        :param learning_rate: Learning rate for the optimizer
        :param optimizer_betas: Betas for the AdamW optimizer
        :param optimizer_eps: Epsilon for the AdamW optimizer

        """
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
        self.save_hyperparameters()
        self.number_of_codebooks = num_vq
        self.total_tokens = num_vq + 1
        self.learning_rate = learning_rate
        self.optimizer_betas = optimizer_betas
        self.optimizer_eps = optimizer_eps

    def sample(self, clip_feature: torch.Tensor, if_categorial: bool = False) -> torch.Tensor:
        """
        Sample from the model

        :param clip_feature: CLIP feature
        :param if_categorial: If categorical, default is False
        :return: Sampled tensor
        """
        return self.trans_encoder.sample(clip_feature, if_categorial)

    def forward(self, idxs: torch.Tensor, clip_feature: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model

        :param idxs: Post tokenized indices
        :param clip_feature: CLIP feature
        :return: logits
        """
        return self.trans_encoder(idxs, clip_feature)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Training step for the model

        :param batch: Tuple of (text_features, skels_indices)
        :param batch_idx: Batch index
        :return: Loss for the batch
        """
        text_features, skels_indices = batch
        cls_pred = self.forward(skels_indices, text_features)
        cls_pred = cls_pred.contiguous()

        all_loss = []
        all_accuracy = []
        for batch in range(len(skels_indices)):
            loss_cls = nn.CrossEntropyLoss()(
                cls_pred[batch, :-1, :].view(-1, self.total_tokens), skels_indices[batch].view(-1)
            )
            accuracy = (cls_pred[batch, :-1, :].argmax(-1) == skels_indices[batch].view(-1)).float().mean()
            all_accuracy.append(accuracy)
            all_loss.append(loss_cls)

        loss_cls = torch.stack(all_loss).mean()
        avg_accuracy = torch.stack(all_accuracy).mean()
        self.log("train_loss", loss_cls, prog_bar=True)
        self.log("train_accuracy", avg_accuracy, prog_bar=True)

        # Log the learning rate
        self.log("learning_rate", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True)

        return loss_cls

    def configure_optimizers(self) -> dict:
        """
        Configure the optimizer and learning rate scheduler

        :return: Dictionary containing the optimizer and scheduler
        """
        optim = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
        )

        # Learning rate scheduler
        def lr_lambda(epoch: int) -> float:
            if epoch > 5000:
                return 5e-6
            else:
                return 1e-4

        scheduler = LambdaLR(optim, lr_lambda)

        return {
            "optimizer": optim,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Validation step for the model

        :param batch: Tuple of (text_features, skels_indices)
        :param batch_idx: Batch index
        :return: Loss for the batch
        """
        text_features, skels_indices = batch
        cls_pred = self.forward(skels_indices, text_features)  # Shape: (batch_size, seq_len, num_classes)
        cls_pred = cls_pred[:, :-1, :]

        # Reshape for loss computation
        logits = cls_pred.reshape(-1, self.total_tokens)  # Shape: (batch_size * seq_len, num_classes)
        targets = skels_indices.reshape(-1)  # Shape: (batch_size * seq_len)

        # Compute loss
        loss_cls = nn.CrossEntropyLoss()(logits, targets)

        # Log the loss (optional)
        self.log("val_loss", loss_cls, prog_bar=True)

        return loss_cls
