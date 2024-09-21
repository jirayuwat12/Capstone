from lightning.pytorch import LightningModule
from lightning.pytorch.core.optimizer import LightningOptimizer
from torch.optim.optimizer import Optimizer
from T2M_GPT.models.vqvae import HumanVQVAE
from T2M_GPT.utils.losses import ReConsLoss
from T2M_GPT.models.encdec import Encoder, Decoder
import torch
from torch import nn
from argparse import Namespace

class HumanVQVAEWrapper(LightningModule):
    def __init__(self, args: Namespace) -> None:
        super().__init__()
        self.args = args
        self.loss = ReConsLoss(args.recons_loss, args.nb_joints)
        self.model = HumanVQVAE(args,
                                args.nb_code,
                                args.code_dim,
                                args.output_emb_width,
                                args.down_t,
                                args.stride_t,
                                args.width,
                                args.depth,
                                args.dilation_growth_rate,
                                "tanh",
                                None,
                                self.device)
        
        # Set init param to 0
        for p in self.model.parameters():
            p.data.fill_(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch

        pred_x, loss_commit, perplexity = self.model(x)
        loss_motion = self.loss(pred_x, y)
        loss_vel = self.loss.forward_vel(pred_x, y)

        loss = loss_motion + self.args.commit * loss_commit + self.args.loss_vel * loss_vel
        self.log("train_loss", loss, prog_bar=True)

        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr, betas=(0.9, 0.99), weight_decay=self.args.weight_decay)
        return optimizer

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch

        pred_x, loss_commit, perplexity = self.model(x)
        loss_motion = self.loss(pred_x, y)
        loss_vel = self.loss.forward_vel(pred_x, y)

        loss = loss_motion + self.args.commit * loss_commit + self.args.loss_vel * loss_vel

        self.log("val_loss", loss, prog_bar=True)