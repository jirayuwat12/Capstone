from argparse import Namespace

import torch
from lightning.pytorch import LightningModule
from lightning.pytorch.core.optimizer import LightningOptimizer
from T2M_GPT.models.encdec import Decoder, Encoder
from T2M_GPT.models.vqvae import HumanVQVAE
from T2M_GPT.utils.losses import ReConsLoss
from torch import nn
from torch.optim.optimizer import Optimizer


class HumanVQVAEWrapper(LightningModule):
    def __init__(self, args: Namespace) -> None:
        super().__init__()
        self.args = args
        self.model = HumanVQVAE(
            args,
            args.nb_code,
            args.code_dim,
            args.output_emb_width,
            args.down_t,
            args.stride_t,
            args.width,
            args.depth,
            args.dilation_growth_rate,
            "relu",
            None,
            self.device,
        )
        self.smooth_l1_loss = nn.SmoothL1Loss()

        # Initialize the model weights to 0
        for m in self.model.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.zeros_(m.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def get_codebook_indices(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.encode(x)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x = batch

        pred_x, loss_commit, _ = self.model(x)

        if batch_idx % 10 == 0:
            print("x = ", x)
            print("pred_x = ", pred_x)

        motion_loss = self.compute_motion_loss(x, pred_x)
        loss = loss_commit + motion_loss
        print("motion_loss = ", motion_loss)
        print("loss_commit = ", loss_commit)
        self.log("train_loss", loss, prog_bar=True)

        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.args.lr, betas=(0.9, 0.99), weight_decay=self.args.weight_decay
        )
        return optimizer

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x = batch

        pred_x, loss_commit, _ = self.model(x)

        motion_loss = self.compute_motion_loss(x, pred_x)
        loss = loss_commit + motion_loss

        self.log("val_loss", loss, prog_bar=True)

    def compute_motion_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Compute the motion loss which
        # x shape = (batch_size, seq_len, nb_joints)
        # y shape = (batch_size, seq_len, nb_joints)
        # Then motion loss would be the average of the smooth l1 loss each minibatch
        return self.smooth_l1_loss(x, y).mean()


if __name__ == "__main__":
    from T2M_GPT.options import option_vq

    argsparser = option_vq.get_args_parser()
    argsparser.add_argument("--nb_joints", type=int, default=150, help="number of joints")
    args = argsparser.parse_args()

    model = HumanVQVAEWrapper(args)
    x = torch.randn(1, 32, 150)
    y = torch.randn(1, 32, 150)

    x_hat, _, _ = model(x)
    print("x_hat = ", x_hat)
    print("x_hat shape = ", x_hat.shape)
    print("x shape = ", x.shape)

    import torchsummary

    torchsummary.summary(model.model.vqvae.encoder, (150, 32))
