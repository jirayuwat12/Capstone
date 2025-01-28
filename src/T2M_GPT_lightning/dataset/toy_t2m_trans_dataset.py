import clip
import torch
from torch import nn
from torch.utils.data import Dataset

from T2M_GPT_lightning.models.vqvae.vqvae import VQVAEModel


class ToyDataset(Dataset):
    def __init__(
        self, clip_model: nn.Module, vq_vae_model: VQVAEModel, text_path: str, skels_path: str, block_size: int
    ) -> None:
        super().__init__()

        self.clip_model = clip_model.eval()
        self.vq_vae_model = vq_vae_model.eval()

        self.text_path = text_path
        self.skels_path = skels_path
        self.block_size = block_size

        self.texts = self.load_texts()
        self.texts_features = self.get_text_features().to(self.vq_vae_model.device)

        self.skels = self.load_skels()

    def load_skels(self) -> list[torch.Tensor]:
        skels = []
        with open(self.skels_path, "r") as f:
            for line in f.readlines():
                skel = torch.tensor([float(val) for val in line.strip().split()])
                skel = skel.reshape(-1, 151)[:, :150]
                skels.append(skel)
        return skels

    def load_texts(self) -> list[str]:
        with open(self.text_path, "r") as f:
            return [text.strip() for text in f.readlines()]

    def get_text_features(self) -> torch.Tensor:
        tokenized_texts = clip.tokenize(self.texts)
        return self.clip_model.encode_text(tokenized_texts).detach()

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            text_features: Text features
            skels_indices: Skeleton indices
        """
        skel = self.skels[idx].unsqueeze(0)
        if self.vq_vae_model.device != skel.device:
            skel = skel.to(self.vq_vae_model.device)
        code_indices = self.vq_vae_model.compute_codebook_indices(skel)[0].squeeze(-1).detach()
        # add stop token
        code_indices = torch.cat(
            [code_indices, torch.tensor([self.vq_vae_model.quantizer.codebook_size], device=code_indices.device)]
        )
        return self.texts_features[idx], code_indices


if __name__ == "__main__":
    from T2M_GPT_lightning.models.vqvae.vqvae import VQVAEModel

    vae_model = VQVAEModel.load_from_checkpoint(
        "./src/T2M_GPT_lightning/models/vqvae/weights/vq_vae_model.pth", learning_rate=0
    )

    clip_model, _ = clip.load("ViT-B/32")
    dataset = ToyDataset(clip_model, vae_model, "./data/toy_data/train.text", "./data/toy_data/train.skels")

    print(dataset[0][0].shape, dataset[0][1].shape)
