import torch
import yaml

from capstone_utils.skeleton_utils.skeleton import (
    BODY_RANGE_IN_FLATTENED,
    CORE_RANGE_IN_FLATTENED,
    FACE_RANGE_IN_FLATTENED,
    HAND_RANGE_IN_FLATTENED,
)
from T2M_GPT_lightning.models.vqvae import VQVAEModel


class EnsembleVQVAEModel:
    def __init__(
        self,
        face_model_path: str | None = None,
        body_model_path: str | None = None,
        hand_model_path: str | None = None,
        core_model_path: str | None = None,
        face_model_config: dict | None = None,
        body_model_config: dict | None = None,
        hand_model_config: dict | None = None,
        core_model_config: dict | None = None,
    ) -> None:
        self.face_model = self.load_model(face_model_path, face_model_config) if face_model_path else None
        self.body_model = self.load_model(body_model_path, body_model_config) if body_model_path else None
        self.hand_model = self.load_model(hand_model_path, hand_model_config) if hand_model_path else None
        self.core_model = self.load_model(core_model_path, core_model_config) if core_model_path else None

    def load_model(self, model_path: str, config_path: str, device: str = "cpu", is_train: bool = False) -> VQVAEModel:
        model = VQVAEModel.load_from_checkpoint(
            model_path, **self.load_config(config_path)["model_hyperparameters"]
        ).to(device)
        if not is_train:
            model.eval()

        return model

    def load_config(self, config_path: str) -> dict:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        return config

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return_x = torch.zeros(x.shape[1], 1659)

        if self.face_model:
            face_x = x[:, :, FACE_RANGE_IN_FLATTENED.start : FACE_RANGE_IN_FLATTENED.end]
            recon_face = recon_face[0][0].detach()
            recon_face = self.face_model(face_x)
            return_x[:, FACE_RANGE_IN_FLATTENED.start : FACE_RANGE_IN_FLATTENED.end] = recon_face

        if self.body_model:
            body_x = x[:, :, BODY_RANGE_IN_FLATTENED.start : BODY_RANGE_IN_FLATTENED.end]
            recon_body = self.body_model(body_x)
            recon_body = recon_body[0][0].detach()
            return_x[:, BODY_RANGE_IN_FLATTENED.start : BODY_RANGE_IN_FLATTENED.end] = recon_body

        if self.core_model:
            core_x = x[:, :, CORE_RANGE_IN_FLATTENED.start : CORE_RANGE_IN_FLATTENED.end]
            recon_core = self.core_model(core_x)
            recon_core = recon_core[0][0].detach()
            return_x[:, CORE_RANGE_IN_FLATTENED.start : CORE_RANGE_IN_FLATTENED.end] = recon_core

        if self.hand_model:
            hand_x = x[:, :, HAND_RANGE_IN_FLATTENED.start : HAND_RANGE_IN_FLATTENED.end]
            recon_hand = self.hand_model(hand_x)
            recon_hand = recon_hand[0][0].detach()
            return_x[:, HAND_RANGE_IN_FLATTENED.start : HAND_RANGE_IN_FLATTENED.end] = recon_hand

        return return_x.reshape(1, 1, -1, 1659)
