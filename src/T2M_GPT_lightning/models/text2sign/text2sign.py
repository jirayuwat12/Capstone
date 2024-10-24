from typing import Optional

import clip
import clip.model
import torch
import yaml

from T2M_GPT_lightning.models.vqvae.vqvae import VQVAEModel as VQVAE
from T2M_GPT_lightning.models_wrapper.t2m_trans_wrapper import Text2MotionTransformerWrapper as T2MTransformer


class Text2Sign:
    def __init__(
        self, vq_vae_model: VQVAE, clip_model: clip.clip, t2m_trans_model: T2MTransformer, device: Optional[str] = None
    ) -> None:
        """
        Initialize the Text2Sign model

        Args:
            vq_vae_model (VQVAE): VQ-VAE model
            clip_model (clip.clip): CLIP model
            t2m_trans_model (T2MTransformer): T2M Transformer model
            device (Optional[str]): Device to use for the models

        Raises:
            ValueError: If the input models are not of the correct type
        """
        if not isinstance(vq_vae_model, VQVAE):
            raise ValueError(f"vq_vae_model must be an instance of VQVAE ({type(vq_vae_model)})")
        if not isinstance(clip_model, clip.model.CLIP):
            raise ValueError(f"clip_model must be an instance of clip.model.CLIP ({type(clip_model)})")
        if not isinstance(t2m_trans_model, T2MTransformer):
            raise ValueError(f"T2MTransformer must be an instance of T2MTransformer ({type(t2m_trans_model)})")

        self.vq_vae_model = vq_vae_model
        self.clip_model = clip_model
        self.t2m_trans_model = t2m_trans_model

        if device is not None:
            self.vq_vae_model.to(device)
            self.clip_model.to(device)
            self.t2m_trans_model.to(device)

    @classmethod
    def from_path(
        cls,
        vq_vae_model_path: str,
        vq_vae_config_path: str,
        clip_model_path: str,
        t2m_trans_model_path: str,
        t2m_trans_config_path: str,
        device: Optional[str] = None,
    ) -> "Text2Sign":
        """
        Load the models from the given paths

        Args:
            vq_vae_model_path (str): Path to the VQ-VAE model
            vq_vae_config_path (str): Path to the VQ-VAE config
            clip_model_path (str): Path to the CLIP model
            t2m_model_path (str): Path to the T2M Transformer model
            t2m_config_path (str): Path to the T2M Transformer config
            device (Optional[str]): Device to use for the models

        Returns:
            Text2Sign: Instance of the Text2Sign class
        """
        # Load VQ-VAE model
        with open(vq_vae_config_path, "r") as f:
            vq_vae_config = yaml.safe_load(f)
        vq_vae_model = VQVAE.load_from_checkpoint(vq_vae_model_path, **vq_vae_config)
        vq_vae_model.eval()

        # Load CLIP model
        clip_model, _ = clip.load(clip_model_path)
        clip_model.eval()

        # Load T2M Transformer model
        with open(t2m_trans_config_path, "r") as f:
            t2m_trans_config = yaml.safe_load(f)
        t2m_trans_model = T2MTransformer.load_from_checkpoint(t2m_trans_model_path, **t2m_trans_config)
        t2m_trans_model.eval()

        return cls(vq_vae_model, clip_model, t2m_trans_model, device)

    def text_to_indices(self, text: str) -> torch.Tensor:
        """
        Convert text to indices

        Args:
            text (str): Input text

        Returns:
            torch.Tensor: Indices of the text
        """
        # Tokenize the text and get the text features
        tokenized_text = clip.tokenize(text)
        text_features = self.clip_model.encode_text(tokenized_text).detach().to(self.t2m_trans_model.device)

        with torch.no_grad():
            # Sample the skeletons
            skels_indices = self.t2m_trans_model.sample(text_features)
            # Remove the stop token index (first occurrence)
            stop_token_idx = self.vq_vae_model.quantizer.codebook_size
            stop_token_idx = torch.where(skels_indices == stop_token_idx)[0]
            if len(stop_token_idx) == 0:
                stop_token_idx = len(skels_indices)
            else:
                stop_token_idx = stop_token_idx[0].item()
            skels_indices = skels_indices[: stop_token_idx + 1]

        return skels_indices

    def text_to_skels(self, text: str) -> torch.Tensor:
        """
        Convert text to skeletons

        Args:
            text (str): Input text

        Returns:
            torch.Tensor: Skeletons which is shape of `(T, skel_dim)`
                - `T`: Number of frames in the animation
                - `skel_dim`: Dimension of the skeleton
        """
        skels_indices = self.text_to_indices(text)

        # Reconstruct the skeletons
        if self.vq_vae_model.device != skels_indices.device:
            skels_indices = skels_indices.to(self.vq_vae_model.device)
        skels = self.vq_vae_model.decode_indices(skels_indices)

        return skels

    def text_to_animation(self, text: str, animation_path: str) -> None:
        """
        Convert text to animation

        Args:
            text (str): Input text
            animation_path (str): Path to save the animation
        """
        raise NotImplementedError("text_to_animation method is not implemented yet")
