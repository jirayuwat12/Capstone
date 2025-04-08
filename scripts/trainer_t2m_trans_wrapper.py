import argparse
import warnings

import clip
import lightning
import wandb
import yaml
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader

from capstone_utils.dataloader.collate_fn import minibatch_padding_collate_fn
from T2M_GPT_lightning.dataset.toy_t2m_trans_dataset import ToyDataset
from T2M_GPT_lightning.models.vqvae.vqvae import VQVAEModel
from T2M_GPT_lightning.models_wrapper.t2m_trans_wrapper import Text2MotionTransformerWrapper

warnings.filterwarnings("ignore")

# Set the random seed
lightning.seed_everything(42)

# Load the configuration
DEFAULT_CONFIG_PATH = "./configs/trainer_t2m_trans_wrapper.yaml"
parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, default=DEFAULT_CONFIG_PATH)
args = parser.parse_args()
CONFIG_PATH = args.config_path
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# Login to wandb
wandb.login(key=config["wandb_api_key"])

# Initialize the model
model_hyperparameters = config["model_hyperparameters"]
if config["resume_weights_path"] == "":
    t2m_trans_model = Text2MotionTransformerWrapper(**model_hyperparameters)
else:
    print("Loading model from checkpoint")
    t2m_trans_model = Text2MotionTransformerWrapper.load_from_checkpoint(
        config["resume_weights_path"], **model_hyperparameters
    )

# Initialize the dataset
clip_model, _ = clip.load("ViT-B/32")
with open(config["vq_vae_model_config_path"], "r") as f:
    vq_vae_config = yaml.safe_load(f)
vq_vae_model = VQVAEModel.load_from_checkpoint(config["vq_vae_model_path"], **vq_vae_config["model_hyperparameters"])
if config["is_toy"]:
    train_dataset = ToyDataset(
        clip_model=clip_model,
        vq_vae_model=vq_vae_model,
        text_path=config["train_text_path"],
        skels_path=config["train_skels_path"],
        joint_size=vq_vae_config["joint_size"],
        has_timestamp=vq_vae_config["is_data_has_timestamp"],
    )
    val_dataset = ToyDataset(
        clip_model=clip_model,
        vq_vae_model=vq_vae_model,
        text_path=config["val_text_path"],
        skels_path=config["val_skels_path"],
        joint_size=vq_vae_config["joint_size"],
        has_timestamp=vq_vae_config["is_data_has_timestamp"],
    )
else:
    assert False, "Not implemented yet"

# Initialize the dataloaders
train_loader = DataLoader(
    train_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=minibatch_padding_collate_fn
)
val_loader = DataLoader(
    val_dataset, batch_size=config["batch_size"], shuffle=False, collate_fn=minibatch_padding_collate_fn
)

# Initialize the logger
wandb_logger = WandbLogger(name=config["log_folder_name"], project="t2m-trans", log_model=True)

# Initialize the trainer
trainer = Trainer(log_every_n_steps=10, max_epochs=config["max_epochs"], logger=wandb_logger)

# Train the model
trainer.fit(t2m_trans_model, train_loader, val_loader)

# Save the model
trainer.save_checkpoint(config["save_weights_path"])
