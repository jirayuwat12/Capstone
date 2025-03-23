import argparse
import os

import lightning
import yaml
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader

import wandb
from capstone_utils.dataloader.collate_fn import minibatch_padding_collate_fn
from T2M_GPT_lightning.dataset.toy_vq_vae_dataset import ToyDataset
from T2M_GPT_lightning.models.vqvae.vqvae import VQVAEModel

# Set the random seed
lightning.seed_everything(42)

# Load the configuration
DEFAULT_CONFIG_PATH = "./configs/trainer_vq_vae.yaml"
parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, default=DEFAULT_CONFIG_PATH)
args = parser.parse_args()
CONFIG_PATH = args.config_path
with open(CONFIG_PATH, "r") as config_file:
    config = yaml.safe_load(config_file)

# Login to wandb
wandb.login(key=config["wandb_api_key"])

# Load the model
model_hyperparameters = config["model_hyperparameters"]
model = VQVAEModel(**model_hyperparameters)

# Initialize the dataset
train_dataset = ToyDataset(
    data_path=config["train_data_path"],
    joint_size=config["joint_size"],
    window_size=config["window_size"],
    normalise=config["normalize_data"],
    is_data_has_timestamp=config["is_data_has_timestamp"],
)
test_dataset = ToyDataset(
    data_path=config["val_data_path"],
    joint_size=config["joint_size"],
    window_size=config["window_size"],
    normalise=config["normalize_data"],
    is_data_has_timestamp=config["is_data_has_timestamp"],
)

# Initialize the dataloaders
train_loader = DataLoader(
    train_dataset, batch_size=config["batch_size"], collate_fn=minibatch_padding_collate_fn, shuffle=True
)
test_loader = DataLoader(
    test_dataset, batch_size=config["batch_size"], collate_fn=minibatch_padding_collate_fn, shuffle=False
)

# Initialize the logger
wandb_logger = WandbLogger(name=config["log_folder_name"], project="vqvae", log_model=True)

# Create checkpoint callback
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath=os.path.join(wandb_logger.experiment.dir, "checkpoints"),
    filename="vqvae-{epoch:02d}-{val_loss:.4f}",
    save_top_k=1,
    mode="min",
    every_n_epochs=config["save_every_n_epochs"],
)

# Initialize the trainer
trainer = Trainer(
    log_every_n_steps=10, max_epochs=config["max_epochs"], logger=wandb_logger, callbacks=[checkpoint_callback]
)

# Train the model
trainer.fit(
    model, train_loader, test_loader, ckpt_path=config["resume_weight_path"] if config["resume_weight_path"] else None
)

# Save the model
trainer.save_checkpoint(config["save_weight_path"])
