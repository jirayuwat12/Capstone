import argparse
import os

import pandas as pd
import time
import yaml
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from capstone_utils.dataloader.collate_fn import minibatch_padding_collate_fn
from T2M_GPT_lightning.dataset.toy_vq_vae_dataset import ToyDataset
from T2M_GPT_lightning.models.vqvae.vqvae import VQVAEModel

# Load the configuration
DEFAULT_CONFIG_PATH = "./configs/trainer_vq_vae.yaml"
parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, default=DEFAULT_CONFIG_PATH)
args = parser.parse_args()
CONFIG_PATH = args.config_path
with open(CONFIG_PATH, "r") as config_file:
    config = yaml.safe_load(config_file)

# Load the model
model_hyperparameters = config["model_hyperparameters"]
if config["resume_weight_path"]:
    model = VQVAEModel.load_from_checkpoint(config["resume_weight_path"], **model_hyperparameters)
else:
    model = VQVAEModel(**model_hyperparameters)

start_load_dataset_time = time.time()
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
print(f"VQVAR load dataset time: {time.time() - start_load_dataset_time}")

# Initialize the logger
csv_logger = CSVLogger("logs", name=config["log_folder_name"])

# Create checkpoint callback
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath=os.path.join(csv_logger.log_dir, "checkpoints"),
    filename="vqvae-{epoch:02d}-{val_loss:.4f}",
    save_top_k=1,
    mode="min",
    every_n_epochs=config["save_every_n_epochs"],
)

# Initialize the trainer
trainer = Trainer(
    log_every_n_steps=10, max_epochs=config["max_epochs"], logger=csv_logger, callbacks=[checkpoint_callback]
)

start_train_time = time.time()
# Train the model
trainer.fit(model, train_loader, test_loader)
print(f"VQVAE train time: {time.time() - start_train_time}")

# Save config file into the logging directory
with open(csv_logger.log_dir + "/config.yaml", "w") as config_file:
    yaml.safe_dump(config, config_file)

# Save the model
trainer.save_checkpoint(config["save_weight_path"])

# Get logging path
logging_path = model.logger.log_dir
# Read csv file
df = pd.read_csv(logging_path + "/metrics.csv")
# Plot the loss
plt.title("Train Loss")
plt.plot(df.loc[df["train_loss"].notnull(), "train_loss"])
plt.xlabel("Epoch")
plt.ylabel("Loss")
# Save the plot
plt.savefig(logging_path + "/train_loss.png")
