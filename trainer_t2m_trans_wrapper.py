import argparse
import warnings

import clip
import pandas as pd
import yaml
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import CSVLogger
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from T2M_GPT_lightning.dataset.toy_t2m_trans_dataset import ToyDataset
from T2M_GPT_lightning.models.vqvae.vqvae import VQVAEModel
from T2M_GPT_lightning.models_wrapper.t2m_trans_wrapper import Text2MotionTransformerWrapper

warnings.filterwarnings("ignore")

# Load the configuration
DEFAULT_CONFIG_PATH = "./configs/trainer_t2m_trans_wrapper.yaml"
parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, default=DEFAULT_CONFIG_PATH)
args = parser.parse_args()
CONFIG_PATH = args.config_path
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# Initialize the model
model_hyperparameters = config["model_hyperparameters"]
t2m_trans_model = Text2MotionTransformerWrapper(**model_hyperparameters)

# Initialize the dataset
clip_model, _ = clip.load("ViT-B/32")
vq_vae_model = VQVAEModel.load_from_checkpoint(config["vq_vae_model_path"], learning_rate=None)
if config["is_toy"]:
    train_dataset = ToyDataset(
        clip_model=clip_model,
        vq_vae_model=vq_vae_model,
        text_path=config["train_text_path"],
        skels_path=config["train_skels_path"],
        block_size=config["model_hyperparameters"]["block_size"],
    )
    val_dataset = ToyDataset(
        clip_model=clip_model,
        vq_vae_model=vq_vae_model,
        text_path=config["val_text_path"],
        skels_path=config["val_skels_path"],
        block_size=config["model_hyperparameters"]["block_size"],
    )
else:
    assert False, "Not implemented yet"

# Initialize the dataloaders
train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

# Initialize the logger
csv_logger = CSVLogger("logs", name="t2m_trans")

# Initialize the trainer
trainer = Trainer(log_every_n_steps=10, max_epochs=config["max_epochs"], logger=csv_logger)

# Train the model
trainer.fit(t2m_trans_model, train_loader, val_loader)

# Save the model
trainer.save_checkpoint(config["save_weights_path"])

# Get logging path
logging_path = t2m_trans_model.logger.log_dir

# Save config
with open(logging_path + "/config.yaml", "w") as f:
    yaml.dump(config, f)

# Read csv file
df = pd.read_csv(logging_path + "/metrics.csv")
# Plot the loss
plt.title("Train Loss")
plt.plot(df.loc[df["train_loss"].notnull(), "train_loss"])
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.show()