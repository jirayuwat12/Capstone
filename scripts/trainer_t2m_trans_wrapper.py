import argparse
import time
import warnings

import clip
import pandas as pd
import yaml
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import DeviceStatsMonitor
from lightning.pytorch.loggers import CSVLogger
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from capstone_utils.dataloader.collate_fn import minibatch_padding_collate_fn
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
if config["resume_weights_path"] == "":
    t2m_trans_model = Text2MotionTransformerWrapper(**model_hyperparameters)
else:
    print("Loading model from checkpoint")
    t2m_trans_model = Text2MotionTransformerWrapper.load_from_checkpoint(
        config["resume_weights_path"], **model_hyperparameters
    )

start_load_dataset_time = time.time()
# Initialize the dataset
clip_model, _ = clip.load(config['clip_model_path'])
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
print(f"T2M trans load dataset time: {time.time() - start_load_dataset_time}")

# Initialize the logger
csv_logger = CSVLogger("logs", name=config["log_folder_name"])

# Create device stats monitor
device_stats_monitor = DeviceStatsMonitor()

# Initialize the trainer
trainer = Trainer(
    log_every_n_steps=10, max_epochs=config["max_epochs"], logger=csv_logger, callbacks=[device_stats_monitor], strategy='ddp_find_unused_parameters_true'
)

start_train_time = time.time()
# Train the model
trainer.fit(t2m_trans_model, train_loader, val_loader)
print(f"T2M trans train time: {time.time() - start_train_time}")

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
# Save the plot
plt.savefig(logging_path + "/train_loss.png")
