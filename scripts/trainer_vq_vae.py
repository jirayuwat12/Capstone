import argparse

import pandas as pd
import yaml
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import CSVLogger
from matplotlib import pyplot as plt
from T2M_GPT_lightning.dataset.toy_vq_vae_dataset import ToyDataset
from T2M_GPT_lightning.models.vqvae.vqvae import VQVAEModel
from torch.utils.data import DataLoader

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

# Initialize the dataset
train_dataset = ToyDataset(
    data_path=config["train_data_path"],
    joint_size=config["joint_size"],
    window_size=config["window_size"],
    normalise=config["normalize_data"],
)
test_dataset = ToyDataset(
    data_path=config["val_data_path"],
    joint_size=config["joint_size"],
    window_size=config["window_size"],
    normalise=config["normalize_data"],
)

# Initialize the dataloaders
train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

# Initialize the logger
csv_logger = CSVLogger("logs", name=config["log_folder_name"])

# Initialize the trainer
trainer = Trainer(log_every_n_steps=10, max_epochs=config["max_epochs"], logger=csv_logger)

# Train the model
trainer.fit(model, train_loader, test_loader)

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
# Show the plot
plt.show()
