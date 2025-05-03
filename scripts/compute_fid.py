import os
import torch
import numpy as np
import yaml
from tqdm import tqdm
from T2M_GPT_lightning.dataset.toy_vq_vae_dataset import ToyDataset as VQVAE_Dataset
from T2M_GPT_lightning.models.vqvae.vqvae import VQVAEModel

from T2M_GPT.utils.eval_trans import calculate_frechet_feature_distance


CONFIG_PATH = "./configs/compute_fid.yaml"
with open(CONFIG_PATH, "r") as config_file:
    config = yaml.safe_load(config_file)


def compute_fid(config_folder: str):
    config_files = os.listdir(config_folder)

    looper = tqdm(config_files, desc="Computing FID", unit="file", disable=True)
    for config_file in looper:
        train_config_path = os.path.join(config_folder, config_file)

        # Check if the file is a YAML file
        if not train_config_path.endswith(".yaml"):
            looper.set_postfix_str(f"Skipping {config_file}")
            continue
        train_config = yaml.safe_load(open(train_config_path, "r"))

        # Check if the file contains the required keys
        if "train_data_path" not in train_config and "train_data_tensor_path" not in train_config:
            looper.set_postfix_str(f"Skipping {config_file}")
            continue

        # Load train/val datasets
        train_dataset = VQVAE_Dataset(
            data_path=train_config["train_data_path"] if "train_data_path" in train_config else None,
            data_tensor_path=train_config["train_data_tensor_path"] if "train_data_tensor_path" in train_config else None,
            joint_size=train_config["joint_size"],
            window_size=train_config["window_size"],
            normalise=train_config["normalize_data"],
            is_data_has_timestamp=train_config["is_data_has_timestamp"],
            data_spec=train_config["data_spec"] if "data_spec" in train_config else "all",
        )
        val_dataset = VQVAE_Dataset(
            data_path=train_config["val_data_path"] if "val_data_path" in train_config else None,
            data_tensor_path=train_config["val_data_tensor_path"] if "val_data_tensor_path" in train_config else None,
            joint_size=train_config["joint_size"],
            window_size=train_config["window_size"],
            normalise=train_config["normalize_data"],
            is_data_has_timestamp=train_config["is_data_has_timestamp"],
            data_spec=train_config["data_spec"] if "data_spec" in train_config else "all",
        )
        train_all = torch.concatenate(train_dataset.data, axis=0)
        val_all = torch.concatenate(val_dataset.data, axis=0)

        # Get predictions
        model = VQVAEModel.load_from_checkpoint(
            train_config["save_weight_path"],
            **train_config["model_hyperparameters"]
        )
        train_pred = []
        for i in range(len(train_dataset)):
            train_pred.append(model(train_dataset[i].unsqueeze(0).float())[0][0].detach().cpu().numpy())
        train_pred = torch.concatenate(train_pred, axis=0)
        val_pred = []
        for i in range(len(val_dataset)):
            val_pred.append(model(val_dataset[i].unsqueeze(0).float())[0][0].detach().cpu().numpy())
        val_pred = torch.concatenate(val_pred, axis=0)

        print(train_all.shape)
        print(train_pred.shape)
        print(val_all.shape)
        print(val_pred.shape)

            



if __name__ == "__main__":
    compute_fid(config["config_folder"])
