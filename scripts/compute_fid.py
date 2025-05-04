import os

import numpy as np
import torch
import yaml
from tqdm import tqdm

from T2M_GPT.utils.eval_trans import calculate_frechet_feature_distance
from T2M_GPT_lightning.dataset.toy_vq_vae_dataset import ToyDataset as VQVAE_Dataset
from T2M_GPT_lightning.models.vqvae.vqvae import VQVAEModel

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
            print(f"Skipping {config_file} as it is not a YAML file")
            continue
        train_config = yaml.safe_load(open(train_config_path, "r"))

        # Check if the file contains the required keys
        if "train_data_path" not in train_config and "train_data_tensor_path" not in train_config:
            print(f"Skipping {config_file} as it does not contain train_data_path or train_data_tensor_path")
            continue
    
        # Check if the model save path exists
        if not os.path.exists(train_config["save_weight_path"]):
            print(f"Model save path {train_config['save_weight_path']} does not exist")
            continue

        # Check if already computed
        fid_path = train_config["save_weight_path"].replace(".pth", "_fid.txt")
        if os.path.exists(fid_path):
            print(f"FID already computed for {train_config['save_weight_path']}")
            continue

        # Load train/val datasets
        train_dataset = VQVAE_Dataset(
            data_path=train_config["train_data_path"] if "train_data_path" in train_config else None,
            data_tensor_path=(
                train_config["train_data_tensor_path"] if "train_data_tensor_path" in train_config else None
            ),
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
        train_all = []
        val_all = []

        # Get predictions
        model = VQVAEModel.load_from_checkpoint(
            train_config["save_weight_path"], **train_config["model_hyperparameters"]
        ).to(device="cuda" if torch.cuda.is_available() else "cpu")
        train_pred = []
        # for i in range(len(train_dataset)):
        for i in tqdm(range(len(train_dataset)), desc="Computing train predictions", unit="sample", leave=False):
            train_pred.append(model(train_dataset[i].unsqueeze(0).float().to(model.device))[0][0].detach().cpu())
            train_all.append(train_dataset[i].reshape(-1, train_config["model_hyperparameters"]["skels_dim"])[: train_pred[-1].shape[0]])
        train_all = torch.concatenate(train_all, dim=0).to(device="cpu")
        train_pred = torch.concatenate(train_pred, axis=0).to(device="cpu")
        val_pred = []
        # for i in range(len(val_dataset)):
        for i in tqdm(range(len(val_dataset)), desc="Computing val predictions", unit="sample", leave=False):
            val_pred.append(model(val_dataset[i].unsqueeze(0).float().to(model.device))[0][0].detach().cpu())
            val_all.append(val_dataset[i].reshape(-1, train_config["model_hyperparameters"]["skels_dim"])[: val_pred[-1].shape[0]])
        val_all = torch.concatenate(val_all, dim=0).to(device="cpu")
        val_pred = torch.concatenate(val_pred, axis=0).to(device="cpu")

        del train_dataset
        del val_dataset

        # Compute FID
        train_fid = calculate_frechet_feature_distance(
            train_pred,
            train_all,
        )
        val_fid = calculate_frechet_feature_distance(
            val_pred,
            val_all,
        )
        looper.set_postfix_str(f"Train FID: {train_fid:.4f}, Val FID: {val_fid:.4f}")
        # Save FID
        with open(fid_path, "w") as fid_file:
            fid_file.write(f"train_fid: {train_fid:.4f}\n val_fid: {val_fid:.4f}\n")

        print(f"Model: {os.path.basename(train_config['save_weight_path'])} done")


if __name__ == "__main__":
    compute_fid(config["config_folder"])
