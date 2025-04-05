import yaml
import argparse
import os
import glob
import subprocess

# Create argparse object
parser = argparse.ArgumentParser()
parser.add_argument(
    "--config_path", type=str, default="./configs/trainer_vq_vae.yaml", help="Path to the config file"
)
parser.add_argument(
    "--epoch-per-iteration", type=int, required=True, help="Number of epochs per iteration"
)

args = parser.parse_args()
# Load the configuration
with open(args.config_path, "r") as config_file:
    main_config = yaml.safe_load(config_file)

os.makedirs("temp_config", exist_ok=True)

for epoch_num in range(args.epoch_per_iteration, main_config["max_epochs"]+args.epoch_per_iteration, args.epoch_per_iteration):
    # Create a new config for this iteration
    config = main_config.copy()
    config["max_epochs"] = epoch_num
    config["log_folder_name"] = f"{main_config['log_folder_name']}_iter_{epoch_num}"

    # {main_config["wandb_save_dir"]}/{main_config["log_folder_name"]}/wandb/run-*/files/checkpoints/*.ckpt
    # Find the latest checkpoint file
    prev_log_folder = f"{main_config['log_folder_name']}_iter_{epoch_num - args.epoch_per_iteration}"
    checkpoint_file = glob.glob(
        f"{main_config['wandb_save_dir']}/{prev_log_folder}/wandb/run-*/files/checkpoints/*.ckpt"
    )[0]
    config["resume_weight_path"] = checkpoint_file

    # Save the new config to a file
    temp_config_path = os.path.join("temp_config", f"config_iter_{epoch_num:06d}.yaml")
    with open(temp_config_path, "w") as temp_config_file:
        yaml.dump(config, temp_config_file)
