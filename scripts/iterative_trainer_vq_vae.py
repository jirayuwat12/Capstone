import argparse
import glob
import os
import shutil
import subprocess

import yaml

# Create argparse object
parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, default="./configs/trainer_vq_vae.yaml", help="Path to the config file")
parser.add_argument("--epoch-per-iteration", type=int, required=True, help="Number of epochs per iteration")
parser.add_argument("--python-path", type=str, default="python")
parser.add_argument("--start-epoch", type=int, default=None, help="Starting epoch for the training")
parser.add_argument("--backup-every", type=int, default=1000, help="Backup every N epochs")

args = parser.parse_args()

# Load the configuration
with open(args.config_path, "r") as config_file:
    main_config = yaml.safe_load(config_file)

os.makedirs("temp_config", exist_ok=True)

for epoch_num in range(
    args.start_epoch if args.start_epoch is not None else args.epoch_per_iteration,
    main_config["max_epochs"] + args.epoch_per_iteration,
    args.epoch_per_iteration,
):
    # Create a new config for this iteration
    config = main_config.copy()
    config["max_epochs"] = min(epoch_num, main_config["max_epochs"])
    config["log_folder_name"] = f"{main_config['log_folder_name']}"

    # Find the latest checkpoint file
    checkpoint_file = None
    if epoch_num != args.epoch_per_iteration:
        prev_log_path = os.path.join(
            main_config["wandb_save_dir"],
            config["log_folder_name"],
            "wandb",
            "run-*",
            "files",
            "checkpoints",
            "*.ckpt",
        )
        checkpoint_file = sorted(glob.glob(prev_log_path))[-1]
        # if it is first epoch in each 1K epoch, then save the checkpoint file
        if epoch_num // args.backup_every != (epoch_num - args.epoch_per_iteration) // args.backup_every:
            os.makedirs(
                os.path.join(
                    main_config["wandb_save_dir"],
                    config["log_folder_name"],
                    "wandb",
                    f"backup-{epoch_num:06d}",
                ),
                exist_ok=True,
            )
            shutil.copy(
                checkpoint_file,
                os.path.join(
                    main_config["wandb_save_dir"],
                    config["log_folder_name"],
                    "wandb",
                    f"backup-{epoch_num:06d}",
                    os.path.basename(checkpoint_file),
                ),
            )
        # check that epoch are too far
        checkpoint_epoch = int(checkpoint_file.split("epoch=")[-1].split("-")[0])
        if epoch_num - checkpoint_epoch - 1 > args.epoch_per_iteration:
            print(f"Checkpoint epoch {checkpoint_epoch} is too far from current epoch {epoch_num}.")
            break
        # Update the config with the path to the checkpoint file
        config["resume_weight_path"] = checkpoint_file

        # Add the wandb_run_id to the config (run-<not used>-<id>)
        run_id = os.path.dirname(os.path.dirname(os.path.dirname(checkpoint_file))).split("-")[-1]
        config["wandb_run_id"] = run_id
    else:
        config["resume_weight_path"] = ""

    # Save the new config to a file
    temp_config_path = os.path.join("temp_config", f"config_iter_{epoch_num:06d}.yaml")
    with open(temp_config_path, "w") as temp_config_file:
        yaml.dump(config, temp_config_file)

    # Run the training script with the new config
    subprocess.run(
        [
            args.python_path,
            "-m",
            "scripts.trainer_vq_vae",
            "--config",
            temp_config_path,
        ]
    )

    # Clean up the temp config file
    os.remove(temp_config_path)
    # Remove the previous log folder if it exists
    if checkpoint_file is not None:
        prev_log_folder = os.path.dirname(os.path.dirname(os.path.dirname(checkpoint_file)))
        if os.path.exists(prev_log_folder):
            shutil.rmtree(prev_log_folder)
