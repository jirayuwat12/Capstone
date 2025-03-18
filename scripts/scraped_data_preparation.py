import argparse
import json
import logging
import os

import pandas as pd
import yaml
from tqdm import tqdm

from .convert_vdo_to_skeletons import convert_vdo_to_skeleton_main
from .deblur_vdo_using_BIN import deblur_vdo_using_BIN_main
from .norm_standardize import norm_standardize

# Create a parser object
args = argparse.ArgumentParser()
args.add_argument("--config_path", default="./configs/scraped_data_preparation.yaml")
args = args.parse_args()

# Load the configuration file
with open(args.config_path, "r") as file:
    config = yaml.safe_load(file)

# Create logger
os.makedirs(config["log_folder"], exist_ok=True)
log_file = os.path.join(config["log_folder"], "data_preparation.log")
# Add show on standard output as well
logging.basicConfig(
    level=logging.DEBUG if config["log_level"] == "DEBUG" else logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)
logging.info("Starting data preparation")

# Use BIN to deblur the video
if config["use_bin"]:
    logging.info("BIN is used for deblurring the video")
    mp4_folder = config["target_folder"]
    looper = tqdm(os.listdir(config["scraped_folder"]), total=len(os.listdir(mp4_folder)), desc=f"Deblurring the video")
    for video_file in looper:
        input_vdo_path = os.path.join(config["scraped_folder"], video_file)
        if input_vdo_path.endswith(".mp4"):
            deblur_vdo_using_BIN_main(
                input_vdo_path=input_vdo_path,
                output_vdo_path=os.path.join(mp4_folder, video_file),
                model_net_name=config["bin_config"]["model_net_name"],
                num_interpolation=config["bin_config"]["num_interpolation"],
                time_step=config["bin_config"]["time_step"],
                model_option_yaml=config["bin_config"]["model_option_yaml"],
            )
else:
    logging.info("BIN is not used for deblurring the video")
    # Move the video files to the target folder
    logging.info("Moving the video files to the target folder")
    os.makedirs(config["target_folder"], exist_ok=True)
    for file in os.listdir(config["scraped_folder"]):
        file_path = os.path.join(config["scraped_folder"], file)
        if os.path.isfile(file_path) and file.endswith(".mp4"):
            os.rename(file_path, os.path.join(config["target_folder"], file))
    logging.info("All video files moved successfully")

# Move the json files to the target folder
logging.info("Moving the json files to the target folder")
os.makedirs(config["target_folder"], exist_ok=True)
for file in os.listdir(config["scraped_folder"]):
    file_path = os.path.join(config["scraped_folder"], file)
    if os.path.isfile(file_path) and file.endswith(".json"):
        os.rename(file_path, os.path.join(config["target_folder"], file))
logging.info("All json files moved successfully")

# Create skeleton files
logging.info("Creating the skeleton files")
convert_vdo_to_skeleton_main(
    face_model_config=config["vdo_to_skeletons_config"]["face_model"],
    hand_model_config=config["vdo_to_skeletons_config"]["hand_model"],
    pose_model_config=config["vdo_to_skeletons_config"]["pose_model"],
    output_folder=config["target_folder"],
    landmarks_format=config["vdo_to_skeletons_config"]["landmarks_format"],
    save_format=config["vdo_to_skeletons_config"]["save_format"],
    is_return_landmarked_vdo=config["vdo_to_skeletons_config"]["is_return_landmarked_vdo"],
    max_distance_between_predicted_hand_and_approximated_hand=config["vdo_to_skeletons_config"][
        "max_distance_between_predicted_hand_and_approximated_hand"
    ],
    save_stats=config["vdo_to_skeletons_config"]["save_stats"],
    vdo_folder=config["target_folder"],
    yes=True,
)
logging.info("All skeleton files created successfully")

# Normalization to Reference Skeleton & Min-Max Scaling
logging.info("Normalization to Reference Skeleton & Min-Max Scaling")
train_folder = config["target_folder"]
reference_dir = None
for file in os.listdir(train_folder):
    file_path = os.path.join(train_folder, file)
    if os.path.isfile(file_path) and file.endswith(".npy"):
        reference_dir = file_path
if reference_dir is None:
    logging.error("Reference skeleton file not found")
    exit(1)
norm_standardize(
    input_dir=config["target_folder"],
    reference_dir=reference_dir,
    output_file=config["target_folder"],
    iterate_split_folder=False,
)
logging.info("Normalization, Scaling successfully")

# Create .txt files for the dataset
logging.info("Creating the .txt files for the dataset")
file_name_list = sorted(os.listdir(os.path.join(config["target_folder"])))
output_path = os.path.join(config["target_folder"], f"train.txt")
if os.path.exists(output_path):
    logging.info(f"Skipping the .txt file for train as the file already exists")
else:
    with open(output_path, "w") as f:
        for file_name in tqdm(file_name_list):
            if file_name.endswith(".json"):
                json_file = json.load(open(os.path.join(config["target_folder"], file_name)))
                f.write(f"{json_file['text']}\n")
logging.info("All .txt files created successfully")

logging.info("Data preparation completed")
