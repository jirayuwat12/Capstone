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
args.add_argument("--config", default="./configs/data_preparation.yaml")
args = args.parse_args()

# Load the configuration file
with open(args.config, "r") as file:
    config = yaml.safe_load(file)

# Create logger
os.makedirs(config["log_folder"], exist_ok=True)
log_file = os.path.join(config["log_folder"], "data_preparation.log")
# Add show on standard output as well
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)
logging.info("Starting data preparation")

# Extract the Phoenix dataset if not provided extracted folder
PHOENIX_BASE_FOLDER = None
if config["phoenix_extracted_folder"] == "":
    # Extract the dataset
    try:
        phoenix_tar_path = config["phoenix_tar_file"]
        phoenix_tar_folder = os.path.dirname(phoenix_tar_path)
        if not os.path.exists(phoenix_tar_path):
            logging.error("Phoenix dataset not found. There might be the issue with the download section")
            exit(1)
        logging.info("Extracting the dataset")
        if os.system(f"tar -xvzf {phoenix_tar_path} -C {phoenix_tar_folder} > /dev/null") != 0:
            raise Exception("Error with tar command")
        logging.info("Dataset extracted successfully")
        PHOENIX_BASE_FOLDER = os.path.join(phoenix_tar_folder, "PHOENIX-2014-T-release-v3", "PHOENIX-2014-T")
    except Exception as err:
        logging.error(f"Error while extracting the dataset: {err}")
        exit(1)
else:
    PHOENIX_BASE_FOLDER = config["phoenix_extracted_folder"]
logging.info(f"Phoenix dataset is available at {os.path.abspath(PHOENIX_BASE_FOLDER)}")

# Create folder for the text files from the dataset
logging.info("Creating the folder for new extracted files")
os.makedirs(config["target_folder"], exist_ok=True)
os.makedirs(os.path.join(config["target_folder"], "dev"), exist_ok=True)
os.makedirs(os.path.join(config["target_folder"], "train"), exist_ok=True)
os.makedirs(os.path.join(config["target_folder"], "test"), exist_ok=True)

# Extract the text files
logging.info("Extracting the text files")
for split in ["dev", "train", "test"]:
    csv_path = os.path.join(PHOENIX_BASE_FOLDER, "annotations", "manual", f"PHOENIX-2014-T.{split}.corpus.csv")
    logging.info(f"loading the file {csv_path}")
    df = pd.read_csv(csv_path, sep="|")
    looper = tqdm(df.iterrows(), total=len(df), desc=f"Extracting annotations for {split}")
    for idx, row in looper:
        try:
            if row["start"] != -1 and row["end"] != -1:
                logging.info(f"Skipping the row {idx}/{row['name']} due to start and end time is not -1")
                continue
            target_json = {
                "name": row["name"],
                "signer": row["speaker"],
                "gloss": row["orth"],
                "text": row["translation"],
            }
            target_path = os.path.join(config["target_folder"], split, f'{row["name"]}.json')
            with open(target_path, "w") as file:
                json.dump(target_json, file)
            logging.debug(f"File {target_path} created successfully")
        except Exception as err:
            logging.error(f"Error while creating the file {target_path}: {err}")
            continue
logging.info("All text files extracted successfully")

# Create a video from folder
logging.info("Creating the video files")
for split in ["dev", "train", "test"]:
    video_parent_folder = os.path.join(PHOENIX_BASE_FOLDER, "features", "fullFrame-210x260px", split)
    video_folders = os.listdir(video_parent_folder)
    logging.info(f"Extracting the videos for {split}(total: {len(video_folders)})")
    looper = tqdm(video_folders, total=len(video_folders), desc=f"Creating the video for {split}")
    for video_folder in looper:
        try:
            video_folder_path = os.path.join(video_parent_folder, video_folder)
            video_file = os.listdir(video_folder_path)[0]
            video_file = os.path.join(video_folder_path, video_file)
            video_file = "/".join(video_file.split("/")[:-1]) + "/images%04d.png"
            output_video_path = os.path.join(config["target_folder"], split, f"{video_folder}.mp4")
            logging.debug(f"Using the video files: {video_file}")
            logging.debug(f"Creating the video {split}/{output_video_path}")
            if (
                os.system(
                    f"ffmpeg -r {config['video_fps']} -i {video_file} -c:v libx264 -vf fps={config['video_fps']} -pix_fmt yuv420p {output_video_path} -y > /dev/null"
                )
                != 0
            ):
                raise Exception("Error with ffmpeg command")
            logging.debug(f"Video {split}/{output_video_path} created successfully")
        except Exception as err:
            logging.error(f"Error while creating the video {output_video_path}: {err}")
            continue
logging.info("All video files extracted successfully")

# Use BIN to deblur the video
if config["use_bin"]:
    logging.info("BIN is used for deblurring the video")
    for split in ["dev", "train", "test"]:
        mp4_folder = os.path.join(config["target_folder"], split)
        looper = tqdm(
            os.listdir(mp4_folder), total=len(os.listdir(mp4_folder)), desc=f"Deblurring the video for {split}"
        )
        for video_file in looper:
            input_vdo_path = os.path.join(mp4_folder, video_file)
            if input_vdo_path.endswith(".mp4"):
                deblur_vdo_using_BIN_main(
                    input_vdo_path=input_vdo_path,
                    output_vdo_path=input_vdo_path,
                    model_net_name=config["bin_config"]["model_net_name"],
                    num_interpolation=config["bin_config"]["num_interpolation"],
                    time_step=config["bin_config"]["time_step"],
                    model_option_yaml=config["bin_config"]["model_option_yaml"],
                )
else:
    logging.info("BIN is not used for deblurring the video")

# Create skeleton files
logging.info("Creating the skeleton files")
for split in ["dev", "train", "test"]:
    convert_vdo_to_skeleton_main(
        face_model_config=config["vdo_to_skeletons_config"]["face_model"],
        hand_model_config=config["vdo_to_skeletons_config"]["hand_model"],
        pose_model_config=config["vdo_to_skeletons_config"]["pose_model"],
        output_folder=os.path.join(config["target_folder"], split),
        landmarks_format=config["vdo_to_skeletons_config"]["landmarks_format"],
        save_format=config["vdo_to_skeletons_config"]["save_format"],
        is_return_landmarked_vdo=config["vdo_to_skeletons_config"]["is_return_landmarked_vdo"],
        max_distance_between_predicted_hand_and_approximated_hand=config["vdo_to_skeletons_config"][
            "max_distance_between_predicted_hand_and_approximated_hand"
        ],
        save_stats=config["vdo_to_skeletons_config"]["save_stats"],
        vdo_folder=os.path.join(config["target_folder"], split),
        yes=True,
    )
logging.info("All skeleton files created successfully")

# Normalization to Reference Skeleton & Min-Max Scaling
logging.info("Normalization to Reference Skeleton & Min-Max Scaling")
train_folder = os.path.join(config["target_folder"], "train")
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
)
logging.info("Normalization, Scaling successfully")

logging.info("Data preparation completed")
