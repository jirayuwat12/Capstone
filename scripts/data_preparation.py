import argparse
import json
import logging
import os

import pandas as pd
import yaml
from tqdm import tqdm

from .deblur_vdo_using_BIN import deblur_vdo_using_BIN_main

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
    level=logging.INFO,
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
        if os.system(f"tar -xvzf {phoenix_tar_path} -C {phoenix_tar_folder}") != 0:
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
    looper = tqdm(df.iterrows(), total=len(df))
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
            looper.set_description(f"Extracting {split} files")
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
    looper = tqdm(video_folders, total=len(video_folders))
    for video_folder in looper:
        try:
            video_folder_path = os.path.join(video_parent_folder, video_folder)
            video_files = os.listdir(video_folder_path)
            video_files.sort()
            video_files = [os.path.join(video_folder_path, file) for file in video_files]
            video_files = " ".join(video_files)
            output_video_path = os.path.join(config["target_folder"], split, f"{video_folder}.mp4")
            if (
                os.system(
                    f"ffmpeg -r {config['video_fps']} -i {video_files} -c:v libx264 -vf fps={config['video_fps']} -pix_fmt yuv420p {output_video_path}"
                )
                != 0
            ):
                raise Exception("Error with ffmpeg command")
            logging.debug(f"Video {split}/{output_video_path} created successfully")
            looper.set_description(f"Extracting {split} videos")
        except Exception as err:
            logging.error(f"Error while creating the video {output_video_path}: {err}")
            continue
logging.info("All video files extracted successfully")

# Use BIN to deblur the video
if config["use_bin"]:
    logging.info("BIN is used for deblurring the video")
    for split in ["dev", "train", "test"]:
        mp4_folder = os.path.join(config["target_folder"], split)
        # TODO: create a folder for deblurred videos
else:
    logging.info("BIN is not used for deblurring the video")
