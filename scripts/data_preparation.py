import argparse
import os
import yaml
import requests
import pandas as pd
import logging
import json
from tqdm import tqdm

# Create a parser object
args = argparse.ArgumentParser()
args.add_argument('--config', default='./configs/data_preparation.yaml')
args = args.parse_args()

# Load the configuration file
with open(args.config, 'r') as file:
    config = yaml.safe_load(file)

# Create logger
os.makedirs(config['log_folder'], exist_ok=True)
log_file = os.path.join(config['log_folder'], 'data_preparation.log')
# Add show on standard output as well
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[logging.FileHandler(log_file), logging.StreamHandler()])
logging.info('Starting data preparation')

# Download and extract the Phoenix dataset if not provided extracted folder
PHOENIX_BASE_FOLDER = None
if config['phoenix_extracted_folder'] == '':
    logging.info('Downloading and extracting the Phoenix dataset due to NOT PROVIDED extracted folder')
    # Download the dataset from the URL
    # TODO: This need at least 35 Gb of RAM to load file

    # Extract the dataset
    phoenix_tar_path = os.path.join(config['phoenix_tar_folder'], 'phoenix-2014-T.v3.tar.gz')
    if not os.path.exists(phoenix_tar_path):
        logging.error('Phoenix dataset not found. There might be the issue with the download section')
        exit(1)
    logging.info('Extracting the dataset')
    os.system(f'tar -xvzf {phoenix_tar_path} -C {config["phoenix_tar_folder"]}')
    logging.info('Dataset extracted successfully')
    PHOENIX_BASE_FOLDER = os.path.join(config['phoenix_tar_folder'], 'PHOENIX-2014-T-release-v3', 'PHOENIX-2014-T')
else:
    PHOENIX_BASE_FOLDER = config['phoenix_extracted_folder']
logging.info(f'Phoenix dataset is available at {os.path.abspath(PHOENIX_BASE_FOLDER)}')

# Extract the text files from the dataset
logging.info('Creating the folder for new extracted files')
os.makedirs(config['target_folder'], exist_ok=True)
os.makedirs(os.path.join(config['target_folder'], 'dev'), exist_ok=True)
os.makedirs(os.path.join(config['target_folder'], 'train'), exist_ok=True)
os.makedirs(os.path.join(config['target_folder'], 'test'), exist_ok=True)

# Extract the text files
logging.info('Extracting the text files')
for split in ['dev', 'train', 'test']:
    csv_path = os.path.join(PHOENIX_BASE_FOLDER, 'annotations', 'manual', f'PHOENIX-2014-T.{split}.corpus.csv')
    logging.info(f'loading the file {csv_path}')
    df = pd.read_csv(csv_path, sep='|')
    looper = tqdm(df.iterrows(), total=len(df))    
    for idx, row in looper:
        if row['start'] != -1 and row['end'] != -1:
            logging.info(f"Skipping the row {idx}/{row['name']} due to start and end time is not -1")
            continue
        target_json = {
            "name": row['name'],
            "signer": row['speaker'],
            "gloss": row['orth'],
            "text": row['translation'],
        }
        target_path = os.path.join(config['target_folder'], split, f'{row["name"]}.json')
        with open(target_path, 'w') as file:
            json.dump(target_json, file)
        logging.debug(f'File {target_path} created successfully')
        looper.set_description(f'Extracting {split} files')
logging.info('All text files extracted successfully')

# Create a video from folder
logging.info('Creating the video files')
for split in ['dev', 'train', 'test']:
    video_parent_folder = os.path.join(PHOENIX_BASE_FOLDER, 'features', 'fullFrame-210x260px', split)
    video_folders = os.listdir(video_parent_folder)
    logging.info(f'Extracting the videos for {split}(total: {len(video_folders)})')
    looper = tqdm(video_folders, total=len(video_folders))
    for video_folder in looper:
        video_folder_path = os.path.join(video_parent_folder, video_folder)
        video_files = os.listdir(video_folder_path)
        # TODO: Use ffmpeg to create the video