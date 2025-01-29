import argparse
import gzip
import json
import os
import pickle

import yaml

# Parse the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="./configs/get_output_text_from_name.yaml")

args = parser.parse_args()

# Load the configuration file
with open(args.config, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# Load the annotations
with open(config["annotation_file"], "rb") as f:
    annotations = pickle.load(gzip.open(f))

# Get the output text from the name
for name in config["target_file_names"]:
    for annotation in annotations:
        if annotation["name"].split("/")[-1] == name:
            with open(os.path.join(config["output_folder"], name + ".json"), "w") as f:
                json.dump(annotation, f, indent=4)
                print(f"Saved annotation for {name} to {config['output_folder']}")
            break
    else:
        print(f"Could not find annotation for {name}")
