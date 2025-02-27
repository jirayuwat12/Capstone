import argparse
import os
import subprocess
import warnings

import cv2
import yaml

from BIN.demo import main as bin_main


def deblur_vdo_using_BIN_main(
    input_vdo_path: str,
    output_vdo_path: str,
    model_net_name: str,
    num_interpolation: int,
    time_step: float,
    model_option_yaml: str,
):
    # constants
    TEMP_FOLDER_NAME = "deblur_vdo_using_BIN_temp"
    VDO_NAME = os.path.basename(input_vdo_path).split(".")[0]
    ALL_CONVERTED_VDO_PATH = f"{TEMP_FOLDER_NAME}/vdo"
    CONVERTED_VDO_PATH = f"{TEMP_FOLDER_NAME}/vdo/{VDO_NAME}"
    DEBLURRED_VDO_PATH = f"{TEMP_FOLDER_NAME}/deblurred"
    IS_VDO_PATH_A_FOLDER = os.path.isdir(input_vdo_path)

    # Prepare the vdo data ----------------------------------------------
    ## Create a temporary folder
    if os.path.exists(TEMP_FOLDER_NAME):
        warnings.warn("The temporary folder already exists. It will be overwritten.")
        os.system(f"rm -r {TEMP_FOLDER_NAME}")
    os.makedirs(TEMP_FOLDER_NAME)
    os.makedirs(ALL_CONVERTED_VDO_PATH)
    os.makedirs(CONVERTED_VDO_PATH)
    os.makedirs(DEBLURRED_VDO_PATH)
    os.makedirs(os.path.dirname(output_vdo_path), exist_ok=True)
    ## Convert from .mp4 to vdo folder if needed
    if not IS_VDO_PATH_A_FOLDER:
        os.system(f"ffmpeg -i {input_vdo_path} {CONVERTED_VDO_PATH}/%05d.png")
        print(f"Converted {input_vdo_path} to {CONVERTED_VDO_PATH}")
        ## Change each file name from 00001.png to 00002.png, 00002.png to 00004.png, etc.
        for i, file_name in enumerate(sorted(os.listdir(CONVERTED_VDO_PATH))):
            os.rename(f"{CONVERTED_VDO_PATH}/{file_name}", f"{CONVERTED_VDO_PATH}/{file_name}.temp")
        for i, file_name in enumerate(sorted(os.listdir(CONVERTED_VDO_PATH))):
            to_be_renamed = f"{CONVERTED_VDO_PATH}/{file_name}"
            new_name = f"{CONVERTED_VDO_PATH}/{str((num_interpolation+1)*(i+1)-1).zfill(5)}.png"
            os.rename(to_be_renamed, new_name)
        print(f"Renamed files in {CONVERTED_VDO_PATH} to match the interpolation")
    else:
        os.system(f"cp -R {input_vdo_path} {ALL_CONVERTED_VDO_PATH}")
        print(f"Copied {input_vdo_path} to {CONVERTED_VDO_PATH}")

    # Run BIN to deblur the vdo -----------------------------------------
    ## Create args namespace
    print("Running BIN to deblur the vdo")
    args = argparse.Namespace()
    args.netName = model_net_name
    args.input_path = ALL_CONVERTED_VDO_PATH
    args.output_path = DEBLURRED_VDO_PATH
    args.time_step = time_step
    args.opt = model_option_yaml
    args.launcher = "none"
    args.gpu_id = -1
    args.local_rank = 0
    ## Run BIN
    bin_main(args)

    # Convert the deblurred vdo to .mp4 ----------------------------------
    ## Convert from vdo folder to .mp4
    original_fps = cv2.VideoCapture(input_vdo_path).get(cv2.CAP_PROP_FPS)
    after_fps = original_fps * (1 / time_step)
    os.system(
        f"ffmpeg -framerate {after_fps} -pattern_type glob -i './{DEBLURRED_VDO_PATH}/{VDO_NAME}/*.png' -c:v libx264 -pix_fmt yuv420p {output_vdo_path}"
    )
    print(f"Converted ./{DEBLURRED_VDO_PATH}/{VDO_NAME} to {output_vdo_path}")

    # Clear temporary folder -------------------------------------------
    os.system(f"rm -r {TEMP_FOLDER_NAME}")
    print(f"Removed {TEMP_FOLDER_NAME}")


if __name__ == "__main__":
    # Create a parser object
    argparser = argparse.ArgumentParser(description="Convert VDO to skeletons")
    argparser.add_argument("--config", help="Path to the config file", default="./configs/deblur_vdo_using_BIN.yaml")
    args = argparser.parse_args()

    # Load the config file
    config_path = args.config
    print(f"Loading the config file from {config_path}")
    with open(config_path, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # Run the main function
    deblur_vdo_using_BIN_main(
        input_vdo_path=config["input_vdo_path"],
        output_vdo_path=config["output_vdo_path"],
        model_net_name=config["model_net_name"],
        num_interpolation=config["num_interpolation"],
        time_step=config["time_step"],
        model_option_yaml=config["model_option_yaml"],
    )
