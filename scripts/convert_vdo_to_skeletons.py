import argparse
import os
import time
import warnings

import cv2
import mediapipe as mp
import numpy as np
import yaml
from tqdm import tqdm

from mediapipe_utils.face_landmarker import FaceLandmarker
from mediapipe_utils.hand_landmarker import HandLandmarker
from mediapipe_utils.pose_landmarker import PoseLandmarker

# Create a parser object
argparser = argparse.ArgumentParser(description="Convert VDO to skeletons")
argparser.add_argument("--config", help="Path to the config file", default="./configs/vdo_to_skeletons.yaml")
argparser.add_argument("-y", "--yes", help="Skip the confirmation", action="store_true")
args = argparser.parse_args()

# Load the config file
config_path = args.config
with open(config_path, "r") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

# Load the face, hand, and pose landmarker
face_landmarker = FaceLandmarker(config["face_model"])
hand_landmarker = HandLandmarker(config["hand_model"])
pose_landmarker = PoseLandmarker(config["pose_model"])

# Iterate through the folder
vdo_file_list = []
if "vdo_folder" in config:
    vdo_folder = config["vdo_folder"]
    print(f"Processing folder {vdo_folder}")
    for file in os.listdir(vdo_folder):
        if file.endswith(".mp4"):
            vdo_file_list.append(os.path.join(vdo_folder, file))
else:
    vdo_file_list = [config["vdo_file"]]
print(f"Converting {len(vdo_file_list)} VDO files to landmarks")

# Create the output folder
if not os.path.exists(config["output_folder"]):
    os.makedirs(config["output_folder"])
else:
    time.sleep(0.2)
    if args.yes:
        is_remove = "y"
    else:
        is_remove = input(f"Output folder {config['output_folder']} already exists. Overwrite it? (y/n): ")
    if is_remove.lower() != "y":
        raise ValueError(f"Output folder {config['output_folder']} already exists")
    os.makedirs(config["output_folder"], exist_ok=True)

# Iterate through the VDO files
looper = tqdm(vdo_file_list)
for vdo_file in looper:
    looper.set_description(f'Processing {os.path.basename(vdo_file).split(".")[0]}')

    # Extract the face and pose landmarks
    face_landmarks, face_output_stat = face_landmarker.landmark_vdo(vdo_file, output_stat=True)
    pose_landmarks, pose_output_stat = pose_landmarker.landmark_vdo(vdo_file, output_stat=True)
    # Get left hand approximation from pose landmarks
    approx_left_hand_landmarks = pose_landmarker.get_approx_left_hand_landmarks(pose_landmarks)
    approx_right_hand_landmarks = pose_landmarker.get_approx_right_hand_landmarks(pose_landmarks)
    hand_landmarks, hand_output_stat = hand_landmarker.landmark_vdo(
        vdo_file,
        output_stat=True,
        approx_left_hand_landmarks=approx_left_hand_landmarks,
        approx_right_hand_landmarks=approx_right_hand_landmarks,
        max_distance_between_predicted_hand_and_approximated_hand=config[
            "max_distance_between_predicted_hand_and_approximated_hand"
        ],
    )

    # Save the landmarks
    all_landmarks = np.concatenate([face_landmarks, hand_landmarks, pose_landmarks], axis=1)
    save_path = os.path.join(config["output_folder"], os.path.basename(vdo_file).replace(".mp4", ".npy"))
    to_save = None
    if config["landmarks_format"] == "normalized":
        to_save = all_landmarks
    elif config["landmarks_format"] == "pixel":
        original_vdo = cv2.VideoCapture(vdo_file)
        width = int(original_vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(original_vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
        original_vdo.release()
        all_landmarks[:, :, 0] *= width
        all_landmarks[:, :, 1] *= height
        all_landmarks[:, :, 2] *= width
        to_save = all_landmarks
    else:
        raise ValueError(f"Invalid landmarks format {config['landmarks_format']}")

    if config["save_format"] == "npy":
        print(to_save.shape)
        np.save(save_path, to_save)
    elif config["save_format"] == "txt":
        frame_length, landmark_amount, dimension = to_save.shape
        to_save = to_save.reshape(frame_length, landmark_amount * dimension)
        print(to_save.shape)
        np.savetxt(save_path, to_save)
    else:
        raise ValueError(f"Invalid save format {config['save_format']}")

    # Save the landmarks video
    if config["is_return_landmarked_vdo"]:
        original_vdo = cv2.VideoCapture(vdo_file)
        fps = original_vdo.get(cv2.CAP_PROP_FPS)
        width = int(original_vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(original_vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"MP4V")
        output_vdo = cv2.VideoWriter(
            os.path.join(config["output_folder"], os.path.basename(vdo_file)), fourcc, fps, (width * 3, height)
        )

        for frame_index in range(int(original_vdo.get(cv2.CAP_PROP_FRAME_COUNT))):
            _, frame = original_vdo.read()
            original_frame = cv2.resize(frame, (width, height))
            overwrite_frame = original_frame.copy()
            white_frame = np.ones_like(original_frame) * 255
            canvas = [overwrite_frame, white_frame]

            # Draw the face landmarks
            for landmark in face_landmarks[frame_index]:
                x, y, z = landmark
                x, y = int(x * width), int(y * height)
                for i, c in enumerate(canvas):
                    cv2.circle(c, (x, y), 0, (0, 0, 255), -1)

            # Draw the hand landmarks
            for landmark in hand_landmarks[frame_index, :21]:
                x, y, z = landmark
                x, y = int(x * width), int(y * height)
                for i, c in enumerate(canvas):
                    cv2.circle(c, (x, y), 1, (150, 150, 0), -1)
            for landmark in hand_landmarks[frame_index, 21:]:
                x, y, z = landmark
                x, y = int(x * width), int(y * height)
                for i, c in enumerate(canvas):
                    cv2.circle(c, (x, y), 1, (0, 180, 180), -1)

            # Draw the pose landmarks
            for landmark in pose_landmarks[frame_index]:
                x, y, z = landmark
                x, y = int(x * width), int(y * height)
                for i, c in enumerate(canvas):
                    cv2.circle(c, (x, y), 2, (255, 0, 0), -1)

            # Concatenate the frames
            canvas = [canvas[0], original_frame, canvas[1]]
            frame = np.concatenate(canvas, axis=1)
            output_vdo.write(frame)

        # Release the video
        output_vdo.release()
        original_vdo.release()

    # Print the statistics
    print("Face landmark statistics:")
    print("\t", face_output_stat)
    print("Hand landmark statistics:")
    print("\t", hand_output_stat)
    print("Pose landmark statistics:")
    print("\t", pose_output_stat)
    if config["save_stats"]:
        with open(
            os.path.join(config["output_folder"], os.path.basename(vdo_file).replace(".mp4", ".txt")), "w"
        ) as file:
            file.write(f"Face landmark statistics:\n\t{face_output_stat}\n")
            file.write(f"Hand landmark statistics:\n\t{hand_output_stat}\n")
            file.write(f"Pose landmark statistics:\n\t{pose_output_stat}\n")
