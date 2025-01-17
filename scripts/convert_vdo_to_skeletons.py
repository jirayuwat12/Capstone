import argparse
import os
import warnings

import cv2
import mediapipe as mp
import numpy as np
import yaml
from mediapipe_utils.face_landmarker import FaceLandmarker
from mediapipe_utils.hand_landmarker import HandLandmarker
from mediapipe_utils.pose_landmarker import PoseLandmarker
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import RunningMode
from tqdm import tqdm

# Create a parser object
argparser = argparse.ArgumentParser(description="Convert VDO to skeletons")
argparser.add_argument("--config", help="Path to the config file", default="./configs/vdo_to_skeletons.yaml")
args = argparser.parse_args()

# Load the config file
config_path = args.config
with open(config_path, "r") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

# Load the face model
face_config = config["face_model"]
face_landmarker = FaceLandmarker(face_config)

# Load the hand model
hand_config = config["hand_model"]
hand_landmarker = HandLandmarker(hand_config)

# Load the pose model
pose_config = config["pose_model"]
pose_landmarker = PoseLandmarker(pose_config)

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
print(f"Converting {len(vdo_file_list)} VDO files to face landmarks")

# Iterate through the VDO files
os.makedirs(config['output_folder'], exist_ok=True)
looper = tqdm(vdo_file_list)
for vdo_file in looper:
    looper.set_description(f'Processing {os.path.basename(vdo_file).split(".")[0]}')

    # Extract the landmarks
    face_landmarks = face_landmarker.landmark_vdo(vdo_file)
    hand_landmarks = hand_landmarker.landmark_vdo(vdo_file)
    pose_landmarks = pose_landmarker.landmark_vdo(vdo_file)

    # Save the landmarks
    all_landmarks = np.concatenate([face_landmarks, hand_landmarks, pose_landmarks], axis=1)
    save_path = os.path.join(config['output_folder'], os.path.basename(vdo_file).replace('.mp4', '.npy'))
    to_save = None
    if config['landmarks_format'] == 'normalized':
        to_save = all_landmarks
    elif config['landmarks_format'] == 'pixel':
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
    
    if config['save_format'] == 'npy':
        print(to_save.shape)
        np.save(save_path, to_save)
    elif config['save_format'] == 'txt':
        frame_length, landmark_amount, dimension = to_save.shape
        to_save = to_save.reshape(frame_length, landmark_amount*dimension)
        print(to_save.shape)
        np.savetxt(save_path, to_save)
    else:
        raise ValueError(f"Invalid save format {config['save_format']}")

    # Save the landmarks video
    if config['is_return_landmarked_vdo']:
        original_vdo = cv2.VideoCapture(vdo_file)
        fps = original_vdo.get(cv2.CAP_PROP_FPS)
        width = int(original_vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(original_vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        output_vdo = cv2.VideoWriter(os.path.join(config['output_folder'], os.path.basename(vdo_file)), fourcc, fps, (width*3, height))

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
            for landmark in hand_landmarks[frame_index]:
                x, y, z = landmark
                x, y = int(x * width), int(y * height)
                for i, c in enumerate(canvas):
                    cv2.circle(c, (x, y), 1, (0, 255, 0), -1)

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
