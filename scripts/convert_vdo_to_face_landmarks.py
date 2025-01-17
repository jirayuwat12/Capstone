import argparse
import os
import warnings

import cv2
import mediapipe as mp
import numpy as np
import yaml
from capstone_utils.dataclasses import VDOFaceLandmarks
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import RunningMode
from tqdm import tqdm

# Create a parser object
argparser = argparse.ArgumentParser(description="Convert VDO to face landmarks")
argparser.add_argument("--config", help="Path to the config file", default="./configs/vdo_to_face_landmarks.yaml")
args = argparser.parse_args()

# Load the config file
config_path = args.config
with open(config_path, "r") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

# Load the model
base_options = python.BaseOptions(model_asset_path=config["face_landmarker_model_path"])
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    min_face_detection_confidence=0.0,
    min_face_presence_confidence=0.0,
    min_tracking_confidence=0.0,
    num_faces=1,
    running_mode=RunningMode.VIDEO,
)
detector = vision.FaceLandmarker.create_from_options(options)
warnings.warn("Only accepts one face per frame")

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
looper = tqdm(vdo_file_list)
for vdo_file in looper:
    looper.set_description(f'Processing {os.path.basename(vdo_file).split(".")[0]}')

    # Load the VDO file
    cv2_vdo = cv2.VideoCapture(vdo_file)

    # Create a VDOFaceLandmarks object
    vdo_face_landmarks = VDOFaceLandmarks(
        frame_width := int(cv2_vdo.get(cv2.CAP_PROP_FRAME_WIDTH)),
        frame_height := int(cv2_vdo.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )

    # Create output frames and positions
    output_frames = []
    output_vectors = []

    # Iterate through the frames
    for frame_id in range(total_frames := int(cv2_vdo.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cv2_vdo.read()
        # Convert the frame to meidapipe image
        frame_mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        # Process the frame
        face_landmarks = detector.detect_for_video(frame_mp_image, frame_id)

        if config["is_return_landmarked_vdo"]:
            original_framed = frame.copy()
            over_drawed_frame = frame.copy()
            drawed_frame = np.ones_like(frame) * 255

        # Check if face is detected
        if not face_landmarks.face_landmarks:
            warnings.warn(f"No face detected in frame {frame_id+1} in {os.path.basename(vdo_file)}")
            face_landmarks.face_landmarks.append([])

        # Extract the face landmarks
        all_landmarks = []
        for landmark in face_landmarks.face_landmarks[0]:
            # Extract the normalized position
            normed_x, normed_y, normed_z = landmark.x, landmark.y, landmark.z
            normed_position = (normed_x, normed_y, normed_z)
            # Extract the position from the normalized position
            x, y, z = normed_x * frame_width, normed_y * frame_height, normed_z * frame_width
            position = (x, y, z)
            # Append the face landmarks
            vdo_face_landmarks.apped(normed_position, position)

            # Draw the face landmarks
            if config["is_return_landmarked_vdo"]:
                # Draw the face landmarks
                cv2.circle(drawed_frame, (int(x), int(y)), 0, (0, 255, 0), -1)
                cv2.circle(over_drawed_frame, (int(x), int(y)), 0, (0, 255, 0), -1)

            # Append the face landmarks
            if config["landmarks_format"] == "normalized":
                all_landmarks.append(normed_position)
            elif config["landmarks_format"] == "position":
                all_landmarks.append(position)
            else:
                raise ValueError(f"Invalid landmarks format {config['landmarks_format']}")

        # Append the frame to the output frames
        if config["is_return_landmarked_vdo"]:
            output_frames.append((over_drawed_frame, original_framed, drawed_frame))

        # Append the face landmarks to the output vectors if face is detected
        if len(all_landmarks):
            output_vectors.append(all_landmarks)

        # Update the progress bar
        looper.set_postfix(
            {
                "Frame": f"{frame_id+1}/{total_frames}",
            }
        )

    output_folder = config["output_folder"]
    os.makedirs(output_folder, exist_ok=True)

    # Save the ouptut frames
    if config["is_return_landmarked_vdo"]:
        # Create the output folder and name
        output_file = os.path.join(output_folder, os.path.basename(vdo_file))
        # Save the output frames
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        output_vdo = cv2.VideoWriter(output_file, fourcc, 30, (frame_width * 3, frame_height))
        for original_framed, over_drawed_frame, drawed_frame in output_frames:
            output_frame = np.hstack([original_framed, over_drawed_frame, drawed_frame])
            output_vdo.write(output_frame)
        output_vdo.release()

    # Save the output vectors
    output_vectors = np.array(output_vectors)
    output_file = os.path.join(output_folder, f"{os.path.basename(vdo_file).split('.')[0]}.npy")
    if len(output_vectors.shape) != 3:
        warnings.warn(f"Invalid output shape {output_vectors.shape}")
    elif output_vectors.shape[0] != total_frames:
        warnings.warn(f"Output frames {output_vectors.shape[0]} does not match total frames {total_frames}")
    else:
        np.save(output_file, output_vectors)  # Ouput shape: (total_frames, num_landmarks, 3)
