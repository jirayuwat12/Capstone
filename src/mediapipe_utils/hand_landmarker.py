from .landmarker import Landmarker
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import RunningMode
import numpy as np
import cv2
import warnings
import mediapipe as mp
import os


class HandLandmarker(Landmarker):
    def __init__(self, hand_config: dict):
        self.hand_config = hand_config
        self.hand_base_options = python.BaseOptions(model_asset_path=hand_config["hand_landmarker_model_path"])
        self.hand_options = vision.HandLandmarkerOptions(
            base_options=self.hand_base_options,
            num_hands=hand_config["max_num_hands"],
            min_hand_detection_confidence=hand_config["min_hand_detection_confidence"],
            min_hand_presence_confidence=hand_config["min_hand_presence_confidence"],
            min_tracking_confidence=hand_config["min_tracking_confidence"],
            running_mode=RunningMode.VIDEO,
        )


    def landmark_vdo(self, vdo_file: str) -> np.ndarray:
        """
        This function will read the video file and return the face landmarks of the video.

        :param vdo_file: str: The path to the video file
        :return: np.ndarray: The face landmarks of the video. The landmarks will return as a numpy array with shape (n_frames, n_landmarks, 3).
        """
        with vision.HandLandmarker.create_from_options(self.hand_options) as hand_detector:
            # Load the video file
            cv2_vdo = cv2.VideoCapture(vdo_file)

            # Iterate through the video
            landmarks = []
            for frame_index in range(int(cv2_vdo.get(cv2.CAP_PROP_FRAME_COUNT))):
                _, frame = cv2_vdo.read()
                # Convert the frame to meidapipe image
                frame_mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                # Process the frame
                hand_landmarks = hand_detector.detect_for_video(frame_mp_image, frame_index)

                # Check if face is detected
                if not hand_landmarks.hand_landmarks:
                    warnings.warn(f"No hand detected in frame {frame_index+1} in {os.path.basename(vdo_file)}")
                    hand_landmarks.hand_landmarks.append([])

                # Extract the face landmarks
                current_frame_landmarks = []
                concatenated_landmarks = hand_landmarks.hand_landmarks[0] + (hand_landmarks.hand_landmarks[1] if len(hand_landmarks.hand_landmarks) >= 2 else [])
                for landmark in concatenated_landmarks:
                    # Extract the normalized position
                    normed_x, normed_y, normed_z = landmark.x, landmark.y, landmark.z
                    normed_position = (normed_x, normed_y, normed_z)

                    # Append the normalized position to the current frame landmarks
                    current_frame_landmarks.append(normed_position)

                # Append the current frame landmarks to the landmarks list
                if len(current_frame_landmarks) == 0:
                    if self.hand_config['replace_not_found_method'] == 'previous':
                        # Check if the replace_not_found_method is previous
                        if len(landmarks) == 0:
                            warnings.warn(f"There is no hand detected in the first frame of {os.path.basename(vdo_file)}")
                            landmarks.append([])
                        else:
                            landmarks.append(landmarks[-1])
                    else:
                        # Check if the replace_not_found_method is invalid
                        raise ValueError(f"Invalide replace_not_found_method: {self.hand_config['replace_not_found_method']}")
                else:
                    # Normal case
                    landmarks.append(current_frame_landmarks)

            # Convert the landmarks to numpy array
            for frame_index in range(len(landmarks)-1, -1, -1):
                if len(landmarks[frame_index]) == 0:
                    landmarks[frame_index] = landmarks[frame_index+1]
                if len(landmarks[frame_index]) == 21:
                    landmarks[frame_index] = landmarks[frame_index] + [(0, 0, 0)] * 21
            landmarks = np.array(landmarks)
            # Return the landmarks
            return landmarks
