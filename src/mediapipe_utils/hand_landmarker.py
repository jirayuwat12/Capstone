import os
import warnings

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from dataclasses import dataclass
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import RunningMode
import mediapipe.python.solutions.hands as HandLandmark 

from mediapipe_utils.mediapipe_output_stat import MediapipeOutputStat

from .landmarker import Landmarker


@dataclass
class BaseHandLandmark:
    x: float = 0
    y: float = 0
    z: float = 0

class HandLandmarker(Landmarker):
    def __init__(self, hand_config: dict):
        self.hands = HandLandmark.Hands(
            static_image_mode=False,  # Set to False for video processing
            max_num_hands=hand_config['max_num_hands'],
            min_detection_confidence=hand_config['min_detection_confidence'],
            min_tracking_confidence=hand_config['min_tracking_confidence'],
        )

    def landmark_vdo(
        self,
        vdo_file: str,
        output_stat: bool = False,
        approx_left_hand_landmarks: np.ndarray | None = None,
        approx_right_hand_landmarks: np.ndarray | None = None,
        max_distance_between_predicted_hand_and_approximated_hand: float = float("inf"),
    ) -> np.ndarray | tuple[np.ndarray, MediapipeOutputStat]:
        """
        This function will read the video file and return the face landmarks of the video.
        if approx_left_hand_landmarks and approx_right_hand_landmarks are provided, the function will only get the closest hand to the approximated landmarks.

        :param vdo_file: str: The path to the video file
        :param output_stat: bool: Whether to output the statistics of the mediapipe processing
        :param approx_left_hand_landmarks: np.ndarray: The approximated left hand landmarks from the pose landmarks
        :param approx_right_hand_landmarks: np.ndarray: The approximated right hand landmarks from the pose landmarks
        :param max_distance_between_predicted_hand_and_approximated_hand: float: The maximum distance between the predicted hand and the approximated hand. If the distance is greater than this value, the hand will be ignored.
        :return: np.ndarray: The face landmarks of the video. The landmarks will return as a numpy array with shape (n_frames, n_landmarks, 3).
        """
        mediapipe_output_stat = MediapipeOutputStat()
        # Load the video file
        cv2_vdo = cv2.VideoCapture(vdo_file)

        # Iterate through the video
        landmarks = []
        for frame_index in range(int(cv2_vdo.get(cv2.CAP_PROP_FRAME_COUNT))):
            _, frame = cv2_vdo.read()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Process the frame
            hand_landmarks = self.hands.process(frame_rgb)
            mediapipe_output_stat.total_processing_frames += 1

            # Check if face is detected
            if not hand_landmarks.multi_hand_landmarks:
                landmarks.append([[0, 0, 0] for _ in range(42)])
                continue

            # Filter only hands that are close to the approximated landmarks
            concatenated_landmarks = []
            min_distance_to_left = float("inf")
            min_distance_to_right = float("inf")
            closest_left_hand = None
            closest_right_hand = None
            for hand_index, hand_landmark in enumerate(hand_landmarks.multi_hand_landmarks):
                hand_landmark = hand_landmark.landmark
                np_hand_landmark = np.array([(landmark.x, landmark.y) for landmark in hand_landmark])
                mean_hand_landmark = np.mean(np_hand_landmark, axis=0)
                if approx_left_hand_landmarks is None and approx_right_hand_landmarks is None:
                    concatenated_landmarks.extend(hand_landmark)
                else:
                    if approx_left_hand_landmarks is not None:
                        approx_left_hand_landmark = approx_left_hand_landmarks[frame_index][:2]
                        distance_from_approx = np.linalg.norm(mean_hand_landmark - approx_left_hand_landmark)
                        if (
                            distance_from_approx < min_distance_to_left
                            and distance_from_approx < max_distance_between_predicted_hand_and_approximated_hand
                        ):
                            min_distance_to_left = distance_from_approx
                            closest_left_hand = hand_landmark
                    if approx_right_hand_landmarks is not None:
                        approx_right_hand_landmark = approx_right_hand_landmarks[frame_index][:2]
                        distance_from_approx = np.linalg.norm(mean_hand_landmark - approx_right_hand_landmark)
                        if (
                            distance_from_approx < min_distance_to_right
                            and distance_from_approx < max_distance_between_predicted_hand_and_approximated_hand
                        ):
                            min_distance_to_right = distance_from_approx
                            closest_right_hand = hand_landmark
            # Append hand
            if closest_left_hand is not None and closest_right_hand == closest_left_hand:
                if min_distance_to_left < min_distance_to_right:
                    closest_right_hand = None
                else:
                    closest_left_hand = None
            concatenated_landmarks.extend(closest_left_hand if closest_left_hand is not None else [BaseHandLandmark()] * 21)
            concatenated_landmarks.extend(closest_right_hand if closest_right_hand is not None else [BaseHandLandmark()] * 21)
            mediapipe_output_stat.total_found_frames += closest_right_hand is not None or closest_left_hand is not None

            # Extract the face landmarks
            current_frame_landmarks = []
            for landmark in concatenated_landmarks:
                # Extract the normalized position
                normed_x, normed_y, normed_z = landmark.x, landmark.y, landmark.z
                normed_position = (normed_x, normed_y, normed_z)

                # Append the normalized position to the current frame landmarks
                current_frame_landmarks.append(normed_position)

            # Append the current frame landmarks to the landmarks list
            # TODO: apply `replace_not_found_method` here
            landmarks.append(current_frame_landmarks)

        # Convert the landmarks to numpy array
        for i in range(len(landmarks)):
            print(i, len(landmarks[i]))
        landmarks = np.array(landmarks)
        # Return the landmarks
        if output_stat:
            return landmarks, mediapipe_output_stat
        else:
            return landmarks
