import os
import warnings

import cv2
import mediapipe as mp
import mediapipe.python.solutions.pose as PoseLandmark
import numpy as np

from mediapipe_utils.mediapipe_output_stat import MediapipeOutputStat

from .landmarker import Landmarker


class PoseLandmarker(Landmarker):
    LEFT_HAND_LANDMARKS_INDICES = [15, 17, 19, 21]
    RIGHT_HAND_LANDMARKS_INDICES = [16, 18, 20, 22]

    def __init__(self, pose_config: dict):
        self.pose_landmarker = PoseLandmark.Pose(
            min_detection_confidence=pose_config["min_detection_confidence"],
            min_tracking_confidence=pose_config["min_tracking_confidence"],
        )

    def landmark_vdo(
        self, vdo_file: str, output_stat: bool = False
    ) -> np.ndarray | tuple[np.ndarray, MediapipeOutputStat]:
        """
        This function will read the video file and return the face landmarks of the video.

        :param vdo_file: str: The path to the video file
        :return: np.ndarray: The face landmarks of the video. The landmarks will return as a numpy array with shape (n_frames, n_landmarks, 3).
        """
        mediapipe_output_stat = MediapipeOutputStat()
        # Load the video file
        cv2_vdo = cv2.VideoCapture(vdo_file)

        # Iterate through the video
        landmarks = []
        for frame_index in range(int(cv2_vdo.get(cv2.CAP_PROP_FRAME_COUNT))):
            ret, frame = cv2_vdo.read()
            if ret is False:
                break
            # Convert the frame to meidapipe image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Process the frame
            pose_landmarks = self.pose_landmarker.process(frame_rgb)
            mediapipe_output_stat.total_processing_frames += 1

            # Check if face is detected
            if not pose_landmarks.pose_landmarks:
                warnings.warn(f"No pose detected in frame {frame_index+1} in {os.path.basename(vdo_file)}")
                pose_landmarks.pose_landmarks.append([])

            # Extract the face landmarks
            current_frame_landmarks = []
            for landmark in pose_landmarks.pose_landmarks.landmark:
                # Extract the normalized position
                normed_x, normed_y, normed_z = landmark.x, landmark.y, landmark.z
                normed_position = (normed_x, normed_y, normed_z)

                # Append the normalized position to the current frame landmarks
                current_frame_landmarks.append(normed_position)

            # Append the current frame landmarks to the landmarks list
            if len(current_frame_landmarks) == 0:
                if self.pose_config["replace_not_found_method"] == "previous":
                    # Check if the replace_not_found_method is previous
                    if len(landmarks) == 0:
                        warnings.warn(f"There is no pose detected in the first frame of {os.path.basename(vdo_file)}")
                        landmarks.append([])
                    else:
                        landmarks.append(landmarks[-1])
                else:
                    # Check if the replace_not_found_method is invalid
                    raise ValueError(
                        f"Invalide replace_not_found_method: {self.face_config['replace_not_found_method']}"
                    )
            else:
                # Normal case
                landmarks.append(current_frame_landmarks)
                mediapipe_output_stat.total_found_frames += 1

        # Convert the landmarks to numpy array
        for frame_index in range(len(landmarks) - 1, -1, -1):
            if len(landmarks[frame_index]) == 0:
                landmarks[frame_index] = landmarks[frame_index + 1]
        landmarks = np.array(landmarks)
        # Return the landmarks
        if output_stat:
            return landmarks, mediapipe_output_stat
        else:
            return landmarks

    def get_approx_left_hand_landmarks(self, pose_landmarks: np.ndarray) -> np.ndarray:
        """
        This function will extract the approximated left hand landmarks from the pose landmarks.

        :param pose_landmarks: np.ndarray: The pose landmarks of the video. The landmarks should be in the shape (n_frames, n_landmarks, 3).
        :return: np.ndarray: The approximated left hand landmarks. The landmarks will return as a numpy array with shape (n_frames, n_landmarks, 3).
        """
        # Get all the left hand landmarks
        approx_left_hand_landmarks = []
        for left_hand_index in self.LEFT_HAND_LANDMARKS_INDICES:
            approx_left_hand_landmarks.append(pose_landmarks[:, left_hand_index])
        # Convert the landmarks to numpy array
        approx_left_hand_landmarks = np.array(approx_left_hand_landmarks).transpose(1, 0, 2)
        # Get mean of the landmarks (n_frames, n_landmarks, 3) -> (n_frames, 3)
        approx_left_hand_landmarks = np.mean(approx_left_hand_landmarks, axis=1)

        return approx_left_hand_landmarks

    def get_approx_right_hand_landmarks(self, pose_landmarks: np.ndarray) -> np.ndarray:
        """
        This function will extract the approximated right hand landmarks from the pose landmarks.

        :param pose_landmarks: np.ndarray: The pose landmarks of the video. The landmarks should be in the shape (n_frames, n_landmarks, 3).
        :return: np.ndarray: The approximated right hand landmarks. The landmarks will return as a numpy array with shape (n_frames, n_landmarks, 3).
        """
        # Get all the right hand landmarks
        approx_right_hand_landmarks = []
        for right_hand_index in self.RIGHT_HAND_LANDMARKS_INDICES:
            approx_right_hand_landmarks.append(pose_landmarks[:, right_hand_index])
        # Convert the landmarks to numpy array
        approx_right_hand_landmarks = np.array(approx_right_hand_landmarks).transpose(1, 0, 2)
        # Get mean of the landmarks (n_frames, n_landmarks, 3) -> (n_frames, 3)
        approx_right_hand_landmarks = np.mean(approx_right_hand_landmarks, axis=1)

        return approx_right_hand_landmarks
