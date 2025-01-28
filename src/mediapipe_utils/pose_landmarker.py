import os
import warnings

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import RunningMode

from mediapipe_utils.mediapipe_output_stat import MediapipeOutputStat

from .landmarker import Landmarker


class PoseLandmarker(Landmarker):
    LEFT_HAND_LANDMARKS_INDICES = [15, 17, 19, 21]
    RIGHT_HAND_LANDMARKS_INDICES = [16, 18, 20, 22]

    def __init__(self, pose_config: dict):
        self.pose_config = pose_config
        self.pose_base_options = python.BaseOptions(model_asset_path=pose_config["pose_landmarker_model_path"])
        self.pose_options = vision.PoseLandmarkerOptions(
            base_options=self.pose_base_options,
            output_segmentation_masks=pose_config["output_segmentation_masks"],
            num_poses=pose_config["max_num_poses"],
            min_pose_detection_confidence=pose_config["min_pose_detection_confidence"],
            min_pose_presence_confidence=pose_config["min_pose_presence_confidence"],
            min_tracking_confidence=pose_config["min_tracking_confidence"],
            running_mode=RunningMode.VIDEO,
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
        with vision.PoseLandmarker.create_from_options(self.pose_options) as pose_detector:
            # Load the video file
            cv2_vdo = cv2.VideoCapture(vdo_file)

            # Iterate through the video
            landmarks = []
            for frame_index in range(int(cv2_vdo.get(cv2.CAP_PROP_FRAME_COUNT))):
                _, frame = cv2_vdo.read()
                # Convert the frame to meidapipe image
                frame_mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                # Process the frame
                pose_landmarks = pose_detector.detect_for_video(frame_mp_image, frame_index)
                mediapipe_output_stat.total_processing_frames += 1

                # Check if face is detected
                if not pose_landmarks.pose_landmarks:
                    warnings.warn(f"No pose detected in frame {frame_index+1} in {os.path.basename(vdo_file)}")
                    pose_landmarks.pose_landmarks.append([])

                # Extract the face landmarks
                current_frame_landmarks = []
                for landmark in pose_landmarks.pose_landmarks[0]:
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
                            warnings.warn(
                                f"There is no pose detected in the first frame of {os.path.basename(vdo_file)}"
                            )
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
