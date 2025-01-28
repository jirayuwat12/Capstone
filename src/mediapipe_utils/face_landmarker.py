import os
import warnings

import cv2
import mediapipe as mp
import mediapipe.python.solutions.face_mesh as face_mesh
import numpy as np

from mediapipe_utils.mediapipe_output_stat import MediapipeOutputStat

from .landmarker import Landmarker


class FaceLandmarker(Landmarker):
    def __init__(self, face_config: dict):
        self.face_mesh = face_mesh.FaceMesh(
            min_detection_confidence=face_config["min_detection_confidence"],
            min_tracking_confidence=face_config["min_tracking_confidence"],
            max_num_faces=face_config["max_num_faces"],
        )

    def landmark_vdo(
        self, vdo_file: str, output_stat: bool = False
    ) -> np.ndarray | tuple[np.ndarray, MediapipeOutputStat]:
        """
        This function will read the video file and return the face landmarks of the video.

        :param vdo_file: str: The path to the video file
        :param output_stat: bool: Whether to output the statistics of the mediapipe processing
        :return: np.ndarray: The face landmarks of the video. The landmarks will return as a numpy array with shape (n_frames, n_landmarks, 3).
        """
        mediapipe_output_stat = MediapipeOutputStat()
        # Load the video file
        cv2_vdo = cv2.VideoCapture(vdo_file)

        # Iterate through the video
        landmarks = []
        for frame_index in range(int(cv2_vdo.get(cv2.CAP_PROP_FRAME_COUNT))):
            _, frame = cv2_vdo.read()
            # Convert the frame to meidapipe image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Process the frame
            face_landmarks = self.face_mesh.process(frame_rgb)
            mediapipe_output_stat.total_processing_frames += 1

            # Check if face is detected
            if not face_landmarks.multi_face_landmarks[0].landmark:
                warnings.warn(f"No face detected in frame {frame_index+1} in {os.path.basename(vdo_file)}")
                continue

            # Extract the face landmarks
            current_frame_landmarks = []
            for landmark in face_landmarks.multi_face_landmarks[0].landmark:
                # Extract the normalized position
                normed_x, normed_y, normed_z = landmark.x, landmark.y, landmark.z
                normed_position = (normed_x, normed_y, normed_z)

                # Append the normalized position to the current frame landmarks
                current_frame_landmarks.append(normed_position)

            # Append the current frame landmarks to the landmarks list
            if len(current_frame_landmarks) == 0:
                if self.face_config["replace_not_found_method"] == "previous":
                    # Check if the replace_not_found_method is previous
                    if len(landmarks) == 0:
                        warnings.warn(f"There is no face detected in the first frame of {os.path.basename(vdo_file)}")
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
