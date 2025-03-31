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


class FaceLandmarker(Landmarker):
    def __init__(self, face_config: dict):
        self.face_config = face_config
        self.face_base_options = python.BaseOptions(model_asset_path=face_config["face_landmarker_model_path"])
        self.face_options = vision.FaceLandmarkerOptions(
            base_options=self.face_base_options,
            output_face_blendshapes=face_config["output_face_blendshapes"],
            output_facial_transformation_matrixes=False,
            min_face_detection_confidence=face_config["min_face_detection_confidence"],
            min_face_presence_confidence=face_config["min_face_presence_confidence"],
            min_tracking_confidence=face_config["min_tracking_confidence"],
            num_faces=face_config["max_num_faces"],
            running_mode=RunningMode.VIDEO,
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
        media_pipe_output_stat = MediapipeOutputStat()
        with vision.FaceLandmarker.create_from_options(self.face_options) as face_detector:
            # Load the video file
            cv2_vdo = cv2.VideoCapture(vdo_file)

            # Iterate through the video
            landmarks = []
            for frame_index in range(int(cv2_vdo.get(cv2.CAP_PROP_FRAME_COUNT))):
                ret , frame = cv2_vdo.read()
                # print(frame)
                if ret is False:
                    break
                # Convert the frame to meidapipe image
                frame_mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                # Process the frame
                face_landmarks = face_detector.detect_for_video(frame_mp_image, frame_index)
                media_pipe_output_stat.total_processing_frames += 1

                # Check if face is detected
                if not face_landmarks.face_landmarks:
                    warnings.warn(f"No face detected in frame {frame_index+1} in {os.path.basename(vdo_file)}")
                    face_landmarks.face_landmarks.append([])

                # Extract the face landmarks
                current_frame_landmarks = []
                for landmark in face_landmarks.face_landmarks[0]:
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
                            warnings.warn(
                                f"There is no face detected in the first frame of {os.path.basename(vdo_file)}"
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
                    media_pipe_output_stat.total_found_frames += 1

            # Convert the landmarks to numpy array
            for frame_index in range(len(landmarks) - 1, -1, -1):
                if len(landmarks[frame_index]) == 0:
                    landmarks[frame_index] = landmarks[frame_index + 1]
            landmarks = np.array(landmarks)
            # Return the landmarks
            if output_stat:
                return landmarks, media_pipe_output_stat
            else:
                return landmarks
