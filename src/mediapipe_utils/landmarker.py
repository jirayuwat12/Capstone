from abc import ABC, abstractmethod

import numpy as np

from mediapipe_utils.mediapipe_output_stat import MediapipeOutputStat


class Landmarker(ABC):
    @abstractmethod
    def landmark_vdo(
        self, vdo_file: str, output_stat: bool = False
    ) -> np.ndarray | tuple[np.ndarray, MediapipeOutputStat]:
        """
        This function take vdo_file as input and return the landmarks of the face in the video.
        The landmarks will return as a numpy array with shape (n_frames, n_landmarks, 3).
        """
        pass
