from abc import ABC, abstractmethod
import numpy as np

class Landmarker(ABC):
    @abstractmethod
    def landmark_vdo(self, vdo_file: str) -> np.ndarray:
        '''
        This function take vdo_file as input and return the landmarks of the face in the video. 
        The landmarks will return as a numpy array with shape (n_frames, n_landmarks, 3).
        '''
        pass
