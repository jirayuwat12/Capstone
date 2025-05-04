from dataclasses import dataclass


@dataclass
class MediapipeOutputStat:
    """Dataclass for storing mediapipe output statistics"""

    total_processing_frames: int = 0
    total_found_frames: int = 0

    def merge(self, other: "MediapipeOutputStat", inplace: bool = False) -> "MediapipeOutputStat":
        """Merge the current statistics with other statistics"""
        if inplace:
            self.total_processing_frames += other.total_processing_frames
            self.total_found_frames += other.total_found_frames
            return self
        else:
            return MediapipeOutputStat(
                total_processing_frames=self.total_processing_frames + other.total_processing_frames,
                total_found_frames=self.total_found_frames + other.total_found_frames,
            )

    def __str__(self) -> str:
        return f"Total processing frames: {self.total_processing_frames}, Total found frames: {self.total_found_frames}"
