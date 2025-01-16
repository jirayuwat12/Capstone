from dataclasses import dataclass, field


@dataclass
class VDOFaceLandmarks:
    frame_width: int
    frame_height: int
    normed_position: list[tuple[float, float, float]] = field(default_factory=list)
    position: list[tuple[int, int, int]] = field(default_factory=list)

    def apped(self, normed_position: tuple[float, float, float], position: tuple[int, int, int]):
        self.normed_position.append(normed_position)
        self.position.append(position)
