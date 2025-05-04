import warnings
from collections import namedtuple
from dataclasses import dataclass

import torch

list_range_tuple = namedtuple("list_range_tuple", ["start", "end"])


# Their are 1659 joints in total <- 553 * 3
# Face has 1434 joints <- 478 * 3
# Hand has 126 joints <- 42 * 3
# Body has 99 joints <- 33 * 3
JOINT_SIZE = 1659
ALL_RANGE_IN_FLATTENED = list_range_tuple(start=0, end=1659)
FACE_RANGE_IN_FLATTENED = list_range_tuple(start=0, end=1434)
BODY_RANGE_IN_FLATTENED = list_range_tuple(start=1434, end=1659)
HAND_RANGE_IN_FLATTENED = list_range_tuple(start=1434, end=1560)
CORE_RANGE_IN_FLATTENED = list_range_tuple(start=1560, end=1659)
REL_ANG_HAND_RANGE_IN_FLATTENED = [
    list_range_tuple(start=63 * 2, end=63 * 3),  # right hand
    list_range_tuple(start=63 * 5, end=63 * 6),  # left hand
]

RANGE_BY_DATA_SPEC = {
    "all": ALL_RANGE_IN_FLATTENED,
    "face": FACE_RANGE_IN_FLATTENED,
    "body": BODY_RANGE_IN_FLATTENED,
    "core": CORE_RANGE_IN_FLATTENED,
    "hand": HAND_RANGE_IN_FLATTENED,
    "rel_hand": REL_ANG_HAND_RANGE_IN_FLATTENED,
}

SUPPORT_DATA_SPECES = list(RANGE_BY_DATA_SPEC.keys())


@dataclass
class Skeleton:
    flatten_data: torch.Tensor
    joint_size: int = JOINT_SIZE

    def __post_init__(self):
        self.frame_size = self.flatten_data.shape[0]
        if self.joint_size != JOINT_SIZE:
            warnings.warn(
                f"joint_size is {self.joint_size}, but the default joint size is {JOINT_SIZE}. This may cause unexpected behavior."
            )
        if self.flatten_data.shape[1] != self.joint_size:
            raise ValueError(
                f"flatten_data must have shape (frame_size, {self.joint_size}), but got {self.flatten_data.shape}"
            )

    def get_by_data_spec(self, data_spec: str) -> torch.Tensor:
        """
        Get the data by data spec

        :param data_spec: The data spec to get
        :type data_spec: str
        :return: The data
        :rtype: torch.Tensor
        """
        if data_spec not in SUPPORT_DATA_SPECES:
            raise ValueError(f"data_spec must be one of {SUPPORT_DATA_SPECES}, but got {data_spec}")
        # TODO: Refactor this to use the range tuple, generalize the code
        if data_spec == "rel_hand":
            # Get the relative angle of the hand
            return torch.cat(
                [
                    self.flatten_data[
                        :,
                        REL_ANG_HAND_RANGE_IN_FLATTENED[i].start : REL_ANG_HAND_RANGE_IN_FLATTENED[i].end,
                    ] for i in range(len(REL_ANG_HAND_RANGE_IN_FLATTENED))
                ],
                dim=1,
            )
        return self.flatten_data[:, RANGE_BY_DATA_SPEC[data_spec].start : RANGE_BY_DATA_SPEC[data_spec].end]
