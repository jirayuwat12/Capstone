from typing import Sequence

import numpy as np


def position_to_relative_angle(
    positional_joints: np.ndarray, joint_to_prev_joint_index: dict[int, int], root_joint: int
) -> np.ndarray:
    """
    This function converts positional format to relative angle format.
    PS. All angle is the relative angle to the previous joint.

    :param positional_joints: the skeleton joints in positional format `(50, 3)` which is `(x, y, z)`
    :param joint_to_prev_joint_index: the mapping of joint index to previous joint index
    :param root_joint: the root joint index in the skeleton model
    :return: the skeleton joints in relative angle format `(50, 3)` which use spherical coordinates `(r, theta, phi)`
        which is the relative angle to the previous joint
            r = Euclidean distance
            theta = inclination angle (angle from the z-axis)
            phi = azimuthal angle (angle from the x-axis in the x-y plane)

    Ref: https://en.wikipedia.org/wiki/Spherical_coordinate_system
    """
    raise NotImplementedError(
        "This function is deprecated due to new skeleton model. Please implement it in the new skeleton model."
    )


def relative_angle_to_position(
    angle_joints: np.ndarray,
    skeleton_model: Sequence[tuple[int, int, int]],
    joint_to_prev_joint_index: dict[int, int],
    joint_to_child_joint_index: dict[int, Sequence[int]],
    root_joint: int,
) -> np.ndarray:
    """
    This function converts relative angle format to positional format.
    PS. All angle is the relative angle to the previous joint.

    :param angle_joints: the skeleton joints in relative angle format `(50, 3)` which use spherical coordinates `(r, theta, phi)`
        which is the relative angle to the previous joint
            r = Euclidean distance
            theta = inclination angle (angle from the z-axis)
            phi = azimuthal angle (angle from the x-axis in the x-y plane)
    :param skeleton_model: the skeleton model which is a sequence of bone tuples
    :param joint_to_prev_joint_index: the mapping of joint index to previous joint index
    :param joint_to_child_joint_index: the mapping of joint index to child joint index
    :param root_joint: the root joint index in the skeleton model
    :return: the skeleton joints in positional format `(50, 3)` which is `(x, y, z)`
    """
    raise NotImplementedError(
        "This function is deprecated due to new skeleton model. Please implement it in the new skeleton model."
    )


def convert_vdo_positiion_to_relative_angle(
    position_vdo: np.ndarray, joint_to_prev_joint_index: dict[int, int], root_joint: int
) -> np.ndarray:
    """
    This function converts positional format to relative angle format.
    PS. All angle is the relative angle to the previous joint.

    :param position_vdo: the skeleton joints in positional format `(50, 3)` which is `(x, y, z)`
    :param joint_to_prev_joint_index: the mapping of joint index to previous joint index
    :param root_joint: the root joint index in the skeleton model
    :return: the skeleton joints in relative angle format `(50, 3)` which use spherical coordinates `(r, theta, phi)`

    Ref: https://en.wikipedia.org/wiki/Spherical_coordinate_system
    """
    raise NotImplementedError(
        "This function is deprecated due to new skeleton model. Please implement it in the new skeleton model."
    )


def convert_vdo_relative_angle_to_position(
    relative_angle_vdo: np.ndarray,
    skeleton_model: Sequence[tuple[int, int, int]],
    joint_to_prev_joint_index: dict[int, int],
    joint_to_child_joint_index: dict[int, Sequence[int]],
    root_joint: int,
) -> np.ndarray:
    """
    This function converts relative angle format to positional format.
    PS. All angle is the relative angle to the previous joint.

    :param relative_angle_vdo: the skeleton joints in relative angle format `(50, 3)` which use spherical coordinates `(r, theta, phi)`
        which is the relative angle to the previous joint
    :param skeleton_model: the skeleton model which is a sequence of bone tuples
    :param joint_to_prev_joint_index: the mapping of joint index to previous joint index
    :param joint_to_child_joint_index: the mapping of joint index to child joint index
    :param root_joint: the root joint index in the skeleton model
    :return: the skeleton joints in positional format `(50, 3)` which is `(x, y, z)`
    """
    raise NotImplementedError(
        "This function is deprecated due to new skeleton model. Please implement it in the new skeleton model."
    )
