from typing import Sequence

import numpy as np


def position_to_absolute_angle(
    positional_joints: np.ndarray, joint_to_prev_joint_index: dict[int, int], root_joint: int
) -> np.ndarray:
    """
    This function converts positional format to absolute angle format.
    PS. All angle is compared to the normal x, y, z axis.

    :param positional_joints: the skeleton joints in positional format `(50, 3)` which is `(x, y, z)`
    :param joint_to_prev_joint_index: the mapping of joint index to previous joint index
    :param root_joint: the root joint index in the skeleton model
    :return: the skeleton joints in absolute angle format `(50, 3)` which use spherical coordinates `(r, theta, phi)`
        which
            r = Euclidean distance
            theta = inclination angle (angle from the z-axis)
            phi = azimuthal angle (angle from the x-axis in the x-y plane)

    Ref: https://en.wikipedia.org/wiki/Spherical_coordinate_system
    """
    raise NotImplementedError(
        "This function is deprecated due to new skeleton model. Please implement it in the new skeleton model."
    )


def absolute_angle_to_position(
    angle_joints: np.ndarray,
    skeleton_model: Sequence[tuple[int, int, int]],
    joint_to_prev_joint_index: dict[int, int],
    root_joint: int,
) -> np.ndarray:
    """
    This function converts absolute angle format to positional format

    :param angle_joints: the skeleton joints in absolute angle format `(50, 3)` which use spherical coordinates `(r, theta, phi)`
        which
            r = Euclidean distance,
            theta = inclination angle (angle from the z-axis),
            phi = azimuthal angle (angle from the x-axis in the x-y plane)
    :param skeleton_model: the skeleton model structure
    :param joint_to_prev_joint_index: the mapping of joint index to previous joint index
    :param root_joint: the root joint index in the skeleton model
    :return: the skeleton joints in positional format `(50, 3)` which is `(x, y, z)`

    Ref: https://en.wikipedia.org/wiki/Spherical_coordinate_system
    """
    raise NotImplementedError(
        "This function is deprecated due to new skeleton model. Please implement it in the new skeleton model."
    )


def convert_vdo_position_to_absolute_angle(
    positional_joints: np.ndarray, joint_to_prev_joint_index: dict[int, int], root_joint: int
) -> np.ndarray:
    """
    This function converts positional format to absolute angle format.
    PS. All angle is compared to the normal x, y, z axis.

    :param positional_joints: the skeleton joints in positional format `(50, 3)` which is `(x, y, z)`
    :param joint_to_prev_joint_index: the mapping of joint index to previous joint index
    :param root_joint: the root joint index in the skeleton model
    :return: the skeleton joints in absolute angle format `(50, 3)` which use spherical coordinates `(r, theta, phi)`
        which
            r = Euclidean distance
            theta = inclination angle (angle from the z-axis)
            phi = azimuthal angle (angle from the x-axis in the x-y plane)

    Ref: https://en.wikipedia.org/wiki/Spherical_coordinate_system
    """
    raise NotImplementedError(
        "This function is deprecated due to new skeleton model. Please implement it in the new skeleton model."
    )


def convert_vdo_absolute_angle_to_position(
    absolute_angle_vdo: np.ndarray,
    skeleton_model: Sequence[tuple[int, int, int]],
    joint_to_prev_joint_index: dict[int, int],
    root_joint: int,
) -> np.ndarray:
    """
    This function converts absolute angle format to positional format

    :param angle_vdo: the skeleton joints in absolute angle format `(50, 3)` which use spherical coordinates `(r, theta, phi)`
        which
            r = Euclidean distance,
            theta = inclination angle (angle from the z-axis),
            phi = azimuthal angle (angle from the x-axis in the x-y plane)
    :param skeleton_model: the skeleton model structure
    :param joint_to_prev_joint_index: the mapping of joint index to previous joint index
    :param root_joint: the root joint index in the skeleton model
    :return: the skeleton joints in positional format `(50, 3)` which is `(x, y, z)`

    Ref: https://en.wikipedia.org/wiki/Spherical_coordinate_system
    """
    raise NotImplementedError(
        "This function is deprecated due to new skeleton model. Please implement it in the new skeleton model."
    )
