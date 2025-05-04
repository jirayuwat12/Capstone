import warnings
from typing import Sequence

import numpy as np

from capstone_utils.absolute_angle_conversion import absolute_angle_to_position, position_to_absolute_angle


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
    # Convert the positional format to absolute angle
    absolute_angle_joints = position_to_absolute_angle(positional_joints, joint_to_prev_joint_index, root_joint)

    # Convert the absolute angle to relative angle
    relative_angle_joints = np.zeros((positional_joints.shape[0], 3))
    for bone_index in range(len(positional_joints)):
        if bone_index == root_joint:
            relative_angle_joints[bone_index] = absolute_angle_joints[bone_index]
            continue

        # Get the parent joint
        if bone_index not in joint_to_prev_joint_index:
            warnings.warn(f"Bone index {bone_index} does not have a parent joint")
            continue
        parent_joint = joint_to_prev_joint_index[bone_index]

        # Compute the relative angle
        r = absolute_angle_joints[bone_index][0]
        theta = absolute_angle_joints[bone_index][1] - absolute_angle_joints[parent_joint][1]
        phi = absolute_angle_joints[bone_index][2] - absolute_angle_joints[parent_joint][2]

        # Store the relative angle
        relative_angle_joints[bone_index] = np.array([r, theta, phi])

    return relative_angle_joints


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
    absolute_angle_joints = np.zeros((len(angle_joints), 3))

    # Convert the relative angle to absolute angle
    joint_queue = [root_joint]
    while joint_queue:
        current_joint = joint_queue.pop(0)
        if current_joint in joint_to_prev_joint_index:
            parent_joint = joint_to_prev_joint_index[current_joint]
            absolute_angle_joints[current_joint, 1:] = (
                angle_joints[current_joint, 1:] + absolute_angle_joints[parent_joint, 1:]
            )
            absolute_angle_joints[current_joint, 0] = angle_joints[current_joint, 0]
        else:
            absolute_angle_joints[current_joint] = angle_joints[current_joint]

        if current_joint in joint_to_child_joint_index:
            joint_queue.extend(joint_to_child_joint_index[current_joint])

    # Convert the absolute angle to positional format
    positional_joints = absolute_angle_to_position(
        absolute_angle_joints, skeleton_model, joint_to_prev_joint_index, root_joint
    )

    return positional_joints


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
    # Convert the positional format to absolute angle
    is_one_dim_frame = len(position_vdo.shape) == 2
    frame_shape = position_vdo.shape[1:]
    relative_angle_vdo = []
    for frame in position_vdo:
        # Change the frame to the correct shape
        if is_one_dim_frame:
            frame = frame.reshape(-1, 3)
        # Set root joint to 0, 0, 0
        if (frame[root_joint] != 0).all():
            warnings.warn(f"The root joint {root_joint} is not at the origin (0, 0, 0) and will be set to the origin")
            frame = frame - frame[root_joint]
        relative_angle_vdo.append(position_to_relative_angle(frame, joint_to_prev_joint_index, root_joint))
    # Change the relative angle to the correct shape
    if is_one_dim_frame:
        return np.array(relative_angle_vdo).reshape(-1, *frame_shape)
    else:
        return np.array(relative_angle_vdo)


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
    # Convert the relative angle to absolute angle
    is_one_dim_frame = len(relative_angle_vdo.shape) == 2
    frame_shape = relative_angle_vdo.shape[1:]
    absolute_angle_vdo = []
    for frame in relative_angle_vdo:
        # Change the frame to the correct shape
        if is_one_dim_frame:
            frame = frame.reshape(-1, 3)
        absolute_angle_vdo.append(
            relative_angle_to_position(
                frame, skeleton_model, joint_to_prev_joint_index, joint_to_child_joint_index, root_joint
            )
        )
    # Change the absolute angle to the correct shape
    return np.array(absolute_angle_vdo).reshape(-1, *frame_shape)
