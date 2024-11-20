from typing import Sequence, Tuple

import numpy as np

from capstone_utils.skeleton_model import SKELETON_MODEL, ROOT_JOINT, get_previous_bone, compute_joint_tree

import warnings

def position_to_absolute_angle(
    positional_joints: np.ndarray, skeleton_model: Sequence[Tuple[int, int, int]] = SKELETON_MODEL, root_joint: int = ROOT_JOINT
) -> np.ndarray:
    """
    This function converts positional format to absolute angle format

    :param positional_joints: the skeleton joints in positional format `(50, 3)` which is `(x, y, z)`
    :param skeleton_model: the skeleton model structure
    :param root_joint: the root joint index in the skeleton model
    :return: the skeleton joints in absolute angle format `(50, 3)` which use spherical coordinates `(r, theta, phi)`
        which
            r = Euclidean distance
            theta = inclination angle (angle from the z-axis)
            phi = azimuthal angle (angle from the x-axis in the x-y plane)
    
    Ref: https://en.wikipedia.org/wiki/Spherical_coordinate_system
    """
    relative_angle_joints = np.zeros((positional_joints.shape[0], 3))

    if (positional_joints[root_joint] != 0).all():
        warnings.warn(f"The root joint {root_joint} is not at the origin (0, 0, 0)")

    # Calculate the angle and length of each joint
    for first_joint, second_joint, bone_index in SKELETON_MODEL:
        # Get the position
        first_pos = positional_joints[first_joint] # [x, y, z]
        second_pos = positional_joints[second_joint] # [x, y, z]

        # Compute the length and unit vector
        vector = (first_pos - second_pos)
        length = np.sqrt(np.sum(vector ** 2))

        # Compute the spherical coordinates for this vector
        r = length
        theta = np.arccos(vector[2] / r)
        phi = np.atan2(vector[1], vector[0])

        relative_angle_joints[second_joint] = np.array([r, theta, phi])

    return relative_angle_joints

def absolute_angle_to_position(
    angle_joints: np.ndarray, skeleton_model: Sequence[Tuple[int, int, int]] = SKELETON_MODEL, root_joint: int = ROOT_JOINT
) -> np.ndarray:
    """
    This function converts absolute angle format to positional format

    :param angle_joints: the skeleton joints in absolute angle format `(50, 3)` which use spherical coordinates `(r, theta, phi)`
        which
            r = Euclidean distance
            theta = inclination angle (angle from the z-axis)
            phi = azimuthal angle (angle from the x-axis in the x-y plane)
    :param skeleton_model: the skeleton model structure
    :param root_joint: the root joint index in the skeleton model
    :return: the skeleton joints in positional format `(50, 3)` which is `(x, y, z)`

    Ref: https://en.wikipedia.org/wiki/Spherical_coordinate_system
    """
    positional_joints = np.zeros((angle_joints.shape[0], 3))

    # Compute the joint tree
    joint_tree = compute_joint_tree(skeleton_model)

    # Iterate through the joints respecting the tree structure
    queue = [root_joint]
    while queue:
        joint = queue.pop(0)
        # Add the children to the queue
        if joint in joint_tree:
            queue.extend(joint_tree[joint])
        # Skip the root joint
        if joint == root_joint:
            positional_joints[joint] = np.array([0, 0, 0])
            continue

        # Get the parent joint
        parent_joint = get_previous_bone(joint)[0]

        # Get the angle of the joint
        r, theta, phi = angle_joints[joint]

        # Compute the position of the joint
        parent_pos = positional_joints[parent_joint]
        x = parent_pos[0] + (r * np.sin(theta) * np.cos(phi))
        y = parent_pos[1] + (r * np.sin(theta) * np.sin(phi))
        z = parent_pos[2] + (r * np.cos(theta))

        positional_joints[joint] = np.array([x, y, z])

    return positional_joints


if __name__ == "__main__":
    # original_joints = np.linspace(0, 50, 150).reshape(-1, 3)
    original_joints = np.random.rand(50, 3)
    original_joints = original_joints - original_joints[0]
    print(f"Original joints")
    print(original_joints)

    abs_angle = position_to_absolute_angle(original_joints)
    # print(f"Absolute angle joints")
    # print(abs_angle)

    pos_joints = absolute_angle_to_position(abs_angle)
    print("Positional joints")
    print(pos_joints)

    print("Is the original joints and positional joints the same?")
    print(np.allclose(original_joints, pos_joints))
