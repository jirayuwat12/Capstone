from typing import Sequence, Tuple


# skeleton model
# must in format of (start, end, bone)
# bone is the index of the bone
# start and end are the index of the joints
# and the structure must be a tree
# this is the structure of the skeleton model
SKELETON_MODEL = (
    # head
    (0, 1, 0),
    # left shoulder
    (1, 2, 1),
    # left arm
    (2, 3, 2),
    # (3, 4, 3), #
    # Changed to avoid wrist, go straight to hands
    (3, 29, 3),
    # right shoulder
    (1, 5, 1),
    # right arm
    (5, 6, 2),
    # (6, 7, 3), #
    # Changed to avoid wrist, go straight to hands
    (6, 8, 3),
    # left hand - wrist
    # (7, 8, 4), #
    # left hand - palm
    (8, 9, 5),
    (8, 13, 9),
    (8, 17, 13),
    (8, 21, 17),
    (8, 25, 21),
    # left hand - 1st finger
    (9, 10, 6),
    (10, 11, 7),
    (11, 12, 8),
    # left hand - 2nd finger
    (13, 14, 10),
    (14, 15, 11),
    (15, 16, 12),
    # left hand - 3rd finger
    (17, 18, 14),
    (18, 19, 15),
    (19, 20, 16),
    # left hand - 4th finger
    (21, 22, 18),
    (22, 23, 19),
    (23, 24, 20),
    # left hand - 5th finger
    (25, 26, 22),
    (26, 27, 23),
    (27, 28, 24),
    # right hand - wrist
    # (4, 29, 4), #
    # right hand - palm
    (29, 30, 5),
    (29, 34, 9),
    (29, 38, 13),
    (29, 42, 17),
    (29, 46, 21),
    # right hand - 1st finger
    (30, 31, 6),
    (31, 32, 7),
    (32, 33, 8),
    # right hand - 2nd finger
    (34, 35, 10),
    (35, 36, 11),
    (36, 37, 12),
    # right hand - 3rd finger
    (38, 39, 14),
    (39, 40, 15),
    (40, 41, 16),
    # right hand - 4th finger
    (42, 43, 18),
    (43, 44, 19),
    (44, 45, 20),
    # right hand - 5th finger
    (46, 47, 22),
    (47, 48, 23),
    (48, 49, 24),
)
ROOT_JOINT = 0

PREV_JOINT_INDEX = dict(
    (bone[1], bone[0]) for bone in SKELETON_MODEL
)

# This is the format of the 3D data, outputted from the Inverse Kinematics model
def getSkeletalModelStructure() -> tuple[tuple[int, int, int]]:
    """This function returns the skeletal model structure"""
    return SKELETON_MODEL


# get bone colour given index
def get_bone_colour(skeleton: int, j: int) -> tuple[int, int, int]:
    """
    This function returns the colour of the bone given the index

    :param skeleton: The skeleton model structure
    :param j: The index of the bone
    :return: The colour of the bone in the format `(B, G, R)`
    """
    bone = skeleton[j, 2]

    if bone == 0:  # head
        c = (0, 153, 0)
    elif bone == 1:  # Shoulder
        c = (0, 0, 255)

    elif bone == 2 and skeleton[j, 1] == 3:  # left arm
        c = (0, 102, 204)
    elif bone == 3 and skeleton[j, 0] == 3:  # left lower arm
        c = (0, 204, 204)

    elif bone == 2 and skeleton[j, 1] == 6:  # right arm
        c = (0, 153, 0)
    elif bone == 3 and skeleton[j, 0] == 6:  # right lower arm
        c = (0, 204, 0)

    # Hands
    elif bone in [5, 6, 7, 8]:
        c = (0, 0, 255)
    elif bone in [9, 10, 11, 12]:
        c = (51, 255, 51)
    elif bone in [13, 14, 15, 16]:
        c = (255, 0, 0)
    elif bone in [17, 18, 19, 20]:
        c = (204, 153, 255)
    elif bone in [21, 22, 23, 24]:
        c = (51, 255, 255)

    return c


def get_previous_bone(joint_index: int) -> tuple[int, int]:
    """
    This function returns the previous bone (including first joint and second joint) given the joint index

    :param joint_index: The joint index (which is the first joint index of the bone)
    :return: The previous bone in the format `(first_joint, second_joint)`

    :raises ValueError: If the joint index is not found in the skeleton model
    """
    prev_joint_index = PREV_JOINT_INDEX.get(joint_index, None)
    if prev_joint_index is None:
        raise ValueError(f"Joint index {joint_index} not found in the skeleton model")
    
    return prev_joint_index, joint_index


def compute_joint_tree(skeleton: Sequence[Tuple[int, int, int]]) -> dict[int, list[int]]:
    """
    This function computes the joint tree structure from the skeleton model

    :param skeleton: The skeleton model structure
    :return: The joint tree structure in the format `{parent: [child1, child2, ...]}`
    """
    joint_tree = {}
    for parent, child, _ in skeleton:
        if parent not in joint_tree:
            joint_tree[parent] = []
        joint_tree[parent].append(child)

    return joint_tree