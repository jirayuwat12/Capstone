from typing import Sequence


def compute_joint_tree(skeleton: Sequence[tuple[int, int, int]]) -> dict[int, list[int]]:
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
