import argparse

import numpy as np
import torch
from capstone_utils.absolute_angle_conversion import convert_vdo_position_to_absolute_angle
from capstone_utils.skeleton_utils.progressive_trans_model import JOINT_TO_PREV_JOINT_INDEX, ROOT_JOINT

# Parse the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--pos-skeleton", type=str, required=True, help="The path to the positional skeleton file")
parser.add_argument("--output", type=str, required=True, help="The path to the output relative angle skeleton file")
parser.add_argument("--joint-sizes", type=int, default=150, help="The number of joints in the skeleton")
args = parser.parse_args()

# Load the positional skeleton
positional_skeleton = []
with open(args.pos_skeleton, "r") as f:
    data = f.readlines()
    data = [line.strip().split(" ") for line in data]
    data = [[float(val) for val in line] for line in data]
    data = [torch.tensor(line).reshape(-1, args.joint_sizes + 1)[:, :-1] for line in data]
    for line in data:
        positional_skeleton.append(line.numpy())

# Convert the positional skeleton to a relative angle skeleton and save it
with open(args.output, "w") as f:
    for skeletons in positional_skeleton:
        to_save = convert_vdo_position_to_absolute_angle(skeletons, JOINT_TO_PREV_JOINT_INDEX, ROOT_JOINT)
        to_save = np.hstack([to_save, np.arange(to_save.shape[0]).reshape(-1, 1)])
        to_save = to_save.flatten()
        f.write(" ".join(map(str, to_save)) + "\n")
