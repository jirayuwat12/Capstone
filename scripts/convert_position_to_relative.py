import argparse

import torch

from capstone_utils.relative_angle_conversion import position_to_relative_angle
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
        to_save = []
        for frame_index, skeleton in enumerate(skeletons):
            skeleton = skeleton.reshape(-1, 3)
            relative_angle_skeleton = position_to_relative_angle(skeleton, JOINT_TO_PREV_JOINT_INDEX, ROOT_JOINT)
            relative_angle_skeleton = relative_angle_skeleton.reshape(-1)
            to_save.extend(relative_angle_skeleton)
            to_save.append(frame_index)

        f.write(" ".join(map(str, to_save)) + "\n")
