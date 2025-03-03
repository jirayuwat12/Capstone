import os

import numpy as np

from collections import defaultdict


def normalized_data(input_skeleton, reference_skeleton):
    # Normalization to reference_skeleton
    NOSE_JOINT = 520
    LEFT_SHOULDER_JOINT = 531
    RIGHT_SHOULDER_JOINT = 532

    scaled_skeleton = input_skeleton.copy()
    difference_distance = reference_skeleton[0, NOSE_JOINT, :] - input_skeleton[0, NOSE_JOINT, :]
    scaled_factor = np.linalg.norm(
        reference_skeleton[0, LEFT_SHOULDER_JOINT, :] - reference_skeleton[0, RIGHT_SHOULDER_JOINT, :]
    ) / np.linalg.norm(input_skeleton[0, LEFT_SHOULDER_JOINT, :] - input_skeleton[0, RIGHT_SHOULDER_JOINT, :])

    for frame in range(input_skeleton.shape[0]):
        for joint in range(input_skeleton.shape[1]):
            scaled_skeleton[frame, joint, :] += difference_distance
            scaled_skeleton[frame, joint, :] = reference_skeleton[0, NOSE_JOINT, :] + scaled_factor * (
                scaled_skeleton[frame, joint, :] - reference_skeleton[0, NOSE_JOINT, :]
            )

    return scaled_skeleton


def standardized_data(input_skeleton, global_min, global_max, NUM_JOINT):
    scaled_skeleton = (
        2
        * (
            (input_skeleton.reshape(input_skeleton.shape[0], -1) - global_min) / (global_max - global_min + 1e-8)
        ).reshape(-1, NUM_JOINT, 3)
        - 1
    )
    return scaled_skeleton


def compute_global_stats(all_scaled_skeletons):
    all_joint_values = []
    for input_skeleton in all_scaled_skeletons:
        if len(all_joint_values) == 0:
            all_joint_values = input_skeleton
        else:
            all_joint_values = np.concatenate((all_joint_values, input_skeleton), axis=0)
    all_joint_values = all_joint_values.reshape(all_joint_values.shape[0], -1)  # (frame all vdo, joint*3)
    global_min = np.min(all_joint_values, axis=0)  # Global min of each x,y,z joint (1 Joint == 3 min values)
    global_max = np.max(all_joint_values, axis=0)

    print("Global Stats", global_min.shape, global_max.shape)
    return global_min, global_max


def norm_standardize(input_dir: str, reference_dir: str, output_file: str):
    reference_skeleton = np.load(reference_dir)
    all_scaled_skeletons_by_split = defaultdict(list)
    train_scaled_skeletons = []
    NUM_JOINT = reference_skeleton.shape[1]

    for split in ["dev", "train", "test"]:
        for skeleton_file in sorted(os.listdir(os.path.join(input_dir, split))):
            if skeleton_file.endswith(".npy"):
                input_skeleton = np.load(os.path.join(input_dir, split, skeleton_file))
                scaled_skeleton = normalized_data(input_skeleton, reference_skeleton)
                all_scaled_skeletons_by_split[split].append(scaled_skeleton)
                if split == "train":
                    train_scaled_skeletons.append(scaled_skeleton)

    # Compute global statistics
    global_min, global_max = compute_global_stats(train_scaled_skeletons)
    print(global_min.shape, global_max.shape)
    np.save(os.path.join(output_file, "preproces_data_global_min.npy"), global_min)
    np.save(os.path.join(output_file, "preproces_data_global_max.npy"), global_max)

    for split in ["dev", "train", "test"]:
        with open(os.path.join(output_file, split, f"{split}.skels"), "w") as f:
            for scaled_skeleton in all_scaled_skeletons_by_split[split]:
                standardized_skeleton = standardized_data(scaled_skeleton, global_min, global_max, NUM_JOINT)
                if (standardized_skeleton.shape[1] != NUM_JOINT) | (standardized_skeleton.shape[2] != 3):
                    # (standardized_skeleton.min() < -1) |
                    # (standardized_skeleton.max() > 1)):
                    print("Error in standardizing the skeleton")
                    exit(1)
                standardized_skeleton_str = " ".join(map(str, standardized_skeleton.flatten()))
                f.write(standardized_skeleton_str + "\n")
