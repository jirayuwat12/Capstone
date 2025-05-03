"""
This scripts read .skels then
1. convert into original scale
2. replace missing hand joints with interpolated values between nearest detected hand joints
    - if no detected hand joints, use previous frame's hand joints
3. compute relative angles
4. normalize to [-1, 1]
    - write global min/max to .npy which include min/max of
      - global min/max of relative angles
      - global min/max of bone length
5. write to .skels
"""

import os

import numpy as np
import torch
import yaml
from tqdm import tqdm

from capstone_utils.relative_angle_conversion import position_to_relative_angle
from capstone_utils.skeleton_utils.progressive_trans_model import HAND_JOINT_TO_PREV_JOINT_INDEX, ROOT_JOINT

CONFIG_PATH = "./configs/convert_pos_to_rel_angle_hand.yaml"
with open(CONFIG_PATH, "r") as config_file:
    config = yaml.safe_load(config_file)


def convert_flatten_skeleton_to_unnormed_hand(
    flatten_skeleton: torch.Tensor, global_min: torch.Tensor, global_max: torch.Tensor, is_left_hand: bool
) -> torch.Tensor | torch.Tensor:
    output_hand = flatten_skeleton[1434:1497] if not is_left_hand else flatten_skeleton[1497:1560]
    # output_hand = (output_hand + 1) / 2 * (global_max - global_min) + global_min
    # output_hand = output_hand.reshape(-1, 3)
    # output_hand -= output_hand[0]
    unnormed_hand = (output_hand + 1) / 2 * (global_max - global_min) + global_min
    unnormed_hand = unnormed_hand.reshape(-1, 3)
    unnormed_hand -= unnormed_hand[0]

    normed_hand = output_hand.reshape(-1, 3)
    normed_hand -= normed_hand[0]

    return unnormed_hand, normed_hand


if __name__ == "__main__":
    if "input_files" not in config and "output_files" not in config:
        config["input_files"] = [config["input_file"]]
        config["output_files"] = [config["output_file"]]

    NUM_JOINT = 553
    NUM_HAND_JOINT = 21
    global_min = np.load(config["global_min_path"])
    global_max = np.load(config["global_max_path"])

    rhand_global_min = global_min[1434:1497]
    rhand_global_max = global_max[1434:1497]
    near_r_target = 0

    lhand_global_min = global_min[1497:1560]
    lhand_global_max = global_max[1497:1560]
    near_l_target = 0

    del global_min
    del global_max

    for skel_file, output_path in zip(config["input_files"], config["output_files"]):
        with open(skel_file, "r") as infile:
            global_min_angle = np.full((NUM_HAND_JOINT * 2, 2), np.inf)
            global_max_angle = np.full((NUM_HAND_JOINT * 2, 2), -np.inf)
            global_min_bl = np.full((NUM_HAND_JOINT * 2, 1), np.inf)  # bone length
            global_max_bl = np.full((NUM_HAND_JOINT * 2, 1), -np.inf)

            print(f"Processing {skel_file} to {output_path}")
            looper = tqdm(infile, desc="Finding global min/max", unit="video(s)")
            for line in looper:
                # FRAME, JOINT*3
                all_joint = np.array(line.split(), dtype=float).reshape(-1, NUM_JOINT, 3)

                # print(all_joint.shape)
                for i in range(all_joint.shape[0]):  # loop through frame dimension
                    frame_joint = all_joint[i, :, :].flatten()  # NUM_JOINT*3, 1 frame

                    original_rhand_joints, _ = convert_flatten_skeleton_to_unnormed_hand(
                        frame_joint, rhand_global_min, rhand_global_max, is_left_hand=False
                    )

                    original_lhand_joints, _ = convert_flatten_skeleton_to_unnormed_hand(
                        frame_joint, lhand_global_min, lhand_global_max, is_left_hand=True
                    )

                    # Right hand is missing
                    if (np.allclose(original_rhand_joints, near_r_target, atol=1e-5)) and i > 0:
                        next_detected_frame = i + 1
                        next_detected_hand_joints = None
                        while next_detected_frame < all_joint.shape[0]:
                            next_frame_joint = all_joint[next_detected_frame, :, :].flatten()
                            next_rhand_joints, _ = convert_flatten_skeleton_to_unnormed_hand(
                                next_frame_joint, rhand_global_min, rhand_global_max, is_left_hand=False
                            )

                            if not np.allclose(next_rhand_joints, near_r_target, atol=1e-5):
                                next_detected_hand_joints = all_joint[next_detected_frame, 478:499, :].flatten()
                                break

                            next_detected_frame += 1

                        looper.set_postfix(
                            {
                                "filling": f"Right hand",
                                "from": i,
                                "to": next_detected_frame,
                            }
                        )

                        if next_detected_hand_joints is None:
                            # Use previous frame's hand joints to all the rest
                            previous_frame_joint = all_joint[i - 1, :, :].flatten()
                            previous_rhand_joints = previous_frame_joint[1434:1497]
                            for j in range(i, all_joint.shape[0]):
                                all_joint[j, 478:499, :] = previous_rhand_joints.reshape(-1, 3)
                        else:
                            # Get previous frame's hand joints
                            previous_frame_joint = all_joint[i - 1, :, :].flatten()
                            previous_rhand_joints = previous_frame_joint[1434:1497]
                            # Interpolate between previous and next detected hand joints
                            for j in range(i, next_detected_frame):
                                alpha = (j - i) / (next_detected_frame - i)
                                all_joint[j, 478:499, :] = (
                                    (1 - alpha) * previous_rhand_joints + alpha * next_detected_hand_joints
                                ).reshape(-1, 3)
                    # Re-compute original_rhand_joints
                    _, normed_rhand_joints = convert_flatten_skeleton_to_unnormed_hand(
                        all_joint[i, :, :].flatten(), rhand_global_min, rhand_global_max, is_left_hand=False
                    )

                    # Left hand is missing
                    if (np.allclose(original_lhand_joints, near_l_target, atol=1e-5)) and i > 0:
                        next_detected_frame = i + 1
                        next_detected_hand_joints = None
                        while next_detected_frame < all_joint.shape[0]:
                            next_frame_joint = all_joint[next_detected_frame, :, :].flatten()
                            next_lhand_joints, _ = convert_flatten_skeleton_to_unnormed_hand(
                                next_frame_joint, lhand_global_min, lhand_global_max, is_left_hand=True
                            )

                            if not np.allclose(next_lhand_joints, near_l_target, atol=1e-5):
                                next_detected_hand_joints = all_joint[next_detected_frame, 499:520, :].flatten()
                                break

                            next_detected_frame += 1

                        looper.set_postfix(
                            {
                                "filling": f"Left hand",
                                "from": i,
                                "to": next_detected_frame,
                            }
                        )

                        if next_detected_hand_joints is None:
                            # Use previous frame's hand joints to all the rest
                            previous_frame_joint = all_joint[i - 1, :, :].flatten()
                            previous_lhand_joints = previous_frame_joint[1497:1560]
                            for j in range(i, all_joint.shape[0]):
                                all_joint[j, 499:520, :] = previous_lhand_joints.reshape(-1, 3)
                        else:
                            # Get previous frame's hand joints
                            previous_frame_joint = all_joint[i - 1, :, :].flatten()
                            previous_lhand_joints = previous_frame_joint[1497:1560]
                            # Interpolate between previous and next detected hand joints
                            for j in range(i, next_detected_frame):
                                alpha = (j - i) / (next_detected_frame - i)
                                all_joint[j, 499:520, :] = (
                                    (1 - alpha) * previous_lhand_joints + alpha * next_detected_hand_joints
                                ).reshape(-1, 3)
                    # Re-compute original_lhand_joints
                    _, normed_lhand_joints = convert_flatten_skeleton_to_unnormed_hand(
                        all_joint[i, :, :].flatten(), lhand_global_min, lhand_global_max, is_left_hand=True
                    )

                    rhand_rel_angle = position_to_relative_angle(
                        normed_rhand_joints, HAND_JOINT_TO_PREV_JOINT_INDEX, ROOT_JOINT
                    )
                    lhand_rel_angle = position_to_relative_angle(
                        normed_lhand_joints, HAND_JOINT_TO_PREV_JOINT_INDEX, ROOT_JOINT
                    )

                    hand_rel_angle = np.concatenate((rhand_rel_angle, lhand_rel_angle), axis=0)  # (42, 3)

                    global_max_angle = np.maximum(global_max_angle, hand_rel_angle[:, 1:])
                    global_min_angle = np.minimum(global_min_angle, hand_rel_angle[:, 1:])
                    global_max_bl = np.maximum(global_max_bl, hand_rel_angle[:, :1])
                    global_min_bl = np.minimum(global_min_bl, hand_rel_angle[:, :1])

            # Save global min and max for later use
            np.save(os.path.join(os.path.dirname(output_path), "global_min_rel_angle.npy"), global_min_angle)
            np.save(os.path.join(os.path.dirname(output_path), "global_max_rel_angle.npy"), global_max_angle)
            np.save(os.path.join(os.path.dirname(output_path), "global_min_bl.npy"), global_min_bl)
            np.save(os.path.join(os.path.dirname(output_path), "global_max_bl.npy"), global_max_bl)

        with open(skel_file, "r") as infile, open(output_path, "w") as outfile:
            print(f"Processing {skel_file} to {output_path}")
            # for line in tqdm(infile, desc="Processing frames", unit="video(s)"):
            looper = tqdm(infile, desc="Processing frames", unit="video(s)")
            for line in looper:
                # FRAME, JOINT*3
                all_joint = np.array(line.split(), dtype=float).reshape(-1, NUM_JOINT, 3)
                # print(all_joint.shape)
                for i in range(all_joint.shape[0]):  # loop through frame dimension
                    frame_joint = all_joint[i, :, :].flatten()  # NUM_JOINT*3, 1 frame

                    original_rhand_joints, _ = convert_flatten_skeleton_to_unnormed_hand(
                        frame_joint, rhand_global_min, rhand_global_max, is_left_hand=False
                    )

                    original_lhand_joints, _ = convert_flatten_skeleton_to_unnormed_hand(
                        frame_joint, lhand_global_min, lhand_global_max, is_left_hand=True
                    )

                    # Right hand is missing
                    if (np.allclose(original_rhand_joints, near_r_target, atol=1e-5)) and i > 0:
                        next_detected_frame = i + 1
                        next_detected_hand_joints = None
                        while next_detected_frame < all_joint.shape[0]:
                            next_frame_joint = all_joint[next_detected_frame, :, :].flatten()
                            next_rhand_joints, _ = convert_flatten_skeleton_to_unnormed_hand(
                                next_frame_joint, rhand_global_min, rhand_global_max, is_left_hand=False
                            )

                            if not np.allclose(next_rhand_joints, near_r_target, atol=1e-5):
                                next_detected_hand_joints = all_joint[next_detected_frame, 478:499, :].flatten()
                                break

                            next_detected_frame += 1

                        looper.set_postfix(
                            {
                                "filling": f"Right hand",
                                "from": i,
                                "to": next_detected_frame,
                            }
                        )

                        if next_detected_hand_joints is None:
                            # Use previous frame's hand joints to all the rest
                            previous_frame_joint = all_joint[i - 1, :, :].flatten()
                            previous_rhand_joints = previous_frame_joint[1434:1497]
                            for j in range(i, all_joint.shape[0]):
                                all_joint[j, 478:499, :] = previous_rhand_joints.reshape(-1, 3)
                        else:
                            # Get previous frame's hand joints
                            previous_frame_joint = all_joint[i - 1, :, :].flatten()
                            previous_rhand_joints = previous_frame_joint[1434:1497]
                            # Interpolate between previous and next detected hand joints
                            for j in range(i, next_detected_frame):
                                alpha = (j - i) / (next_detected_frame - i)
                                all_joint[j, 478:499, :] = (
                                    (1 - alpha) * previous_rhand_joints + alpha * next_detected_hand_joints
                                ).reshape(-1, 3)
                    # Re-compute original_rhand_joints
                    _, normed_rhand_joints = convert_flatten_skeleton_to_unnormed_hand(
                        all_joint[i, :, :].flatten(), rhand_global_min, rhand_global_max, is_left_hand=False
                    )

                    # Left hand is missing
                    if (np.allclose(original_lhand_joints, near_l_target, atol=1e-5)) and i > 0:
                        next_detected_frame = i + 1
                        next_detected_hand_joints = None
                        while next_detected_frame < all_joint.shape[0]:
                            next_frame_joint = all_joint[next_detected_frame, :, :].flatten()
                            next_lhand_joints, _ = convert_flatten_skeleton_to_unnormed_hand(
                                next_frame_joint, lhand_global_min, lhand_global_max, is_left_hand=True
                            )

                            if not np.allclose(next_lhand_joints, near_l_target, atol=1e-5):
                                next_detected_hand_joints = all_joint[next_detected_frame, 499:520, :].flatten()
                                break

                            next_detected_frame += 1

                        looper.set_postfix(
                            {
                                "filling": f"Left hand",
                                "from": i,
                                "to": next_detected_frame,
                            }
                        )

                        if next_detected_hand_joints is None:
                            # Use previous frame's hand joints to all the rest
                            previous_frame_joint = all_joint[i - 1, :, :].flatten()
                            previous_lhand_joints = previous_frame_joint[1497:1560]
                            for j in range(i, all_joint.shape[0]):
                                all_joint[j, 499:520, :] = previous_lhand_joints.reshape(-1, 3)
                        else:
                            # Get previous frame's hand joints
                            previous_frame_joint = all_joint[i - 1, :, :].flatten()
                            previous_lhand_joints = previous_frame_joint[1497:1560]
                            # Interpolate between previous and next detected hand joints
                            for j in range(i, next_detected_frame):
                                alpha = (j - i) / (next_detected_frame - i)
                                all_joint[j, 499:520, :] = (
                                    (1 - alpha) * previous_lhand_joints + alpha * next_detected_hand_joints
                                ).reshape(-1, 3)
                    # Re-compute original_lhand_joints
                    _, normed_lhand_joints = convert_flatten_skeleton_to_unnormed_hand(
                        all_joint[i, :, :].flatten(), lhand_global_min, lhand_global_max, is_left_hand=True
                    )

                    rhand_rel_angle = position_to_relative_angle(
                        normed_rhand_joints, HAND_JOINT_TO_PREV_JOINT_INDEX, ROOT_JOINT
                    )
                    lhand_rel_angle = position_to_relative_angle(
                        normed_lhand_joints, HAND_JOINT_TO_PREV_JOINT_INDEX, ROOT_JOINT
                    )

                    hand_rel_angle = np.concatenate((rhand_rel_angle, lhand_rel_angle))  # (42, 3)

                    computed_global_min = np.concatenate((global_min_bl, global_min_angle), axis=1)
                    computed_global_max = np.concatenate((global_max_bl, global_max_angle), axis=1)

                    scaled_hand_rel_angle = (
                        2
                        # * ((hand_rel_angle - global_min_angle) / (global_max_angle - global_min_angle + 1e-15))
                        * ((hand_rel_angle - computed_global_min) / (computed_global_max - computed_global_min + 1e-15))
                        - 1
                    )  # (N ,42*3)
                    # scaled_hand_rel_angle = hand_rel_angle.copy()
                    outfile.write(" ".join(map(str, scaled_hand_rel_angle.flatten())) + " ")
                outfile.write("\n")
