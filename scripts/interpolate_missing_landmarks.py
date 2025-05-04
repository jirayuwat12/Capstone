"""
This scripts read .skels then
1. convert into original scale
2. replace missing hand joints with interpolated values between nearest detected hand joints
    - if no detected hand joints, use previous frame's hand joints
3. write to .skels
"""

import numpy as np
import torch
import yaml
from tqdm import tqdm

CONFIG_PATH = "./configs/interpolate_missing_landmarks.yaml"
with open(CONFIG_PATH, "r") as config_file:
    config = yaml.safe_load(config_file)


def convert_flatten_skeleton_to_unnormed_hand(
    flatten_skeleton: torch.Tensor, global_min: torch.Tensor, global_max: torch.Tensor, is_left_hand: bool
) -> torch.Tensor | torch.Tensor:
    output_hand = flatten_skeleton[1434:1497] if not is_left_hand else flatten_skeleton[1497:1560]

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
        with open(skel_file, "r") as infile, open(output_path, "w") as outfile:
            print(f"Processing {skel_file} to {output_path}")
            # for line in tqdm(infile, desc="Processing frames", unit="video(s)"):
            looper = tqdm(infile, desc="Processing frames", unit="video(s)")
            left_hand_filled = 0
            right_hand_filled = 0
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

                        if next_detected_hand_joints is None:
                            # Use previous frame's hand joints to all the rest
                            previous_frame_joint = all_joint[i - 1, :, :].flatten()
                            previous_rhand_joints = previous_frame_joint[1434:1497]
                            for j in range(i, all_joint.shape[0]):
                                all_joint[j, 478:499, :] = previous_rhand_joints.reshape(-1, 3)
                            right_hand_filled += all_joint.shape[0] - i
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
                            right_hand_filled += next_detected_frame - i

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

                        if next_detected_hand_joints is None:
                            # Use previous frame's hand joints to all the rest
                            previous_frame_joint = all_joint[i - 1, :, :].flatten()
                            previous_lhand_joints = previous_frame_joint[1497:1560]
                            for j in range(i, all_joint.shape[0]):
                                all_joint[j, 499:520, :] = previous_lhand_joints.reshape(-1, 3)
                            left_hand_filled += all_joint.shape[0] - i
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
                            left_hand_filled += next_detected_frame - i

                    # Re-compute original_lhand_joints
                    _, normed_lhand_joints = convert_flatten_skeleton_to_unnormed_hand(
                        all_joint[i, :, :].flatten(), lhand_global_min, lhand_global_max, is_left_hand=True
                    )

                    looper.set_postfix(
                        left_hand_filled=left_hand_filled,
                        right_hand_filled=right_hand_filled,
                    )

                    # outfile.write(" ".join(map(str, scaled_hand_rel_angle.flatten())) + " ")
                # Write the modified skeleton to the output file
                all_joint = all_joint.reshape(-1, NUM_JOINT * 3)
                outfile.write(" ".join(map(str, all_joint.flatten())) + "\n")
