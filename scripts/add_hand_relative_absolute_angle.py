import os

import numpy as np
import torch
import yaml
from tqdm import tqdm

from capstone_utils.absolute_angle_conversion import position_to_absolute_angle
from capstone_utils.relative_angle_conversion import position_to_relative_angle
from capstone_utils.skeleton_utils.progressive_trans_model import HAND_JOINT_TO_PREV_JOINT_INDEX, ROOT_JOINT

CONFIG_PATH = "./configs/add_hand_relative_absolute_angle.yaml"
with open(CONFIG_PATH, "r") as config_file:
    config = yaml.safe_load(config_file)


def convert_flatten_skeleton_to_unnormed_hand(
    flatten_skeleton: torch.Tensor, global_min: torch.Tensor, global_max: torch.Tensor, is_left_hand: bool
) -> torch.Tensor:
    output_hand = flatten_skeleton[1434:1497] if not is_left_hand else flatten_skeleton[1497:1560]
    output_hand = (output_hand + 1) / 2 * (global_max - global_min) + global_min
    output_hand = output_hand.reshape(-1, 3)
    output_hand -= output_hand[0]
    return output_hand


if __name__ == "__main__":
    if "input_files" not in config and "output_files" not in config:
        config["input_files"] = [config["input_file"]]
        config["output_files"] = [config["output_file"]]

    NUM_JOINT = 553
    global_min = np.load(config["global_min_path"])
    global_max = np.load(config["global_max_path"])

    rhand_global_min = global_min[1434:1497]
    rhand_global_max = global_max[1434:1497]
    near_r_target = 0

    lhand_global_min = global_min[1497:1560]
    lhand_global_max = global_max[1497:1560]
    near_l_target = 0

    for skel_file, output_path in zip(config["input_files"], config["output_files"]):
        with open(skel_file, "r") as infile, open(output_path, "w") as outfile:
            # for line in infile:
            # for line in tqdm(infile, desc="Processing frames", unit="video(s)"):
            looper = tqdm(infile, desc="Processing frames", unit="video(s)")
            for line in looper:
                # FRAME, JOINT*3
                all_joint = np.array(line.split(), dtype=float).reshape(-1, NUM_JOINT, 3)
                # print(all_joint.shape)
                for i in range(all_joint.shape[0]):  # loop through frame dimension
                    frame_joint = all_joint[i, :, :].flatten()  # NUM_JOINT*3, 1 frame
                    original_rhand_joints = frame_joint[1434:1497].reshape(-1, 3)  # hand joint, 3
                    original_lhand_joints = frame_joint[1497:1560].reshape(-1, 3)

                    original_rhand_joints = original_rhand_joints - original_rhand_joints[0]
                    original_lhand_joints = original_lhand_joints - original_lhand_joints[0]

                    # Right hand is missing
                    # if (np.allclose(original_rhand_joints, 0, atol=1e-5)) and i > 0:
                    if (
                        np.allclose(
                            convert_flatten_skeleton_to_unnormed_hand(
                                frame_joint, rhand_global_min, rhand_global_max, False
                            ),
                            0,
                            atol=1e-5,
                        )
                    ) and i > 0:
                        next_detected_frame = i + 1
                        next_detected_hand_joints = None
                        while next_detected_frame < all_joint.shape[0]:
                            next_frame_joint = all_joint[next_detected_frame, :, :].flatten()

                            # if not np.allclose(next_frame_joint, 0, atol=1e-5):
                            if not np.allclose(
                                convert_flatten_skeleton_to_unnormed_hand(
                                    next_frame_joint, rhand_global_min, rhand_global_max, False
                                ),
                                0,
                                atol=1e-5,
                            ):
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
                    original_rhand_joints = all_joint[i, 478:499, :]
                    original_rhand_joints = original_rhand_joints - original_rhand_joints[0]

                    # Left hand is missing
                    # if (np.allclose(original_lhand_joints, 0, atol=1e-5)) and i > 0:
                    if (
                        np.allclose(
                            convert_flatten_skeleton_to_unnormed_hand(
                                frame_joint, lhand_global_min, lhand_global_max, True
                            ),
                            0,
                            atol=1e-5,
                        )
                    ) and i > 0:
                        next_detected_frame = i + 1
                        next_detected_hand_joints = None
                        while next_detected_frame < all_joint.shape[0]:
                            next_frame_joint = all_joint[next_detected_frame, :, :].flatten()

                            # if not np.allclose(next_frame_joint, 0, atol=1e-5):
                            if not np.allclose(
                                convert_flatten_skeleton_to_unnormed_hand(
                                    next_frame_joint, lhand_global_min, lhand_global_max, True
                                ),
                                0,
                                atol=1e-5,
                            ):
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
                    original_lhand_joints = all_joint[i, 499:520, :]
                    original_lhand_joints = original_lhand_joints - original_lhand_joints[0]

                    rhand_abs_angle = position_to_absolute_angle(
                        original_rhand_joints, HAND_JOINT_TO_PREV_JOINT_INDEX, ROOT_JOINT
                    )
                    lhand_abs_angle = position_to_absolute_angle(
                        original_lhand_joints, HAND_JOINT_TO_PREV_JOINT_INDEX, ROOT_JOINT
                    )

                    rhand_rel_angle = position_to_relative_angle(
                        original_rhand_joints, HAND_JOINT_TO_PREV_JOINT_INDEX, ROOT_JOINT
                    )
                    lhand_rel_angle = position_to_relative_angle(
                        original_lhand_joints, HAND_JOINT_TO_PREV_JOINT_INDEX, ROOT_JOINT
                    )

                    # Concat original, absolute, and relative angles
                    rhand_overrep = np.concatenate(
                        (original_rhand_joints, rhand_abs_angle, rhand_rel_angle)
                    )  # hand_joint *3 representation, 3
                    lhand_overrep = np.concatenate((original_lhand_joints, lhand_abs_angle, lhand_rel_angle))
                    hand_overrep = np.concatenate((rhand_overrep, lhand_overrep)).flatten()  # hand_joint*6, 3 -> 378
                    outfile.write(" ".join(map(str, hand_overrep.flatten())))
                    outfile.write(" ")
                outfile.write("\n")

        # print("Checker")
        # with open(output_path, 'r') as outfile:
        #     for line in outfile:
        #         all_joint = np.array(line.split(), dtype=float).reshape(-1, 126*3)
        #         print(all_joint.shape)
