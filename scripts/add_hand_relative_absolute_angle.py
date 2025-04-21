import os
import yaml
import numpy as np

from capstone_utils.skeleton_utils.progressive_trans_model import (
    ROOT_JOINT,
    HAND_JOINT_TO_PREV_JOINT_INDEX,
)
from capstone_utils.absolute_angle_conversion import position_to_absolute_angle
from capstone_utils.relative_angle_conversion import position_to_relative_angle

CONFIG_PATH = "./configs/add_hand_relative_absolute_angle.py"
with open(CONFIG_PATH, "r") as config_file:
    config = yaml.safe_load(config_file)

if __name__ == "__main__":
    skel_file = config["input_file"]
    output_path = config["output_file"]
    NUM_JOINT = 553

    with open(skel_file, 'r') as infile, open(output_path, 'w') as outfile:
        for line in infile:
            all_joint = np.array(line.split(), dtype=float).reshape(NUM_JOINT, 3, -1)
            print(all_joint.shape)
            for i in range(all_joint.shape[2]):  # loop through last dimension
                frame_joint = all_joint[:, :, i].reshape(NUM_JOINT * 3, -1)
                original_rhand_joints = frame_joint[1434:1497].reshape(-1 ,3)
                original_lhand_joints = frame_joint[1497:1560].reshape(-1 ,3)

                original_rhand_joints = original_rhand_joints - original_rhand_joints[0]
                original_lhand_joints = original_lhand_joints - original_lhand_joints[0]

                rhand_abs_angle = position_to_absolute_angle(original_rhand_joints, HAND_JOINT_TO_PREV_JOINT_INDEX, ROOT_JOINT)
                lhand_abs_angle = position_to_absolute_angle(original_lhand_joints, HAND_JOINT_TO_PREV_JOINT_INDEX, ROOT_JOINT)

                rhand_rel_angle = position_to_relative_angle(original_rhand_joints, HAND_JOINT_TO_PREV_JOINT_INDEX, ROOT_JOINT)
                lhand_rel_angle = position_to_relative_angle(original_lhand_joints, HAND_JOINT_TO_PREV_JOINT_INDEX, ROOT_JOINT)

                # Concat original, absolute, and relative angles
                rhand_overrep = np.concatenate((original_rhand_joints, rhand_abs_angle, rhand_rel_angle))
                lhand_overrep = np.concatenate((original_lhand_joints, lhand_abs_angle, lhand_rel_angle))
                hand_overrep = np.concatenate((rhand_overrep, lhand_overrep)).reshape(126*3, -1)
                
                outfile.write(' '.join(map(str, hand_overrep.flatten())))
                outfile.write(" ")
            outfile.write("\n")
            

    # print("Checker")
    # with open(output_path, 'r') as outfile:
    #     for line in outfile:
    #         all_joint = np.array(line.split(), dtype=float).reshape(126, 3, -1)
    #         print(all_joint.shape)