################################################################# Dear Tata
# This code based on Build version but you only need to save
# 2 angles per joint which means there are 21*2*2 = 84 data per frame
# Follow next TODO, and don't make Out-Of-Memory error
###########################################################################
import yaml
import numpy as np
from tqdm import tqdm

from capstone_utils.skeleton_utils.progressive_trans_model import (
    ROOT_JOINT,
    HAND_JOINT_TO_PREV_JOINT_INDEX,
)
from capstone_utils.relative_angle_conversion import position_to_relative_angle

CONFIG_PATH = "./configs/convert_pos_to_rel_angle_hand.yaml"
with open(CONFIG_PATH, "r") as config_file:
    config = yaml.safe_load(config_file)

if __name__ == "__main__":
    if "input_files" not in config and "output_files" not in config:
        config["input_files"] = [config["input_file"]]
        config["output_files"] = [config["output_file"]]

    NUM_JOINT = 553
    NUM_HAND_JOINT = 21
    global_min = np.load("/mnt/disks/general_backup/cropped_output2/preproces_data_global_min.npy")
    global_max = np.load("/mnt/disks/general_backup/cropped_output2/preproces_data_global_max.npy")


    for skel_file, output_path in zip(config["input_files"], config["output_files"]):
        with open(skel_file, 'r') as infile, open(output_path, 'w') as outfile:
            # for line in infile:
            
            global_min_angle = np.full((NUM_HAND_JOINT*2,2) , np.inf)
            global_max_angle = np.full((NUM_HAND_JOINT*2,2) , -np.inf)
            print(f"Processing {skel_file} to {output_path}")
            for line in tqdm(infile, desc="Processing frames", unit="video(s)"):
                # FRAME, JOINT*3
                all_joint = np.array(line.split(), dtype=float).reshape(-1, NUM_JOINT, 3)
               
                # print(all_joint.shape)
                for i in range(all_joint.shape[0]):  # loop through frame dimension
                    frame_joint = all_joint[i, :, :].flatten() # NUM_JOINT*3, 1 frame
                    original_rhand_joints = frame_joint[1434:1497]
                    original_lhand_joints = frame_joint[1497:1560]

                    # TODO: Please, unnormalize the position first
                    # unnorm_skels = (original_skels + 1) * (global_max - global_min) / 2 + global_min
                    # e.g. global min file -> /mnt/disks/general_backup/cropped_output2/preproces_data_global_min.npy
                    # Preprocess: Norm > Scale
                    # Postprocess: Unscale 
                    rhand_global_min = global_min[1434:1497]
                    rhand_global_max = global_max[1434:1497]

                    lhand_global_min = global_min[1497:1560]
                    lhand_global_max = global_max[1497:1560]

                    original_rhand_joints = (original_rhand_joints + 1)/2*(rhand_global_max - rhand_global_min) + rhand_global_min
                    original_lhand_joints = (original_lhand_joints + 1)/2*(lhand_global_max - lhand_global_min) + lhand_global_min

                    original_rhand_joints = (original_rhand_joints - original_rhand_joints[0]).reshape(-1,3)
                    original_lhand_joints = (original_lhand_joints - original_lhand_joints[0]).reshape(-1,3)

                    rhand_rel_angle = position_to_relative_angle(original_rhand_joints, HAND_JOINT_TO_PREV_JOINT_INDEX, ROOT_JOINT)
                    lhand_rel_angle = position_to_relative_angle(original_lhand_joints, HAND_JOINT_TO_PREV_JOINT_INDEX, ROOT_JOINT)
                    
                    hand_rel_angle = np.concatenate((rhand_rel_angle, lhand_rel_angle), axis=0) #(42, 2)
                    # original_hand_joints = np.concatenate((original_rhand_joints, original_lhand_joints), axis=1)

                    # TODO: Write joints frame by frame
                    global_max_angle = np.maximum(global_max_angle, np.max(hand_rel_angle, axis=0))
                    global_min_angle = np.minimum(global_min_angle, np.min(hand_rel_angle, axis=0))

                    
            # TODO: Normalize the angle (like global_max and global_min but for both angles)
            # Normalized angle must be between 0 and 1 (I will use sigmoid function)
            # also save the angle min and max for later use (please, save it in the same folder as the skel file)
            # you need to do 2 loops to normalize the angle
            #   - first loop to find the min and max (which you can do in the loop above)
            #   - second loop to normalize the angle
            # CAUTION: Process line by line, like the above loop, to prevent Out-Of-Memory error
            print(f"Processing2 {skel_file} to {output_path}")
            for line in tqdm(infile, desc="Processing frames", unit="video(s)"):
                # FRAME, JOINT*3
                all_joint = np.array(line.split(), dtype=float).reshape(-1, NUM_JOINT, 3)
                # print(all_joint.shape)
                for i in range(all_joint.shape[0]):  # loop through frame dimension
                    frame_joint = all_joint[i, :, :].flatten() # NUM_JOINT*3, 1 frame
                    original_rhand_joints = frame_joint[1434:1497]
                    original_lhand_joints = frame_joint[1497:1560]

                    # TODO: Please, unnormalize the position first
                    # unnorm_skels = (original_skels + 1) * (global_max - global_min) / 2 + global_min
                    # e.g. global min file -> /mnt/disks/general_backup/cropped_output2/preproces_data_global_min.npy
                    # Preprocess: Norm > Scale
                    # Postprocess: Unscale 
                    rhand_global_min = global_min[1434:1497]
                    rhand_global_max = global_max[1434:1497]

                    lhand_global_min = global_min[1497:1560]
                    lhand_global_max = global_max[1497:1560]

                    original_rhand_joints = (original_rhand_joints + 1)/2*(rhand_global_max - rhand_global_min) + rhand_global_min
                    original_lhand_joints = (original_lhand_joints + 1)/2*(lhand_global_max - lhand_global_min) + lhand_global_min

                    original_rhand_joints = (original_rhand_joints - original_rhand_joints[0]).reshape(-1,3)
                    original_lhand_joints = (original_lhand_joints - original_lhand_joints[0]).reshape(-1,3)

                    rhand_rel_angle = position_to_relative_angle(original_rhand_joints, HAND_JOINT_TO_PREV_JOINT_INDEX, ROOT_JOINT)
                    lhand_rel_angle = position_to_relative_angle(original_lhand_joints, HAND_JOINT_TO_PREV_JOINT_INDEX, ROOT_JOINT)
                    
                    hand_rel_angle = np.concatenate((rhand_rel_angle, lhand_rel_angle), axis=0) #(42, 2)
                    original_hand_joints = np.concatenate((original_rhand_joints, original_lhand_joints), axis=1) #(42, 3)
                    
                    scaled_hand_rel_angle = 2*((hand_rel_angle - global_min_angle) / (global_max_angle - global_min_angle + 1e-8)).reshape(-1,NUM_HAND_JOINT*2) - 1 #(N ,42*2)
                    
                    hand_representation = np.concatenate((original_hand_joints,scaled_hand_rel_angle), axis=1)
                    
                    outfile.write(" ".join(map(str, hand_representation.flatten())) + " ")
                outfile.write("\n")

                    
                
