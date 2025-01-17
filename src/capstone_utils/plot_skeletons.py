import math

import cv2
import numpy as np
from capstone_utils.skeleton_utils import get_bone_colour
from capstone_utils.skeleton_utils.progressive_trans_model import SKELETON_MODEL


def plot_skeletons_video(
    joints: np.ndarray,
    file_path: str,
    video_name: str,
    references: np.ndarray | None = None,
    skip_frames: int = 1,
    sequence_ID: str | None = None,
    pad_token: int = 0,
    frame_offset: tuple[int, int] = (0, 0),
    debug: bool = False,
) -> None:
    """
    This function plots a video of the given joints, with the option to include reference joints.

    :param joints: The joints to plot and the joints must in positional format and in shape `(T, joints_3)`
    :param file_path: The file path to save the video
    :param video_name: The name of the video (must include file extension)
    :param references: The reference joints to plot alongside the predicted joints and must be in the same format as `joints`
    :param skip_frames: The number of frames to skip between each frame
    :param sequence_ID: The sequence ID to include in the video
    :param pad_token: The token to pad the joints with
    """
    if debug:
        print("--- Plotting Skeletons Video ---")
    # Create video template
    FPS = 25 // skip_frames
    video_file = file_path + "/{}.mp4".format(video_name.split(".")[0])
    if debug:
        print(f"Creating video at {video_file}")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    frame_size = (650, 650) if references is None else (1300, 650)
    video = cv2.VideoWriter(video_file, fourcc, float(FPS), frame_size, True)

    for frame_index, frame_joints in enumerate(joints):
        # Reached padding
        if pad_token in frame_joints:
            if debug:
                print(f"Padding reached at frame {frame_index}")
                print(f"\tPadding token: {pad_token}")
                print(f"\tPadding frame: {frame_joints}")
            continue

        # Initialise frame of white
        frame = np.ones((650, 650, 3), np.uint8) * 255

        # Cut off the percent_tok, multiply by 3 to restore joint size
        frame_joints = frame_joints[:] * 3
        frame_joints_2d = np.reshape(frame_joints, (50, 3))[:, :2]

        # Draw the frame given 2D joints and add text
        if debug:
            print(f"Drawing frame {frame_index} to video")
        draw_frame_2D(frame, frame_joints_2d, offset=frame_offset)
        if debug:
            print(f"Drawing text to frame {frame_index}")
        cv2.putText(frame, "Predicted Sign Pose", (180, 600), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # If reference is provided, create and concatenate on the end
        if references is not None:
            if debug:
                print(f"Reference provided for frame {frame_index}")
                print(f"Shape of references: {references.shape}")
            # Extract the reference joints
            ref_joints = references[frame_index]
            if debug:
                print(f"Reference joints for frame {frame_index}")
                print(f"Shape of reference joints: {ref_joints.shape}")
            # Initialise frame of white
            ref_frame = np.ones((650, 650, 3), np.uint8) * 255

            # Cut off the percent_tok and multiply each joint by 3 (as was reduced in training files)
            ref_joints = ref_joints[:] * 3
            if debug:
                print("ref_joints", ref_joints.shape)
            ref_joints_2d = np.reshape(ref_joints, (50, 3))[:, :2]
            if debug:
                print("ref_joints_2d", ref_joints_2d.shape)

            # Draw these joints on the frame and add text
            draw_frame_2D(ref_frame, ref_joints_2d, offset=(0, -20))
            cv2.putText(ref_frame, "Ground Truth Pose", (190, 600), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            # Concatenate the two frames
            frame = np.concatenate((frame, ref_frame), axis=1)

            # Add the sequence ID to the frame
            sequence_ID_write = "Sequence ID: " + sequence_ID.split("/")[-1]
            cv2.putText(frame, sequence_ID_write, (700, 635), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Write the video frame
        if debug:
            print(f"Writing frame {frame_index} to video")
            print(frame.shape)
        video.write(frame)
    # Release the video
    video.release()

    if debug:
        print(f"Video saved at {video_file}")
        print("--- Plotting Skeletons Video ---")


def draw_line(
    im: np.ndarray, joint1: np.ndarray, joint2: np.ndarray, c: tuple[int, int, int] = (0, 0, 255), width: int = 3
) -> None:
    """
    This function draws a line between two points on the given image.

    :param im: The image to draw the line on
    :param joint1: The first joint to draw the line between and must has only 2 values (x, y)
    :param joint2: The second joint to draw the line between and must has only 2 values (x, y)
    :param c: The colour of the line in the format `(B, G, R)`
    :param width: The width of the line to draw
    """
    thresh = -100
    if joint1[0] > thresh and joint1[1] > thresh and joint2[0] > thresh and joint2[1] > thresh:
        # compute the center of the line using average of the two points
        center = (int((joint1[0] + joint2[0]) / 2), int((joint1[1] + joint2[1]) / 2))
        # compute the length of the line using the euclidean distance between the two points
        length = int(math.sqrt(((joint1[0] - joint2[0]) ** 2) + ((joint1[1] - joint2[1]) ** 2)) / 2)
        # compute the angle of the line using the arctan of the two points
        angle = math.degrees(math.atan2((joint1[0] - joint2[0]), (joint1[1] - joint2[1])))

        # draw the line
        cv2.ellipse(im, center, (width, length), -angle, 0.0, 360.0, c, -1)


def draw_frame_2D(frame: np.ndarray, joints: np.ndarray, offset: tuple[int, int] = (350, 250)) -> None:
    """
    This function draws the 2D joints on the given frame.

    :param frame: The frame to draw the joints on
    :param joints: The joints to draw on the frame and must in the positional format `(joints, 2)`
    :param offset: The offset to center the skeleton around
    """
    # Line to be between the stacked
    draw_line(frame, [1, 650], [1, 1], c=(0, 0, 0), width=1)
    # Give an offset to center the skeleton around

    # Get the skeleton structure details of each bone, and size
    skeleton = SKELETON_MODEL
    skeleton = np.array(skeleton)

    number = skeleton.shape[0]

    # Increase the size and position of the joints
    joints = joints * 10 * 12 * 2
    joints = joints + np.ones((50, 2)) * offset

    # Loop through each of the bone structures, and plot the bone
    for j in range(number):
        c = get_bone_colour(skeleton, j)

        draw_line(
            frame,
            [joints[skeleton[j, 0]][0], joints[skeleton[j, 0]][1]],
            [joints[skeleton[j, 1]][0], joints[skeleton[j, 1]][1]],
            c=c,
            width=1,
        )
