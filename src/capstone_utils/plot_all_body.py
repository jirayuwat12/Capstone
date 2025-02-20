import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


def connection_pairs():
    connection_pairs_list = [
        # mouth
        (78, 191),
        (191, 80),
        (80, 81),
        (81, 82),
        (82, 13),
        (13, 312),
        (312, 311),
        (311, 310),
        (310, 415),
        (415, 308),
        (308, 324),
        (324, 318),
        (318, 402),
        (402, 317),
        (317, 14),
        (14, 87),
        (87, 178),
        (178, 88),
        (88, 95),
        (95, 78),
        (61, 185),
        (185, 40),
        (40, 39),
        (39, 37),
        (37, 0),
        (0, 267),
        (267, 269),
        (269, 270),
        (270, 409),
        (409, 291),
        (291, 375),
        (375, 321),
        (321, 405),
        (405, 314),
        (314, 17),
        (17, 84),
        (84, 181),
        (181, 91),
        (91, 146),
        (146, 61),
        # face
        (10, 109),
        (109, 67),
        (67, 103),
        (103, 54),
        (54, 21),
        (21, 162),
        (162, 127),
        (127, 234),
        (234, 93),
        (93, 132),
        (132, 58),
        (58, 172),
        (172, 136),
        (136, 150),
        (150, 149),
        (149, 176),
        (176, 148),
        (148, 152),
        (152, 377),
        (377, 400),
        (400, 378),
        (378, 379),
        (379, 365),
        (365, 397),
        (397, 288),
        (288, 361),
        (361, 323),
        (323, 454),
        (454, 356),
        (356, 389),
        (389, 251),
        (251, 284),
        (284, 332),
        (332, 297),
        (297, 338),
        (338, 10),
        # right eye
        (33, 246),
        (246, 161),
        (161, 160),
        (160, 159),
        (159, 158),
        (158, 157),
        (157, 173),
        (173, 133),
        (133, 155),
        (155, 154),
        (154, 153),
        (153, 145),
        (145, 144),
        (144, 163),
        (163, 7),
        (7, 33),
        # right eyebrow
        (247, 30),
        (30, 29),
        (29, 27),
        (27, 28),
        (28, 56),
        (56, 190),
        # left eye
        (362, 398),
        (398, 384),
        (384, 385),
        (385, 386),
        (386, 387),
        (387, 388),
        (388, 466),
        (466, 263),
        (263, 249),
        (249, 390),
        (390, 373),
        (373, 374),
        (374, 380),
        (380, 381),
        (381, 382),
        (382, 362),
        # left eyebrow
        (414, 286),
        (286, 258),
        (258, 257),
        (257, 259),
        (259, 260),
        (260, 467),
        # nose
        (2, 1),
        (1, 4),
        (4, 5),
        (5, 195),
        (195, 197),
        (197, 6),
        # LEFT SIDE
        # upper arm (shoulder -> elbow)
        (532, 534),
        # lower arm (elbow -> wrist)
        (534, 536),
        # hand (wrist -> palm)
        (536, 538),
        (538, 540),
        (540, 542),
        (542, 536),
        # left hand
        # thumb
        (499, 500),
        (500, 501),
        (501, 502),
        (502, 503),
        # index
        (504, 505),
        (505, 506),
        (506, 507),
        # middle
        (508, 509),
        (509, 510),
        (510, 511),
        # ring finger
        (512, 513),
        (513, 514),
        (514, 515),
        # little finger
        (516, 517),
        (517, 518),
        (518, 519),
        # RIGHT SIDE
        # upper arm (shoulder -> elbow)
        (531, 533),
        # lower arm (elbow -> wrist)
        (533, 535),
        # hand (wrist -> palm)
        (535, 537),
        (537, 539),
        (539, 541),
        (541, 535),
        # right hand
        # thumb
        (478, 479),
        (479, 480),
        (480, 481),
        (481, 482),
        # index
        (483, 484),
        (484, 485),
        (485, 486),
        # middle
        (487, 488),
        (488, 489),
        (489, 490),
        # ring finger
        (491, 492),
        (492, 493),
        (493, 494),
        # little finger
        (495, 496),
        (496, 497),
        (497, 498),
        # body
        (532, 544),
        (544, 546),
        (531, 543),
        (543, 545),
    ]
    return connection_pairs_list


def create_sign_language_video(file_name, connection_list, joint_number=553, coordinate=3, frame_per_sec_number=24):
    # Read the file with the sign language data
    with open(file_name, "r") as f:
        sign_language_data = f.readlines()
    for line in sign_language_data:
        assert (
            len(line) % joint_number != 0
        ), f"Skipping invalid line with {len(line)} elements (not divisible by {joint_number})"

    # Parse each line into a list of floating-point numbers
    sign_language_data = [[float(word) for word in line.split()] for line in sign_language_data]

    for idx in range(len(sign_language_data)):
        # Use the data
        data = sign_language_data[idx]
        data = np.array(data)

        # check range
        x_coords = data[::3]
        y_coords = data[1::3]

        x_coord_in_frame = np.array(x_coords).reshape(-1, 553)
        y_coord_in_frame = np.array(y_coords).reshape(-1, 553)
        print(f"X Coordinate Shape: {x_coord_in_frame.shape}")
        print(f"Y Coordinate Shape: {y_coord_in_frame.shape}")

        is_x_coord_as_same_as_previous = np.all(np.isclose(x_coord_in_frame[1:], x_coord_in_frame[:-1], atol=1), axis=1)
        is_y_coord_as_same_as_previous = np.all(np.isclose(y_coord_in_frame[1:], y_coord_in_frame[:-1], atol=1), axis=1)
        is_frame_same_as_previous = is_x_coord_as_same_as_previous & is_y_coord_as_same_as_previous
        print(f"X Coordinate same amount: {np.sum(is_x_coord_as_same_as_previous)}")
        print(f"Y Coordinate same amount: {np.sum(is_y_coord_as_same_as_previous)}")
        print(f"Frame same amount: {np.sum(is_frame_same_as_previous)}")

        # return
        data_count = len(data)  # 328482
        frame_count = data_count // (joint_number * coordinate)  # 328482 // (553 * 3) = 198

        # Calculate the duration of the video in seconds
        video_duration_seconds = frame_count / frame_per_sec_number  # 198 / 24 = 8.25
        print(f"Video Duration: {video_duration_seconds:.2f} seconds")

        # Create a figure for the animation
        fig, ax = plt.subplots(figsize=(50, 50))

        # Set axis limits according to the specified 2D plane bounds
        ax.set_xlim(min(x_coords) - 10, max(x_coords) + 10)
        ax.set_ylim(max(y_coords) + 10, min(y_coords) - 10)

        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.set_title("2D Plot of Joints")

        # Initialize a scatter plot object (empty for now)
        scatter = ax.scatter([], [], color="blue", label="Joints", s=10)

        # # Initialize a line object to connect points (empty for now)
        # line, = ax.plot([], [], color='red', linestyle='-', linewidth=2, label='Connection')

        # Initialize a list to store annotations
        annotations = []

        # Define pairs o
        # Create a list of line objects, one for each connection pair
        lines = [ax.plot([], [], color="red", linestyle="-", linewidth=2)[0] for _ in connection_list]

        # Function to connect specified joints in the frame
        def connect_joints(x_coords, y_coords, connection_pairs):
            for i, (joint1, joint2) in enumerate(connection_pairs):
                if joint1 < len(x_coords) and joint2 < len(x_coords):
                    # Update the corresponding line for each connection pair
                    lines[i].set_data(
                        [x_coords[joint1], x_coords[joint2]],  # X coordinates of the two points
                        [y_coords[joint1], y_coords[joint2]],  # Y coordinates of the two points
                    )
                else:
                    lines[i].set_data([], [])  # Hide the line if the points are out of bounds

        # Function to update the plot for each frame
        def update(frame):
            # Calculate the indices for the current frame
            start_idx = frame * joint_number * coordinate
            end_idx = (frame + 1) * joint_number * coordinate
            # if frame%10==0:
            #     print(f"this is frame#{frame} : {start_idx}, {end_idx}")

            # Extract x and y coordinates for the current frame
            x_coords = data[start_idx:end_idx:coordinate]  # Every 3rd value starting from index 0 (x-coordinates)
            y_coords = data[
                start_idx + 1 : end_idx : coordinate
            ]  # Every 3rd value starting from index 1 (y-coordinates)

            # Update the scatter plot
            scatter.set_offsets(np.c_[x_coords, y_coords])

            # Clear previous annotations
            for ann in annotations:
                ann.remove()
            annotations.clear()

            # Add new annotations for each point (with larger font size)
            for i, (x, y) in enumerate(zip(x_coords, y_coords)):
                ann = ax.annotate(
                    str(i % 553),
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                    fontsize=8,  # Increase font size here
                )
                annotations.append(ann)

            # Connect specified joints with lines
            connect_joints(x_coords, y_coords, connection_list)

            # Save the current frame as an image
            # plt.savefig(f"/content/drive/MyDrive/Capstone/output_vdo/frame_{frame:04d}.png")  # Save with zero-padded frame number
            # print(f"Saved frame {frame}")

            return (scatter,)

        # Create an animation object
        ani = animation.FuncAnimation(fig, update, frames=frame_count, interval=1000 / frame_per_sec_number, blit=True)

        # Save the animation as a video (e.g., 'sign_language_video.mp4')
        ani.save(
            f"./#{idx}.mp4",
            writer="ffmpeg",
            fps=frame_per_sec_number,
        )

        # Optionally, show the animation
        # plt.show()

        print(f"Video#{idx} saved successfully!")


# Example usage
if __name__ == "__main__":
    create_sign_language_video(
        "/content/drive/MyDrive/Capstone/2110488-Capstone-Text2Sign/T2S-GPT/result/DVQVAE_trainer_9/X_re.skels",
        joint_number=553,
        coordinate=3,
        frame_per_sec_number=24,
        connection_list=connection_pairs(),
    )
