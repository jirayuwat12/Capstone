import argparse

import cv2
import numpy as np

# Create a parser object
argparser = argparse.ArgumentParser(description="Merge VDO to the same canvas")
argparser.add_argument("--left_vdo", help="Path to the left VDO file", required=True)
argparser.add_argument("--right_vdo", help="Path to the right VDO file", required=True)
argparser.add_argument("--output_vdo", help="Path to the output VDO file", required=True)

args = argparser.parse_args()

# Load the VDO files
left_vdo = cv2.VideoCapture(args.left_vdo)
right_vdo = cv2.VideoCapture(args.right_vdo)

# Get the frame rate of the VDO files
left_fps = left_vdo.get(cv2.CAP_PROP_FPS)
right_fps = right_vdo.get(cv2.CAP_PROP_FPS)
skip_factor = 1
if left_fps != right_fps:
    # Compute the skip factor for the right VDO
    skip_factor = int(right_fps / left_fps)
    print(f"Skip factor: {skip_factor}")
print(f"Left frames per second: {left_fps}, Right frames per second: {right_fps}")
print(
    f"Left frame count: {left_vdo.get(cv2.CAP_PROP_FRAME_COUNT)}, Right frame count: {right_vdo.get(cv2.CAP_PROP_FRAME_COUNT)}"
)
print(f"Right frame count after skipping: {right_vdo.get(cv2.CAP_PROP_FRAME_COUNT)/skip_factor}")

# Get the width and height of the VDO files
left_width = int(left_vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
left_height = int(left_vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
right_width = int(right_vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
right_height = int(right_vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))

output_width = left_width + right_width
output_height = max(left_height, right_height)

# Create the output VDO
fourcc = cv2.VideoWriter_fourcc(*"MP4V")
output_vdo = cv2.VideoWriter(args.output_vdo, fourcc, left_fps, (output_width, output_height))

# Merge the VDO files
_, _ = left_vdo.read()
while True:
    left_ret, left_frame = left_vdo.read()
    # Skip some frames in the right VDO
    right_frame = None
    for _ in range(skip_factor):
        right_ret, right_frame = right_vdo.read()

    if not left_ret or not right_ret:
        break

    # Resize the right frame to match the height of the left frame
    right_frame = cv2.resize(right_frame, (right_width * left_height // right_height, left_height))

    left_frame = cv2.resize(left_frame, (left_width * output_height // left_height, output_height))
    output_frame = np.zeros((output_height, output_width, 3), dtype=np.uint8)
    output_frame[:, :left_width] = left_frame
    output_frame[:, left_width:] = right_frame

    output_vdo.write(output_frame)

# Release the VDO files
left_vdo.release()
right_vdo.release()
output_vdo.release()
print(f"Saved the merged VDO to {args.output_vdo}")
