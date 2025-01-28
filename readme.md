# Capstone code

## Description

- As we need to change some code from T2M-GPT to make it work on our dataset, so I download the code and make some changes to it on this repo.

## Installation
```bash
conda create --name capstone python=3.12 -y
conda activate capstone
pip install -r requirements.txt
pip install -e .
```

## Utilities functions

- `plot_skeletons_video` from `capstone_utils.plot_skeletons` to plot the skeleton on the video.
- `position_to_absolute_angle`, `absolute_angle_to_position` from `capstone_utils.absolute_angle_conversion` to convert the position to the absolute angle and vice versa.
  - absolute angle is the angle between each bone and the x, y, z axis.
- `position_to_relative_angle`, `relative_angle_to_position` from `capstone_utils.relative_angle_conversion` to convert the position to the relative angle and vice versa.
  - relative angle is the angle between each bone and the previous bone.
- bone model in `capstone_utils.skeleton_utils` to get the bone model including
- `bone_model` to get the bone model.
   1. bone model from progressive transformer paper.

## Scripts

### Convert position to relative angle

this file is in `scripts/convert_position_to_relative_angle.py` to convert the position to the relative angle.

to use it:
```bash
python ./scripts/convert_position_to_relative.py --pos-skeleton ./data/toy_data/train.skels \
                                                 --output ./data/toy_data/train.relative.skels \
                                                 --joint-sizes 150
```

### Convert position to absolute angle

this file is in `scripts/convert_position_to_absolute_angle.py` to convert the position to the absolute angle.

to use it:
```bash
python ./scripts/convert_position_to_absolute.py --pos-skeleton ./data/toy_data/train.skels \
                                                 --output ./data/toy_data/train.absolute.skels \
                                                 --joint-sizes 150
```

### Convert video to face landmarks

this file is in `scripts/convert_vdo_to_face_landmarks.py` to convert the video to the face landmarks.

<u>note</u>: this script need model weights to work which you can download from [here](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task) or
```bash
wget -O face_landmarker_v2_with_blendshapes.task -q https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

to use it:
```bash
python ./scripts/convert_vdo_to_face_landmarks.py --config ./configs/vdo_to_face_landmarks.example.yaml
```

### Convert video to skeleton

<u>note</u>: this script need model weights to work which you can download from [here](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task), [here](https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task), and [here](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task) or run this command to download them:
```bash
wget -O face_landmarker_v2_with_blendshapes.task -q https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task &&
wget -O pose_landmarker.task -q https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task &&
wget -O hand_landmarker.task -q https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

to use it:
```bash
python ./scripts/convert_vdo_to_skeleton.py --config ./configs/vdo_to_skeleton.yaml
```

#### Configs detail

This script use configuration yaml file. You can look the example in `configs/vdo_to_skeleton.yaml` file.

- `vdo_file`: the video file path. or `vdo_folder` to convert all videos in the folder.
- `output_folder`: the output folder to save the skeletons.
- `is_return_landmarked_vdo`: if you want to return the video with the landmarks drawn on it.
- `landmarks_format`: there are 2 options `normalized` and `pixel`.
  - `normalized`: the landmarks will be in the range [0, 1] which is `pixel / frame_size`.
  - `pixels`: the landmarks will be in the range [0, frame_size].
  - for `z` axis it will be in the range [0, 1] which divided by the frame size in `x` axis.
- `save_format`: the format of the saved skeletons file. there are 2 options `npy` and `txt`.
  - `npy`: save the skeletons in numpy format (used pickle).
  - `txt`: save the skeletons in text format (readable by IDE).
- Each model setting
  - You must specify the model path that you downloaded before.
  - Else is model parameters which you can find in the model documentation.

#### Output detail

There are 3 parts of the output:
1. face landmarks; including `478` landmarks.
2. hand landmarks; including `42` landmarks(21 for each hand).
3. pose landmarks; including `33` landmarks.

The output will be saved in the output folder and the name will be the same as the video file name + `.npy`.

The output shape will depends on the `save_format`:
- `npy`: the shape will be `(n_frames, n_landmarks, 3)` where the last axis is the `x, y, z` coordinates.
- `txt`: the shape will be `(n_frames, n_landmarks * 3)` where the last axis is the `x, y, z` coordinates.

<u>note</u>: `n_landmarks` is the sum of the face, hand, and pose landmarks.

## Training

### VQ-VAE
```bash
python trainer_vq_vae.py --config_path configs/to_vq_vae_config_path.yaml
```
which will use configuration file `configs/to_vq_vae_config_path.yaml` to train the model.

### T2M-GPT
```bash
python trainer_t2m_gpt.py --config_path configs/to_t2m_gpt_config_path.yaml
```
which will use configuration file `configs/to_t2m_gpt_config_path.yaml` to train the model.

All default configuration files are in `configs` folder.

## format code

```bash
isort .
black . -l 120
```
or 
```bash
make format all
```

- weight loss on figer to make it more detailes
- finetune from the checkpoint and use decay 0.99
- decay=0.5 is OK then change after in real dataset.
