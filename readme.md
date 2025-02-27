# Capstone code

## Description

- As we need to change some code from T2M-GPT to make it work on our dataset, so I download the code and make some changes to it on this repo.

## Installation
```bash
> conda create --name capstone python=3.12 -y
> conda activate capstone
> pip install -e .
```

## Scripts

### Convert position to relative or absolute angle

Convert the file in the position format to the relative angle format or the absolute angle format.

to use it:
```bash
> python -m scripts.convert_position_to_relative
or
> python -m scripts.convert_position_to_absolute
```
about the configuration file, you can find it in `configs/convert_position_to_relative.yaml` and `configs/convert_position_to_absolute.yaml`.

### Convert video to skeleton

This script needs the face landmarker mediapipe model to convert the video to the skeleton which you can download using the following command:
```bash
> TODO: to be added
```

to use it:
```bash
> python -m scripts.convert_vdo_to_skeleton
```

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
python -m scripts.trainer_vq_vae
```
which will use configuration file `configs/trainer_vq_vae.yaml` to train the model.

### T2M-GPT
```bash
python -m scripts.trainer_t2m_trans_wrapper
```
which will use configuration file `configs/trainer_t2m_trans_wrapper.yaml` to train the model.

## format code

```bash
isort .
black . -l 120
```
or 
```bash
make format all
```