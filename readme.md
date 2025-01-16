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

to use it:
```bash
python ./scripts/convert_vdo_to_face_landmarks.py --config ./configs/vdo_to_face_landmarks.example.yaml
```

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
