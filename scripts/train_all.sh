#!/bin/bash

echo "Start training all models"
conda init && conda activate capstone
python -m scripts.trainer_vq_vae --config_path /Users/jirayuwat/Desktop/Capstone/configs/trainer_vq_vae.original_paper.yaml
python -m scripts.trainer_t2m_trans_wrapper --config_path /Users/jirayuwat/Desktop/Capstone/configs/trainer_t2m_trans_wrapper.original_paper.yaml
