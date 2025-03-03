#!/bin/bash

echo "Start preparing data"
conda init && conda activate capstone
python -m scripts.data_preparation --config_path /Users/jirayuwat/Desktop/Capstone/configs/data_preparation.lanta.yaml