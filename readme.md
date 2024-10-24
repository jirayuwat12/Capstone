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

## Traing

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
