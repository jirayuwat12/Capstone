# Capstone code

## Description

- As we need to change some code from T2M-GPT to make it work on our dataset, so I download the code and make some changes to it on this repo.

## Installation
```bash
conda create --name capstone python=3.12
conda activate capstone
pip install -r requirements.txt
pip install -e .
```

## format code
```bash
black . -l 120
```
or 
```bash
make format all
```