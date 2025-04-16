# MoUDE

This is the official repository for MoUDE: Source-Free Incremental Domain Learning for Semantic Segmentation via Mixtures of Unsupervised Domain Experts

## Overview

This repository contains code for training semantic segmentation models with domain adaptation techniques, focusing on weather and lighting condition adaptation. It supports both full finetuning and LoRA-based parameter-efficient finetuning.

## Installation

### For Conda:
```bash
conda env create -f environment_full.yml
conda activate cu11
```

### For Pip:
```bash
python3 -m venv cu11
source cu11/bin/activate
pip install -r requirements.txt
```

## Data Preparation

### Supported Datasets:
- Cityscapes
- ACDC
- Weather Corrupted Cityscapes
- Cityscapes-C


### Dataset Setup:

1. **ACDC Dataset**:
   - Download ACDC dataset
   - Store in `project/acdc` directory

2. **Cityscapes and Weather-corrupted Cityscapes**:
   - Download Cityscapes and Cityscapes-C datasets
   - Store in `project/data` directory

Your data directory structure should look like:
```
data/
├── cityscapes/
│   ├── leftImg8bit_trainvaltest/
│   ├── gtFine_trainvaltest/
│   └── ...
acdc/
    ├── rgb_anon/
    ├── gt/
    └── ...
```

## Training

You can run the training scripts using bash or submit via sbatch:

```bash
bash /BS/DApt/work/project/segformer_test/scripts/script.sh
```

### Training Options:

1. **LoRA-only Finetuning**:
   - Use `tools/train.py` script

2. **Full Finetuning**:
   - Use `tools/train_lora_ft.py` script

Example training command:
```bash
python3 -m torch.distributed.launch --nproc_per_node=2 --master_port=29578 tools/train_lora_ft.py local_config/my_models/cityc_snow_cota.py --launcher pytorch
```

## Configuration

All training configuration files are located in `local_config/my_models/`. Choose the appropriate configuration based on your adaptation scenario:

- For IDASS adaptation: Use `cityc_snow_idass.py`
- For COTTA adaptation: Use `cityc_snow_cota.py`

## Validation

You can evaluate your trained models using:
```bash
python tools/test.py local_config/my_models/cityc_snow_cota.py /path/to/checkpoint --eval mIoU
```

## Pre-trained Weights

Coming soon.
