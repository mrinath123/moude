# MoUDE
This is the official repository for MoUDE: Source-Free Incremental Domain Learning for Semantic Segmentation via Mixtures of Unsupervised Domain Experts

## Overview
Semantic segmentation models struggle to adapt to new domains in real-world scenarios where access to source data is restricted due to privacy or storage constraints. Traditional domain adaptation methods require simultaneous access to source and target domains, an impractical assumption in dynamic environments, and face challenges such as catastrophic forgetting when adapting incrementally to multiple domains. To address these issues, we propose MoUDE, a Source-Free Incremental Domain Adaptation framework that enables sequential adaptation to new domains without source data while preserving performance on previously encountered domains. MoUDE employs a two-stage approach: first, it learns domain-specific experts by freezing a pretrained source model and adapting lightweight LoRA modules using self-training on unlabeled target data. Second, it introduces a Mixture of Domain Experts, combining predictions from domain-specific experts through adaptive gating networks ensuring robust cross-domain performance. Evaluations on ACDC, Cityscapes-C, and Weather Cityscapes datasets demonstrate significant performance gains over baselines, establishing MoUDE as a robust solution for source-free incremental domain adaptation in semantic segmentation.

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
   - Download ACDC dataset from the [official website](https://acdc.vision.ee.ethz.ch/)
   - Store in the `acdc` directory in your project root
2. **Cityscapes and Cityscapes-C**:
   - Download Cityscapes from the [official website](https://www.cityscapes-dataset.com/)
   - Download or generate Cityscapes-C (weather corrupted Cityscapes)
   - Store in the `data` directory in your project root

Your data directory structure should look like:
```
data/
├── cityscapes/
│   ├── leftImg8bit_trainvaltest/
│   ├── gtFine_trainvaltest/
│   └── ...
├── acdc/
    ├── rgb_anon/
    ├── gt/
    └── ...
```

## Training
MoUDE employs a two-stage training process:

### Stage 1: Domain Expert Training
Train domain-specific experts using LoRA modules:

```bash
bash scripts/train.sh
```

#### Training Options:
1. **LoRA-only Finetuning**:
   - Use the `tools/train.py` script
2. **Full Finetuning**:
   - Use the `tools/train_lora_ft.py` script

Example training command:
```bash
python -m torch.distributed.launch --nproc_per_node=2 --master_port=29578 tools/train_lora_ft.py configs/moude/cityc_snow_cota.py --launcher pytorch
```

## Validation
You can evaluate your trained models using:
```bash
python tools/test.py configs/moude/cityc_snow_cota.py /path/to/checkpoint --eval mIoU
```

### Stage 2: Gating Network Training
After training domain experts in Stage 1, train the gating networks SERIALLY:

```bash
# Run the gating network training scripts in sequence
python gate_nightVrest.py
python gate_rainVrest.py
python gate_snowVres.py
```

**Note**: It is VERY IMPORTANT to run these scripts SERIALLY, one after the other completes.

## Inference
For inference with the trained model:

```bash
python infer_onevsrest.py
```
## Configuration
All training configuration files are located in `configs/moude/`. Choose the appropriate configuration based on your adaptation scenario:


## Pre-trained Weights
Coming soon.

