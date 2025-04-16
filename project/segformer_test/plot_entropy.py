import torch
import torch.nn as nn
import numpy as np
from mmseg.registry import MODELS
from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.model import BaseModule
from mmengine.registry import init_default_scope
init_default_scope('mmseg')
from mmseg.models.segmentors import EncoderDecoder
from mmseg.datasets.acdc import ACDCDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torchvision.utils import save_image
import os
from torch.cuda import amp
import matplotlib.pyplot as plt



def label_entropy_per_class(data_loader):
    num_classes = 19
    patch_rows, patch_cols = 4, 4  # 4x4 patches in a 16-patch grid
    num_patches = patch_rows * patch_cols

    # To store entropies of all patches across all batches
    entropies_per_class = torch.zeros(num_patches, num_classes, device='cuda')
    patch_counts = torch.zeros(num_patches, device='cuda')

    sample  = next(iter(data_loader))
    gts = []
    for gt in sample['data_samples']:
        gt = gt._gt_sem_seg.data
        gt = gt.float()
        gts.append(gt)
    
    gt = torch.stack(gts, dim=0).cuda()
    gt = F.interpolate(gt, size=(540, 960), mode='nearest')
    unfold = torch.nn.Unfold(kernel_size=(540 // patch_rows, 960 // patch_cols), stride=(540 // patch_rows, 960 // patch_cols))
    predx = gt.float()
    x = unfold(predx)  # B (c*p*p) no.of patches
    y = x.view(gt.shape[0], num_patches, gt.shape[1], 540 // patch_rows, 960 // patch_cols)  # (B, num_patches, c, p, p)
    
    batch, num_patches, channels, patch_height, patch_width = y.shape
    z = y.view(batch * num_patches, patch_height * patch_width)

    class_dist = torch.zeros(batch * num_patches, num_classes, device=z.device).long()
    

    for i in range(class_dist.shape[0]):
        for n in range(num_classes):
            class_dist[i, n] = (z[i] == n).sum()

    
    import pdb;pdb.set_trace()
    class_dist /= class_dist.sum(dim=-1, keepdim=True)

    #import pdb;pdb.set_trace()

    # Compute entropy for each class in each patch
    entropy_per_class = -class_dist * torch.log2(class_dist + 1e-12)
    entropy_per_class = entropy_per_class.view(batch, num_patches, num_classes)

    # Accumulate entropies per class
    entropies_per_class += torch.sum(entropy_per_class, dim=0)
    patch_counts += batch

    # Calculate the average entropy for each class across all patches and images
    avg_entropy_per_class_per_patch = entropies_per_class / patch_counts.view(-1, 1)

    # Print the average entropy for each class per patch
    for patch_idx in range(num_patches):
        print(f"{patch_idx+1}th patch class-wise avg entropy:")
        for class_idx in range(num_classes):
            print(f"  Class {class_idx}: {avg_entropy_per_class_per_patch[patch_idx, class_idx].item()}")
    
    return avg_entropy_per_class_per_patch