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
import wandb


def load_dataloader():
    dataloaders = []
    val_dataloaders = []
    #Fog
    dataset_config = '/BS/DApt/work/project/segformer_test/local_config/_base_/datasets/acdc_fog.py'
    cfg = Config.fromfile(dataset_config)
    dl1 = Runner.build_dataloader(cfg.train_dataloader)
    val_dl1 = Runner.build_dataloader(cfg.val_dataloader)
    dataloaders.append(dl1)
    val_dataloaders.append(val_dl1)
    #Night
    dataset_config = '/BS/DApt/work/project/segformer_test/local_config/_base_/datasets/acdc_night.py'
    cfg = Config.fromfile(dataset_config)
    dl2 = Runner.build_dataloader(cfg.train_dataloader)
    val_dl2 = Runner.build_dataloader(cfg.val_dataloader)
    dataloaders.append(dl2)
    val_dataloaders.append(val_dl2)
    #Rain
    dataset_config = '/BS/DApt/work/project/segformer_test/local_config/_base_/datasets/acdc_rain.py'
    cfg = Config.fromfile(dataset_config)
    dl3 = Runner.build_dataloader(cfg.train_dataloader)
    val_dl3 = Runner.build_dataloader(cfg.val_dataloader)
    dataloaders.append(dl3)
    val_dataloaders.append(val_dl3)
    #Snow
    dataset_config = '/BS/DApt/work/project/segformer_test/local_config/_base_/datasets/acdc_snow.py'
    cfg = Config.fromfile(dataset_config)
    dl4 = Runner.build_dataloader(cfg.train_dataloader)
    val_dl4 = Runner.build_dataloader(cfg.val_dataloader)
    dataloaders.append(dl4)
    val_dataloaders.append(val_dl4)

    return dataloaders,val_dataloaders

data_loaders,val_dataloaders = load_dataloader()

def label_entropy(data_loader):
    num_classes = 19
    entropies = []  # To store entropies of all patches across all batches

    patch_rows, patch_cols = 8,8  # 4x4 patches in a 16-patch grid

    for batch_ndx, sample in enumerate(data_loader):
        gts = []
        for gt in sample['data_samples']:
            gt = gt._gt_sem_seg.data
            gt = gt.float()
            gts.append(gt)
        
        gt = torch.stack(gts, dim=0).cuda()
        gt = F.interpolate(gt, size=(540, 960), mode='nearest')
        unfold = torch.nn.Unfold(kernel_size=(540 // patch_rows, 960 // patch_cols), stride=(540 // patch_rows, 960 // patch_cols))
        predx = gt.float()

        predx = gt.float()
        x = unfold(predx)  # B (c*p*p) no.of patches
        bs, c, h, w = predx.shape
        patch_h, patch_w = 540 // patch_rows, 960 // patch_cols
        num_patches = (h // patch_h) * (w // patch_w)
        a = x.reshape(bs, c, patch_h, patch_w, num_patches).permute(0, 4, 1, 2, 3)
        batch, num_patches, channels, patch_height, patch_width = a.shape
        z = a.reshape(batch * num_patches, patch_height * patch_width)
        class_dist = torch.zeros(batch * num_patches, num_classes, device=z.device)
        for i in range(class_dist.shape[0]):
            for n in range(num_classes):
                class_dist[i, n] = (z[i] == n).float().sum()
        class_dist /= class_dist.sum(dim=-1, keepdim=True)
        # Compute entropy for each patch
        entropy = -torch.sum(class_dist * torch.log2(class_dist + 1e-12), dim=-1)
        entropy = entropy.view(batch, num_patches)

        # Accumulate entropies
        if len(entropies) == 0:
            entropies = entropy
        else:
            entropies = torch.cat((entropies, entropy), dim=0)

    # Calculate the average entropy for each patch across all images
    avg_entropy_per_patch = torch.mean(entropy, dim=0) 

    # Print the average entropy for each patch
    for i, entropy_value in enumerate(avg_entropy_per_patch):
        if i == 0:
            suffix = "st"
        elif i == 1:
            suffix = "nd"
        elif i == 2:
            suffix = "rd"
        else:
            suffix = "th"
        
        print(f"{i+1}{suffix} patch avg entropy: {entropy_value.item():.4f}")

    return avg_entropy_per_patch

def label_entropy_per_class(data_loader):
    num_classes = 19
    patch_rows, patch_cols = 4, 4  # 4x4 patches in a 16-patch grid
    num_patches = patch_rows * patch_cols

    # To store entropies of all patches across all batches
    entropies_per_class = torch.zeros(num_patches, num_classes, device='cuda')
    patch_counts = torch.zeros(num_patches, device='cuda')

    for batch_ndx, sample in enumerate(data_loader):
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
        bs, c, h, w = predx.shape
        patch_h, patch_w = 540 // patch_rows, 960 // patch_cols
        num_patches = (h // patch_h) * (w // patch_w)
        a = x.reshape(bs, c, patch_h, patch_w, num_patches).permute(0, 4, 1, 2, 3)
        # a -> ( B no.of patches c p p )
        batch, num_patches, channels, patch_height, patch_width = a.shape
        z = a.reshape(batch * num_patches, patch_height * patch_width) # batch * num_patches -> total individual patches

        class_dist = torch.zeros(batch * num_patches, num_classes, device=z.device)
        
        for i in range(class_dist.shape[0]):
            for n in range(num_classes):
                class_dist[i, n] = (z[i] == n).sum()
        
        class_dist /= class_dist.sum(dim=-1, keepdim=True)

        #import pdb;pdb.set_trace()

        # Compute entropy for each class in each patch
        entropy_per_class = -class_dist * torch.log2(class_dist + 1e-12)
        #entropy_per_class = -torch.log2(class_dist + 1e-12)
        entropy_per_class = entropy_per_class.view(batch, num_patches, num_classes)

        # Accumulate entropies per class
        entropies_per_class += torch.sum(entropy_per_class, dim=0)
        patch_counts += batch

    #import pdb;pdb.set_trace()
    # Calculate the average entropy for each class across all patches and images
    avg_entropy_per_class_per_patch = entropies_per_class / patch_counts.view(-1, 1)

    # Print the average entropy for each class per patch
    for patch_idx in range(num_patches):
        print(f"{patch_idx+1}th patch class-wise avg entropy:")
        for class_idx in range(num_classes):
            avg_entropy = avg_entropy_per_class_per_patch[patch_idx, class_idx].item()
            print(f"  Class {class_idx}: {avg_entropy:.4f}")
    
    return avg_entropy_per_class_per_patch




def distribution_per_class_per_patch(data_loader):
    num_classes = 19
    patch_rows, patch_cols = 4, 4  # 4x4 patches in a 16-patch grid
    num_patches = patch_rows * patch_cols

    # To store entropies of all patches across all batches
    entropies_per_class = torch.zeros(num_patches, num_classes, device='cuda')
    patch_counts = torch.zeros(num_patches, device='cuda')

    for batch_ndx, sample in enumerate(data_loader):
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
        bs, c, h, w = predx.shape
        patch_h, patch_w = 540 // patch_rows, 960 // patch_cols
        num_patches = (h // patch_h) * (w // patch_w)
        a = x.reshape(bs, c, patch_h, patch_w, num_patches).permute(0, 4, 1, 2, 3)
        # a -> ( B no.of patches c p p )
        batch, num_patches, channels, patch_height, patch_width = a.shape
        z = a.reshape(batch * num_patches, patch_height * patch_width) # batch * num_patches -> total individual patches

        class_dist = torch.zeros(batch * num_patches, num_classes, device=z.device)
        
        for i in range(class_dist.shape[0]):
            for n in range(num_classes):
                class_dist[i, n] = (z[i] == n).sum()
        
        class_dist /= class_dist.sum(dim=-1, keepdim=True)

        #import pdb;pdb.set_trace()

        # Compute entropy for each class in each patch
        entropy_per_class = -class_dist * torch.log2(class_dist + 1e-12)
        #entropy_per_class = -torch.log2(class_dist + 1e-12)
        entropy_per_class = entropy_per_class.view(batch, num_patches, num_classes)

        # Accumulate entropies per class
        entropies_per_class += torch.sum(entropy_per_class, dim=0)
        patch_counts += batch

    #import pdb;pdb.set_trace()
    # Calculate the average entropy for each class across all patches and images
    avg_entropy_per_class_per_patch = entropies_per_class / patch_counts.view(-1, 1)

    # Print the average entropy for each class per patch
    for patch_idx in range(num_patches):
        print(f"{patch_idx+1}th patch class-wise avg entropy:")
        for class_idx in range(num_classes):
            avg_entropy = avg_entropy_per_class_per_patch[patch_idx, class_idx].item()
            print(f"  Class {class_idx}: {avg_entropy:.4f}")
    
    return avg_entropy_per_class_per_patch

def distribution_per_class(data_loader):
    num_classes = 19
    entropies = []  # To store entropies of all patches across all batches

    patch_rows, patch_cols = 8,8  # 4x4 patches in a 16-patch grid

    for batch_ndx, sample in enumerate(data_loader):
        gts = []
        for gt in sample['data_samples']:
            gt = gt._gt_sem_seg.data
            gt = gt.float()
            gts.append(gt)
        
        gt = torch.stack(gts, dim=0).cuda()
        gt = F.interpolate(gt, size=(540, 960), mode='nearest')
        unfold = torch.nn.Unfold(kernel_size=(540 // patch_rows, 960 // patch_cols), stride=(540 // patch_rows, 960 // patch_cols))
        predx = gt.float()

        predx = gt.float()
        x = unfold(predx)  # B (c*p*p) no.of patches
        bs, c, h, w = predx.shape
        patch_h, patch_w = 540 // patch_rows, 960 // patch_cols
        num_patches = (h // patch_h) * (w // patch_w)
        a = x.reshape(bs, c, patch_h, patch_w, num_patches).permute(0, 4, 1, 2, 3)
        batch, num_patches, channels, patch_height, patch_width = a.shape
        z = a.reshape(batch * num_patches, patch_height * patch_width)
        class_dist = torch.zeros(batch * num_patches, num_classes, device=z.device)
        for i in range(class_dist.shape[0]):
            for n in range(num_classes):
                class_dist[i, n] = (z[i] == n).float().sum()
        class_dist /= class_dist.sum(dim=-1, keepdim=True)
        # Compute entropy for each patch
        entropy = -torch.sum(class_dist * torch.log2(class_dist + 1e-12), dim=-1)
        entropy = entropy.view(batch, num_patches)

        # Accumulate entropies
        if len(entropies) == 0:
            entropies = entropy
        else:
            entropies = torch.cat((entropies, entropy), dim=0)

    # Calculate the average entropy for each patch across all images
    avg_entropy_per_patch = torch.mean(entropy, dim=0) 

    # Print the average entropy for each patch
    for i, entropy_value in enumerate(avg_entropy_per_patch):
        if i == 0:
            suffix = "st"
        elif i == 1:
            suffix = "nd"
        elif i == 2:
            suffix = "rd"
        else:
            suffix = "th"
        
        print(f"{i+1}{suffix} patch avg entropy: {entropy_value.item():.4f}")

    return avg_entropy_per_patch

