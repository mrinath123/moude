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

def convert_syncbn_to_bn(module):
    # reqd if training on 1 gpu
    mod = module
    if isinstance(module, nn.SyncBatchNorm):
        mod = nn.BatchNorm2d(module.num_features, module.eps, module.momentum, module.affine, module.track_running_stats)
        mod.running_mean = module.running_mean
        mod.running_var = module.running_var
        if module.affine:
            mod.weight = module.weight
            mod.bias = module.bias
    for name, child in module.named_children():
        mod.add_module(name, convert_syncbn_to_bn(child))
    return mod

def remove_prefix(weights,prefix = 'module'):
    new_state_dict = {}
    for n, v in weights.items():
        if prefix in n:
            parts = n.split('.')
            new_n = '.'.join(parts[1:])
            new_state_dict[new_n] = v
        else:
            new_state_dict[n] = v
    print('prefix removed') 

    return new_state_dict

def load_cotta_model():
    config_path = '/BS/DApt/work/project/segformer_test/local_config/my_models/b5_rain_nlora.py'
    pretrained = '/BS/DApt/work/project/segformer_test/work_dirs/cottav4/fog_wo_rest0.6900_200.pth'
    cfg = Config.fromfile(config_path)
    encdec = EncoderDecoder(cfg.model.backbone, cfg.model.decode_head)
    weights = torch.load(pretrained)
    weights = remove_prefix(weights)
    for m in encdec.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    
    encdec.load_state_dict(weights)
    encdec = convert_syncbn_to_bn(encdec)

    encdec.eval()
    encdec.cuda()

    encdec.requires_grad_(False) # Set requires_grad False

    return encdec

def load_dataloader():
    val_dataloaders = []
    #Fog
    dataset_config = '/BS/DApt/work/project/segformer_test/local_config/_base_/datasets/acdc_fog.py'
    cfg = Config.fromfile(dataset_config)
    val_dl1 = Runner.build_dataloader(cfg.val_dataloader)
    val_dataloaders.append(val_dl1)
    #Night
    dataset_config = '/BS/DApt/work/project/segformer_test/local_config/_base_/datasets/acdc_night.py'
    cfg = Config.fromfile(dataset_config)
    val_dl2 = Runner.build_dataloader(cfg.val_dataloader)
    val_dataloaders.append(val_dl2)
    #Rain
    dataset_config = '/BS/DApt/work/project/segformer_test/local_config/_base_/datasets/acdc_rain.py'
    cfg = Config.fromfile(dataset_config)
    val_dl3 = Runner.build_dataloader(cfg.val_dataloader)
    val_dataloaders.append(val_dl3)
    #Snow
    dataset_config = '/BS/DApt/work/project/segformer_test/local_config/_base_/datasets/acdc_snow.py'
    cfg = Config.fromfile(dataset_config)
    val_dl4 = Runner.build_dataloader(cfg.val_dataloader)
    val_dataloaders.append(val_dl4)

    return val_dataloaders

def intersect_and_union(preds: torch.Tensor, labels: torch.Tensor):
    #Calculate Intersection and Union for a batch of predictions and ground truths.
    num_classes = preds.size(1)
    
    pred_label= torch.argmax(preds, dim=0)  # Convert softmax scores to class predictions
    x = pred_label
    
    #import pdb; pdb.set_trace()
    mask = (labels != 255)

    pred_label = pred_label[mask]
    labels = labels[mask]
    intersect = pred_label[pred_label == labels]
    area_intersect = torch.histc(
        intersect.float(), bins=num_classes, min=0,
        max=num_classes - 1).cpu()
    area_pred_label = torch.histc(
        pred_label.float(), bins=num_classes, min=0,
        max=num_classes - 1).cpu()
    area_label = torch.histc(
        labels.float(), bins=num_classes, min=0,
        max=num_classes - 1).cpu()
    area_union = area_pred_label + area_label - area_intersect

    return area_intersect, area_union, area_pred_label, area_label

def main():
    data_loaders = load_dataloader()
    model = load_model(config_path)
    model = model.cuda()

    def get_results(dataloader):
        results = []
        for batch_ndx, sample in tqdm(enumerate(dataloader), total=len(dataloader)):  
            imgs = []
            for img in sample['inputs']:
                _, current_height, current_width = img.shape
                if img.dtype != torch.float32:
                    img = img.float()
                rgb_image = img[[2, 1, 0], :, :]
                
                mean = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1)
                std = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1)

                #Normalize the image
                normalized_image = (rgb_image - mean) / std
                imgs.append(normalized_image)

            img = torch.stack(imgs, dim = 0).cuda()
            batch_img_metas = [
            dict(
                ori_shape=img.shape[2:],
                img_shape=img.shape[2:],
                pad_shape=img.shape[2:],
                padding_size=[0, 0, 0, 0]
            )
            ] * img.shape[0]

            with torch.no_grad():
                logits = model.encode_decode(img, batch_img_metas)
                final_prediction = torch.softmax(logits, dim=1)

            gts = []
            for gt in sample['data_samples']:
                gt = gt._gt_sem_seg.data
                gt = gt.float()
                gts.append(gt)
        
            gt = torch.stack(gts, dim = 0).cuda()
            gt = F.interpolate(gt , size = (540, 960) , mode='nearest')
        
            #import pdb; pdb.set_trace()
            gt = gt.squeeze(0)
            gt = gt.to(dtype=torch.int64)
            gt = gt.squeeze(1)

            for i in range(final_prediction.shape[0]):

                area_intersect, area_union, area_pred_label, area_label = intersect_and_union(final_prediction[i], gt[i])
                areas = (area_intersect, area_union, area_pred_label, area_label)
                results.append(areas)
    
        results = tuple(zip(*results))
        assert len(results) == 4
        #import pdb; pdb.set_trace()
        #print(torch.sum(results[1][0]))
        total_area_intersect = torch.sum(torch.stack(results[0],dim = 0),dim = 0)
        total_area_union = torch.sum(torch.stack(results[1],dim = 0), dim = 0)
        total_iou = total_area_intersect / total_area_union
        miou = total_iou[~torch.isnan(total_iou)].mean()
        print("mIoU:", miou)
    
    get_results(data_loaders[3])

if __name__ == '__main__':
    main()
