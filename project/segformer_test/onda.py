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
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torchvision.utils import save_image
import os
from torch.cuda import amp
from copy import deepcopy
import wandb
import random

scaler = amp.GradScaler()
target_shape = (960, 540)
target_height = 540
target_width = 960

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

def load_models(config_path):
    pretrained = '/BS/DApt/work/project/segformer_test/pretrained/segformer_mit-b5_8x1_1024x1024_160k_cityscapes_20211206_072934-87a052ec.pth'    
    cfg = Config.fromfile(config_path)
    encdec = EncoderDecoder(cfg.model.backbone, cfg.model.decode_head)
    weights = torch.load(pretrained)

    weights = remove_prefix(weights['state_dict'])

    encdec.load_state_dict(weights)
    encdec = convert_syncbn_to_bn(encdec)

    teacher = encdec
    student = encdec
    
    teacher.cuda()
    student.cuda()

    student.train()
    teacher.eval()
    
    return teacher,student,weights

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


def intersect_and_union(preds: torch.Tensor, labels: torch.Tensor):
    #Calculate Intersection and Union for a batch of predictions and ground truths.
    num_classes = preds.size(1)
    
    pred_label= torch.argmax(preds, dim=0)  # Convert softmax scores to class predictions
    x = pred_label
    
    #import pdb; pdb.set_trace()
    mask = (labels != 255)

    pred_label = pred_label[mask]
    labels = labels[mask]

    #import pdb; pdb.set_trace()

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


def reg_loss(logits,gamma = 0.5):
    log_probs = F.log_softmax(logits, dim=1)
    mean_log_probs = torch.mean(log_probs, dim=(1, 2, 3))
    reg_loss = -gamma * mean_log_probs.mean()
    return reg_loss


def val_model(model,x_model,val_loader):
    val_loss = 0
    total_batches = len(val_loader)
    model.eval()

    results = []
    for batch_ndx, sample in enumerate(val_loader):
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
            logits = model.module.encode_decode(img, batch_img_metas)
            final_prediction = torch.softmax(logits, dim=1)
            t_pred = torch.argmax(final_prediction, dim=1)
            s_logits = x_model.module.encode_decode(img, batch_img_metas)

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

        loss = nn.CrossEntropyLoss()(s_logits,t_pred) + reg_loss(s_logits)
        val_loss += loss.item()
        
    results = tuple(zip(*results))
    assert len(results) == 4
    #import pdb; pdb.set_trace()
    #print(torch.sum(results[1][0]))
    total_area_intersect = torch.sum(torch.stack(results[0],dim = 0),dim = 0)
    total_area_union = torch.sum(torch.stack(results[1],dim = 0), dim = 0)
    total_iou = total_area_intersect / total_area_union
    miou = total_iou[~torch.isnan(total_iou)].mean()
    
    return val_loss/total_batches , miou

def update_ema_variables(t_model, s_model, iteration,alpha_teacher = 0.99):#, iteration):
    alpha_teacher = min(1 - 1 / (iteration + 1), alpha_teacher)
    for ema_param, param in zip(t_model.parameters(), s_model.parameters()):
        ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
        
        ema_param.requires_grad = False
        param.requires_grad = True
    return t_model

def configure_optimizer(model,weight_decay, learning_rate):
        param_dict = {pn: p for pn, p in model.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, eps=1e-8)
        return optimizer

def train_model(t_model, s_model, data_loader, val_dataloader, num_iterations, target_height, target_width, learning_rate, save_dir):

    '''wandb.init(project="Cottav1", config={
        "num_iterations": num_iterations,
        "target_height": target_height,
        "target_width": target_width,
        "learning_rate": learning_rate
    })
    config = wandb.config'''

    for param in t_model.parameters():
        param.requires_grad = False
    
    for param in s_model.parameters():
        param.requires_grad = True

    optimizer = configure_optimizer(s_model, learning_rate=learning_rate, weight_decay = 0.01)
    
    def lr_lambda(iteration):
        warmup_iters = 1000
        if iteration < warmup_iters:
            return iteration / warmup_iters
        return max(0.0, (num_iterations - iteration) / (num_iterations - warmup_iters))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    best_miou = 0.

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    iteration = 0

    for batch_ndx, sample in enumerate(data_loader):
        if iteration >= num_iterations:
            break 
        imgs = []
        for img in sample['inputs']:
            _, current_height, current_width = img.shape
            
            if img.dtype != torch.float32:
                img = img.float()
            
            # Padding or resizing
            if current_height < target_height or current_width < target_width:
                pad_height = max(0, target_height - current_height)
                pad_width = max(0, target_width - current_width)
                padding = (0, pad_width, 0, pad_height)  # (left, right, top, bottom)
                padded_image = F.pad(img, padding, "constant", 0)
            else:
                padded_image = img

            if padded_image.shape[1] > target_height or padded_image.shape[2] > target_width:
                # Resize if larger
                padded_image = F.interpolate(img.unsqueeze(0), size=(target_height, target_width), mode='bilinear', align_corners=False).squeeze(0)

            rgb_image = padded_image[[2, 1, 0], :, :] # bgr to rgb
 
            mean = torch.tensor([123.675, 116.28, 103.53], device=rgb_image.device).view(3, 1, 1)
            std = torch.tensor([58.395, 57.12, 57.375], device=rgb_image.device).view(3, 1, 1)

            # Normalize the image
            normalized_image = (rgb_image - mean) / std
            imgs.append(normalized_image)

        img = torch.stack(imgs, dim=0).cuda()

    #import pdb;pdb.set_trace()

        batch_img_metas = [
            dict(
                ori_shape=img.shape[2:],
                img_shape=img.shape[2:],
                pad_shape=img.shape[2:],
                padding_size=[0, 0, 0, 0]
            )
        ] * img.shape[0]

        with torch.no_grad():
            #Teacher pred
            x = t_model.module.encode_decode(img, batch_img_metas)
            outputs_ema = F.softmax(x, dim=1)
            
            
            # Get maximum confidence scores and pseudo-labels
            max_confidences, pseudo_label = torch.max(outputs_ema, dim=1)
            
            
            # Apply confidence thresholding
            confidence_threshold = 0.962
            mask = max_confidences >= confidence_threshold
            
            # Set low-confidence predictions to -1 (ignore index)
            pseudo_label[~mask] = -1
            import pdb;pdb.set_trace()
    
     
        with amp.autocast():
            s_logits = s_model.module.encode_decode(img, batch_img_metas)
            loss1 = nn.CrossEntropyLoss(ignore_index=-1,label_smoothing = 0.01)(s_logits,pseudo_label)
            loss2 = reg_loss(s_logits)
            loss = loss1+loss2
        

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        scheduler.step()

        # Teacher update
        t_model = update_ema_variables(t_model , s_model,iteration)

        current_loss = loss.item()
        #wandb.log({"train_loss": current_loss, "iteration": iteration})

        # calculate val_loss every 500th iteration
        if iteration % 200 == 0:
            val_loss,miou = val_model(s_model,t_model,val_dataloader)
            #wandb.log({"val_loss": val_loss,"val_miou": miou, "iteration": iteration})
            print(f"Iteration {iteration}, Train_Loss: {current_loss:.4f},  Val_Loss: {val_loss:.4f} ,Val_miou: {miou:.4f} ")
            
        if miou > best_miou:
            best_miou = miou
            model_path = os.path.join(save_dir, f'snow_20k_{best_miou:.4f}_{iteration}.pth')
            torch.save(t_model.state_dict(), model_path)
            print(f"Iteration {iteration}, New best validation miou: {best_miou:.4f}")
        
        if iteration % 20 == 0:
            print(f"Iteration {iteration}, Train_Loss: {current_loss:.4f}")
        
        iteration += 1

    print("Training completed. Best model saved with loss:", best_miou)
        
    #wandb.finish()

def main():
    config_path = '/BS/DApt/work/project/segformer_test/local_config/my_models/b5_fog_nlora.py' #Every domain same config
    target_height = 540
    target_width = 960

    data_loaders,val_dataloaders = load_dataloader()
    tmodel,smodel,weights = load_models(config_path)
    model_anchor = deepcopy(smodel)
    smodel = nn.DataParallel(smodel).cuda()
    tmodel = nn.DataParallel(tmodel).cuda()
    #model = model.cuda()

    train_model(
    t_model = tmodel,
    s_model = smodel,
    data_loader=data_loaders[0],
    val_dataloader = val_dataloaders[0],
    num_iterations=3000,
    target_height=target_height,
    target_width=target_width ,
    learning_rate=1e-5,
    save_dir="/BS/DApt/work/project/segformer_test/work_dirs/cottav1")

if __name__ == '__main__':
    main()

