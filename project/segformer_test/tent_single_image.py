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
import cv2
import matplotlib.pyplot as plt

def save_image(image_tensor, iteration, directory='images'):
    # Ensure the directory exists
    os.makedirs(directory, exist_ok=True)
    filename = os.path.join(directory, f'predicted_segmentation_map_iter_{iteration}.png')
    
    image = image_tensor.cpu().numpy()
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()

scaler = amp.GradScaler()

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

def load_model(config_path):
    pretrained = '/BS/DApt/work/project/segformer_test/pretrained/segformer_mit-b5_8x1_1024x1024_160k_cityscapes_20211206_072934-87a052ec.pth'    
    cfg = Config.fromfile(config_path)
    encdec = EncoderDecoder(cfg.model.backbone, cfg.model.decode_head)
    weights = torch.load(pretrained)
  
    weights = remove_prefix(weights['state_dict'])
    '''for n in weights.keys():s
        print(n)'''
    encdec.load_state_dict(weights)
    encdec = convert_syncbn_to_bn(encdec)
    encdec.cuda()
    #print(encdec)

    for name, param in encdec.named_parameters():
        if 'bn' in name:  # This assumes BatchNorm layers have 'bn' in their name
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    trainable_params = sum(p.numel() for p in encdec.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in encdec.parameters() if not p.requires_grad)
    
    print(f"Number of frozen parameters: {frozen_params}")
    print(f"Number of trainable parameters: {trainable_params}")
            
    return encdec

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

def loss_fn(logits):
    #import pdb;pdb.set_trace()
    probabilities = F.softmax(logits, dim=1)
    log_probabilities = F.log_softmax(logits, dim=1)
    entropy = - (probabilities * log_probabilities).sum(dim=1)
    mean_entropy = entropy.sum()
    return mean_entropy

def intersect_and_union(preds: torch.Tensor, labels: torch.Tensor):
    #Calculate Intersection and Union for a batch of predictions and ground truths.
    num_classes = preds.size(1)
    preds = torch.squeeze(preds,0)
    labels = torch.squeeze(labels,0)
    
    pred_label= torch.argmax(preds, dim=0)  # Convert softmax scores to class predictions

    x = pred_label
    
    mask = (labels != 255)

    pred_label = pred_label[mask]
    labels = labels[mask]

    # Calculate matching pixels
    matching_pixels = (pred_label == labels).sum().item()
    total_pixels = labels.numel()
    
    # Calculate percentage
    matching_percentage = (matching_pixels / total_pixels) * 100
    print(f'matching percentage {matching_percentage}')

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

def val_model(model,img_path,gts,iteration):
    
    model.eval()

    img = cv2.imread(gts)
    img = img[: ,:,0] # only taking 1 channel
    resized_image = cv2.resize(img, (960, 540)) 
    resized_image = torch.tensor(resized_image).unsqueeze(0)

    gt = resized_image.cuda()

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized_image = cv2.resize(img, (960, 540))
    resized_image = resized_image.astype(np.float32) 
    resized_image = torch.tensor(resized_image).permute(2, 0, 1)  
    mean = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1)
    std = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1)
    normalized_image = (resized_image - mean) / std

    full_image = torch.unsqueeze(normalized_image,0)

    batch_img_metas = [
        dict(
            ori_shape=full_image.shape[2:],
            img_shape=full_image.shape[2:],
            pad_shape=full_image.shape[2:],
            padding_size=[0, 0, 0, 0]
        )
    ] * full_image.shape[0]

    with torch.no_grad():
        logits = model._forward(full_image.cuda(), batch_img_metas)
        logits =F.interpolate(logits , size = (540,960) , mode='nearest')
        final_prediction = torch.softmax(logits, dim=1)
    
    save_image(torch.argmax(final_prediction, dim=1).squeeze(), iteration)

    # Debug: Monitor the softmax output distributions
    print(f'Softmax Output Distribution: {final_prediction.mean().item()}')
    
    results = []

    area_intersect, area_union, area_pred_label, area_label = intersect_and_union(final_prediction, gt)
    areas = (area_intersect, area_union, area_pred_label, area_label)
    results.append(areas)
    
    #import pdb; pdb.set_trace()
    results = tuple(zip(*results))
    assert len(results) == 4
    
    #print(torch.sum(results[1][0]))
    total_area_intersect = torch.sum(torch.stack(results[0],dim = 0),dim = 0)
    total_area_union = torch.sum(torch.stack(results[1],dim = 0), dim = 0)
    total_iou = total_area_intersect / total_area_union
    miou = total_iou[~torch.isnan(total_iou)].mean()
    
    return miou
    

def train_model(model, img_path,gts , num_iterations, target_height, target_width, learning_rate, save_dir):

    model.train()
    print('Fog')
    '''wandb.init(project="TentTrainv4", config={
        "num_iterations": num_iterations,
        "target_height": target_height,
        "target_width": target_width,
        "learning_rate": learning_rate
    })'''
    config = wandb.config

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate,weight_decay = 0.01)
    best_miou = 0.

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    iteration = 0
    
 
    while iteration <= num_iterations:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized_image = cv2.resize(img, (960, 540))
        resized_image = resized_image.astype(np.float32) 
        resized_image = torch.tensor(resized_image).permute(2, 0, 1)  
        mean = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1)
        std = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1)
        normalized_image = (resized_image - mean) / std

        full_image = torch.unsqueeze(normalized_image,0)

        batch_img_metas = [
            dict(
                ori_shape=full_image.shape[2:],
                img_shape=full_image.shape[2:],
                pad_shape=full_image.shape[2:],
                padding_size=[0, 0, 0, 0]
            )
        ] * full_image.shape[0]

        with amp.autocast():
            logits = model._forward(full_image.cuda(), batch_img_metas)
            logits =F.interpolate(logits , size = (540,960) , mode='nearest')
            #import pdb;pdb.set_trace()
            loss = loss_fn(logits)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        current_loss = loss.item()
        #wandb.log({"train_loss": current_loss, "iteration": iteration})

        # calculate val_loss every 500th iteration
        if iteration % 100 == 0:
            miou = val_model(model,img_path, gts,iteration)
            #wandb.log({"val_loss": val_loss,"val_miou": miou, "iteration": iteration})
            print(f"Iteration {iteration}, Train_Loss: {current_loss:.4f} ,Val_miou: {miou:.4f} ")
            best_miou = miou
            model_path = os.path.join(save_dir, f'fog_v2_20k_{best_miou:.4f}_{iteration}.pth')
            torch.save(model.state_dict(), model_path)
            print(f"Iteration {iteration}, New best validation miou: {best_miou:.4f}")
        
        if iteration % 50 == 0:
            print(f"Iteration {iteration}, Train_Loss: {current_loss:.4f}")
        
        iteration += 1

def main():
    config_path = '/BS/DApt/work/project/segformer_test/local_config/my_models/b5_fog_nlora.py' #Every domain same config
    target_height = 540
    target_width = 960

    data_loaders,val_dataloaders = load_dataloader()
    dataset = val_dataloaders[0].dataset
    img = dataset[1]['data_samples'].img_path
    gt = dataset[1]['data_samples'].seg_map_path
    model = load_model(config_path)
    #model = nn.DataParallel(model)
    model = model.cuda()

    train_model(
    model = model,
    img_path = img,
    gts = gt,
    num_iterations=20000,
    target_height=target_height,
    target_width=target_width ,
    learning_rate=0.00001,
    save_dir="/BS/DApt/work/project/segformer_test/work_dirs/tentv5")

if __name__ == '__main__':
    main()

