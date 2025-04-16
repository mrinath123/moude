import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib
import cv2
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from mmseg.registry import MODELS
from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.model import BaseModule
from mmengine.registry import init_default_scope
init_default_scope('mmseg')
from mmseg.models.segmentors import EncoderDecoder
from mmseg.datasets.acdc import ACDCDataset
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.utils import save_image
import os
from torch.cuda import amp
from tqdm import tqdm
import torch

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

def load_model():
    config_path = '/BS/DApt/work/project/segformer_test/local_config/my_models/b5_fog_IDASS.py'
    
    #pretrained = '/BS/DApt/work/project/segformer_test/work_dirs/oracle_gt_weigtedloss/fog_oraclev10_8x8_5/teacher.pth'  # weighted loss KL w/GT global
    pretrained = '/BS/DApt/work/project/segformer_test/work_dirs/fog_oraclev10_8x8_2.5/teacher.pth'
    
    
    cfg = Config.fromfile(config_path)
    print(cfg)
    encdec = EncoderDecoder(cfg.model.backbone, cfg.model.decode_head)
    weights = torch.load(pretrained)
  
    encdec.load_state_dict(weights)
    encdec = convert_syncbn_to_bn(encdec)

    encdec.cuda()
    
    encdec.requires_grad_(False) # Set requires_grad False
    
    for m in encdec.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True) # Grad true for norm layers
            
    return encdec

cityscapes_palette = {
    0:  (128, 64, 128),  # Road
    1:  (244, 35, 232),  # Sidewalk
    2:  (70, 70, 70),    # Building
    3:  (102, 102, 156), # Wall
    4:  (190, 153, 153), # Fence
    5:  (153, 153, 153), # Pole
    6:  (250, 170, 30),  # Traffic Light
    7:  (220, 220, 0),   # Traffic Sign
    8:  (107, 142, 35),  # Vegetation
    9:  (152, 251, 152), # Terrain
    10: (70, 130, 180),  # Sky
    11: (220, 20, 60),   # Person
    12: (255, 0, 0),     # Rider
    13: (0, 0, 142),     # Car
    14: (0, 0, 70),      # Truck
    15: (0, 60, 100),    # Bus
    16: (0, 80, 100),    # Train
    17: (0, 0, 230),     # Motorcycle
    18: (119, 11, 32),   # Bicycle
    255: (0, 0, 0)       # Masked (255)
}

def plot_preds(model, val_loader):
    val_loss = 0
    total_batches = len(val_loader)
    model.eval()

    # Define the class names corresponding to class IDs
    class_names = [
        'Road', 'Sidewalk', 'Building', 'Wall', 'Fence', 'Pole',
        'Traffic Light', 'Traffic Sign', 'Vegetation', 'Terrain',
        'Sky', 'Person', 'Rider', 'Car', 'Truck', 'Bus',
        'Train', 'Motorcycle', 'Bicycle'
    ]

    # Normalize the Cityscapes palette colors to [0, 1] range for matplotlib
    color_map = {k: np.array(v) / 255.0 for k, v in cityscapes_palette.items()}

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

            # Normalize the image
            normalized_image = (rgb_image - mean) / std
            imgs.append(normalized_image)

        img = torch.stack(imgs, dim=0).cuda()
         
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
            img_path = gt.img_path
            original_img = cv2.imread(img_path)
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            gt = gt._gt_sem_seg.data
            gt = gt.float()
            gts.append(gt)
        
        gt = torch.stack(gts, dim=0).cuda()

        # Resize ground truth to match the prediction size
        preds = final_prediction[1]
        gt_resized = F.interpolate(gt, size=preds.shape[1:], mode='nearest')
        
        gt_resized = gt_resized.squeeze(0).to(dtype=torch.int64).squeeze(1)
        break
    
    preds = final_prediction[1]
    pred_label = torch.argmax(preds, dim=0)

    pred_label_np = pred_label.cpu().numpy()
    gt_np = gt_resized[1].cpu().numpy()

    # Function to map labels to colors using the Cityscapes palette
    def label_to_color(labels):
        rgba = np.array([color_map[l] for l in labels.flat]).reshape(labels.shape + (3,))
        return rgba

    # Plot the images
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 10))

    ax1.imshow(original_img)
    ax1.set_title('Original Image')
    ax1.axis('off')

    ax2.imshow(label_to_color(gt_np))
    ax2.set_title('Ground Truth')
    ax2.axis('off')

    ax3.imshow(label_to_color(pred_label_np))
    ax3.set_title('Prediction')
    ax3.axis('off')

    plt.tight_layout()

    # Save the plot
    #plt.savefig("segmentation_results.png", bbox_inches="tight")
    plt.show()
    plt.savefig("segmentation_results_moude.png", bbox_inches="tight")

    # Print statistics
    print(f"Total pixels: {pred_label_np.size}")
    print(f"Masked pixels: {np.sum(gt_np == 255)}")
    print(f"Comparable pixels: {pred_label_np.size - np.sum(gt_np == 255)}")
    print(f"Correct pixels: {np.sum((pred_label_np == gt_np) & (gt_np != 255))}")
    print(f"Incorrect pixels: {np.sum((pred_label_np != gt_np) & (gt_np != 255))}")
    print(f"Accuracy: {np.sum((pred_label_np == gt_np) & (gt_np != 255)) / (pred_label_np.size - np.sum(gt_np == 255)):.2%}")


config_path = '/BS/DApt/work/project/segformer_test/local_config/my_models/b5_rain_nlora.py'
model = load_model()
model = model.cuda()
dataloaders,val_dataloaders = load_dataloader()
plot_preds(model,val_dataloaders[2])
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
def create_dirs(base_dir):
    os.makedirs(os.path.join(base_dir, 'gt'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'model_pred'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'cotta_pred'), exist_ok=True)

# Function to map labels to colors using the Cityscapes palette
def label_to_color(labels, color_map):
    height, width = labels.shape  # We assume labels is 2D (height, width)
    # Map labels to colors using the color map
    rgba = np.array([color_map[l] for l in labels.flat]).reshape((height, width, 3))  # Reshape to (height, width, 3)
    return rgba

def save_predictions(img_path, gt, model_pred, cotta_pred, img, original_img, idx, color_map, save_dir):
    # Map labels to colors
    gt_color = label_to_color(gt.cpu().numpy(), color_map)
    model_pred_color = label_to_color(model_pred.cpu().numpy(), color_map)
    cotta_pred_color = label_to_color(cotta_pred.cpu().numpy(), color_map)

    # Convert the input image to RGB and save it
    img = img.cpu().numpy().transpose(1, 2, 0)  # Convert from (C, H, W) to (H, W, C)
    img = np.clip(img, 0, 255).astype(np.uint8)
    
    # Ensure directories exist
    os.makedirs(os.path.join(save_dir, 'gt'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'model_pred'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'cotta_pred'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'original_images'), exist_ok=True)  # Directory for original images

    # Save the images
    plt.imsave(os.path.join(save_dir, 'gt', f'{idx:04d}_gt.png'), gt_color)
    plt.imsave(os.path.join(save_dir, 'model_pred', f'{idx:04d}_model_pred.png'), model_pred_color)
    plt.imsave(os.path.join(save_dir, 'cotta_pred', f'{idx:04d}_cotta_pred.png'), cotta_pred_color)
    plt.imsave(os.path.join(save_dir, 'original_images', f'{idx:04d}_original.png'), original_img)  # Save the original image


def process_and_save_images(model, cotta_model, dataloader, color_map, save_dir):
    model.eval()
    cotta_model.eval()

    for idx, sample in enumerate(tqdm(dataloader, desc="Processing Images")):
        imgs = []
        gt_list = []
        original_images = []
        for img in sample['inputs']:
            _, current_height, current_width = img.shape
            if img.dtype != torch.float32:
                img = img.float()
            rgb_image = img[[2, 1, 0], :, :]  # Convert from BGR to RGB
            
            mean = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1)
            std = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1)

            # Normalize the image
            normalized_image = (rgb_image - mean) / std
            imgs.append(normalized_image)
        
        for gts in sample['data_samples']:
            gt = gts._gt_sem_seg.data  # Access the gt for each image in the batch
            gt_list.append(gt.unsqueeze(0))
        paths = [gts.img_path for gts in sample['data_samples']]

        for p in paths:
             original_img = cv2.imread(p)
             original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB) 
             original_images.append(original_img)

        img = torch.stack(imgs, dim=0).cuda()
        gt = torch.cat(gt_list, dim=0).cuda()
        #ori = torch.cat(original_images, dim=0).cuda()

        batch_img_metas = [
            dict(
                ori_shape=img.shape[2:],
                img_shape=img.shape[2:],
                pad_shape=img.shape[2:],
                padding_size=[0, 0, 0, 0]
            )
        ] * img.shape[0]

        with torch.no_grad():
            model_logits = model.encode_decode(img, batch_img_metas)
            cotta_logits = cotta_model.encode_decode(img, batch_img_metas)

            # Get model predictions and cotta predictions (argmax on logits)
            model_pred = torch.argmax(F.softmax(model_logits, dim=1), dim=1).squeeze(0)
            cotta_pred = torch.argmax(F.softmax(cotta_logits, dim=1), dim=1).squeeze(0)

        preds = model_pred  # Assuming you're working with model predictions
        gt_resized = F.interpolate(gt.float(), size=preds.shape[1:], mode='nearest').squeeze(1).to(dtype=torch.int64)

        
        # Save predictions, ground truth, and original image
        for i in range(gt_resized.shape[0]):  # Iterate over each image in the batch
            #import pdb; pdb.set_trace()
            save_predictions(
                sample['data_samples'][i].img_path, 
                gt_resized[i], model_pred[i], cotta_pred[i], 
                img[i], original_images[i], idx, color_map, save_dir
            )
# Main executionconda a
if __name__ == "__main__":
    save_dir = "segmentation_results/fog"
    create_dirs(save_dir)

    color_map = {k: np.array(v) / 255.0 for k, v in cityscapes_palette.items()}
    
    model = load_model()
    cotta_model = load_cotta_model()
    model.cuda()
    cotta_model.cuda()
    
    _, val_dataloaders = load_dataloader()
    dataloader = val_dataloaders[0]  # Using the Rain validation set as an example

    process_and_save_images(model, cotta_model, dataloader, color_map, save_dir)