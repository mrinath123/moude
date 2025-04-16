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
from tqdm import tqdm


target_shape = (960, 540)
target_height = 960
target_width = 540
mean = [123.675, 116.28, 103.53]
std = [58.395, 57.12, 57.375]

def remove_prefix(weights,prefix = 'backbone'):
    new_state_dict = {}
    for n, v in weights.items():
        if prefix in n:
            parts = n.split('.')
            new_n = '.'.join(parts[1:])
            new_state_dict[new_n] = v
        else:
            new_state_dict[n] = v  # Keep the key unchanged if it doesn't start with 'backbone.'
    return new_state_dict

#x = torch.randn(2,3,960, 540).cuda()

# Load IDASS models
def fog(x):
    #x -> N x C x H x W
    x = x.cuda()
    config_path = '/BS/DApt/work/project/segformer_test/local_config/my_models/b5_fog_IDASS.py'
    pretrained = '/BS/DApt/work/project/segformer_test/work_dirs/b5_fog_IDASS/teacher_state_dict.pth'
    cfg = Config.fromfile(config_path)
    encdec = EncoderDecoder(cfg.model.backbone , cfg.model.decode_head ,init_cfg=dict(type='Pretrained', checkpoint=pretrained))
    weights = torch.load(pretrained)
    encdec.load_state_dict(weights)
    encdec.eval()
    encdec.cuda()
    batch_img_metas = [
                    dict(
                        ori_shape=x.shape[2:],
                        img_shape=x.shape[2:],
                        pad_shape=x.shape[2:],
                        padding_size=[0, 0, 0, 0])
                ] * x.shape[0]
    #print(encdec)
    # Extract features and make predictions
    with torch.no_grad():  # Ensure no gradients are calculated
        x = encdec.encode_decode(x,batch_img_metas)
        ema_softmax = torch.softmax(x, dim=1)
        #x = encdec.extract_feat(x)
        #print(x[0].shape)
        return ema_softmax # N x C X H X W

def night(x):
    #x -> N x C x H x W
    config_path = '/BS/DApt/work/project/segformer_test/local_config/my_models/b5_night_IDASS.py'
    pretrained = '/BS/DApt/work/project/segformer_test/work_dirs/b5_night_IDASS/teacher_state_dict.pth'
    cfg = Config.fromfile(config_path)
    encdec = EncoderDecoder(cfg.model.backbone , cfg.model.decode_head ,init_cfg=dict(type='Pretrained', checkpoint=pretrained))
    weights = torch.load(pretrained)
    encdec.load_state_dict(weights)
    encdec.eval()
    encdec.cuda()
    batch_img_metas = [
                    dict(
                        ori_shape=x.shape[2:],
                        img_shape=x.shape[2:],
                        pad_shape=x.shape[2:],
                        padding_size=[0, 0, 0, 0])
                ] * x.shape[0]
    #print(encdec)
    # Extract features and make predictions
    with torch.no_grad():  # Ensure no gradients are calculated
        x = encdec.encode_decode(x,batch_img_metas)
        ema_softmax = torch.softmax(x, dim=1)
        #x = encdec.extract_feat(x)
        #print(x[0].shape)
        return ema_softmax
    
def rain(x):
    #x -> N x C x H x W
    config_path = '/BS/DApt/work/project/segformer_test/local_config/my_models/b5_rain_IDASS_v3.py'
    pretrained = '/BS/DApt/work/project/segformer_test/work_dirs/b5_rain_IDASS_v3/teacher_state_dict.pth'
    cfg = Config.fromfile(config_path)
    encdec = EncoderDecoder(cfg.model.backbone , cfg.model.decode_head ,init_cfg=dict(type='Pretrained', checkpoint=pretrained))
    weights = torch.load(pretrained)
    encdec.load_state_dict(weights)
    encdec.eval()
    encdec.cuda()
    batch_img_metas = [
                    dict(
                        ori_shape=x.shape[2:],
                        img_shape=x.shape[2:],
                        pad_shape=x.shape[2:],
                        padding_size=[0, 0, 0, 0])
                ] * x.shape[0]
    #print(encdec)
    # Extract features and make predictions
    with torch.no_grad():  # Ensure no gradients are calculated
        x = encdec.encode_decode(x,batch_img_metas)
        ema_softmax = torch.softmax(x, dim=1)
        #x = encdec.extract_feat(x)
        #print(x[0].shape)
        return ema_softmax
    
def snow(x):
    #x -> N x C x H x W
    config_path = '/BS/DApt/work/project/segformer_test/local_config/my_models/b5_snow_IDASS_v2.py'
    pretrained = '/BS/DApt/work/project/segformer_test/work_dirs/b5_snow_IDASS/teacher_state_dict.pth'
    cfg = Config.fromfile(config_path)
    #test= Config(dict(type='TestLoop',mode='slide', crop_size=(1024,1024), stride=(768,768)))
    encdec = EncoderDecoder(cfg.model.backbone , cfg.model.decode_head ,init_cfg=dict(type='Pretrained', checkpoint=pretrained))
    weights = torch.load(pretrained)
    encdec.load_state_dict(weights)
    encdec.eval()
    encdec.cuda()
    batch_img_metas = [
                    dict(
                        ori_shape=x.shape[2:],
                        img_shape=x.shape[2:],
                        pad_shape=x.shape[2:],
                        padding_size=[0, 0, 0, 0])
                ] * x.shape[0]
    #print(encdec)
    # Extract features and make predictions
    with torch.no_grad():  # Ensure no gradients are calculated
        x = encdec.encode_decode(x,batch_img_metas)
        ema_softmax = torch.softmax(x, dim=1)
        #x = encdec.extract_feat(x)
        #print(x[0].shape)
        '''temp = []
        y = encdec.slide_inference(x,batch_img_metas)
        z = encdec.postprocess_result(y)
        for imgs in z:
            ema_softmax = torch.softmax(imgs.seg_logits.data, dim=1)
            temp.append(ema_softmax)
        img = torch.stack(temp, dim = 0).cuda()'''
        return ema_softmax # N x C X H X W
    
# Load for Gate model

def load_models():
    #FOG
    config_path = '/BS/DApt/work/project/segformer_test/local_config/my_models/b5_fog_IDASS.py'
    cfg = Config.fromfile(config_path)
    backbone_fog = MODELS.build(cfg.model.backbone)

    check_point = '/BS/DApt/work/project/segformer_test/work_dirs/b5_fog_IDASS/teacher_state_dict.pth'
    weights = torch.load(check_point)
    weights = remove_prefix(weights)
    backbone_fog.load_state_dict(weights,strict = False)

    #NIGHT
    config_path = '/BS/DApt/work/project/segformer_test/local_config/my_models/b5_night_IDASS.py'
    cfg = Config.fromfile(config_path)
    backbone_night = MODELS.build(cfg.model.backbone)

    check_point = '/BS/DApt/work/project/segformer_test/work_dirs/b5_night_IDASS/teacher_state_dict.pth'
    weights = torch.load(check_point)
    weights = remove_prefix(weights)
    backbone_night.load_state_dict(weights,strict = False)

    #RAIN
    config_path = '/BS/DApt/work/project/segformer_test/local_config/my_models/b5_rain_IDASS_v3.py'
    cfg = Config.fromfile(config_path)
    backbone_rain = MODELS.build(cfg.model.backbone)

    check_point = '/BS/DApt/work/project/segformer_test/work_dirs/b5_rain_IDASS_v3/teacher_state_dict.pth'
    weights = torch.load(check_point)
    weights = remove_prefix(weights)
    backbone_rain.load_state_dict(weights,strict = False)

    #SNOW
    config_path = '/BS/DApt/work/project/segformer_test/local_config/my_models/b5_snow_IDASS_v2.py'
    cfg = Config.fromfile(config_path)
    backbone_snow = MODELS.build(cfg.model.backbone)

    check_point = '/BS/DApt/work/project/segformer_test/work_dirs/b5_snow_IDASS/teacher_state_dict.pth'
    weights = torch.load(check_point)
    weights = remove_prefix(weights)
    backbone_snow.load_state_dict(weights,strict = False)

    backbone_fog.cuda()
    backbone_night.cuda()
    backbone_rain.cuda()
    backbone_snow.cuda()
    
    backbone_fog.eval()
    backbone_night.eval()
    backbone_rain.eval()
    backbone_snow.eval()

    return backbone_fog,backbone_night,backbone_rain,backbone_snow

def load_dataloader():
    dataloaders = []
    #Fog
    dataset_config = '/BS/DApt/work/project/segformer_test/local_config/_base_/datasets/acdc_fog.py'
    cfg = Config.fromfile(dataset_config)
    dl1 = Runner.build_dataloader(cfg.val_dataloader)
    dataloaders.append(dl1)
    #Night
    dataset_config = '/BS/DApt/work/project/segformer_test/local_config/_base_/datasets/acdc_night.py'
    cfg = Config.fromfile(dataset_config)
    dl2 = Runner.build_dataloader(cfg.val_dataloader)
    dataloaders.append(dl2)
    #Rain
    dataset_config = '/BS/DApt/work/project/segformer_test/local_config/_base_/datasets/acdc_rain.py'
    cfg = Config.fromfile(dataset_config)
    dl3 = Runner.build_dataloader(cfg.val_dataloader)
    dataloaders.append(dl3)
    #Snow
    dataset_config = '/BS/DApt/work/project/segformer_test/local_config/_base_/datasets/acdc_snow.py'
    cfg = Config.fromfile(dataset_config)
    dl4 = Runner.build_dataloader(cfg.val_dataloader)
    dataloaders.append(dl4)

    return dataloaders

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
    
def main():
    data_loaders = load_dataloader()
    
    def run_infer(mode):
        if (mode == 4):
            dataloader = [data_loaders[0],data_loaders[1],data_loaders[2],data_loaders[3] ]
        elif (mode == 3):
            dataloader = [data_loaders[0],data_loaders[1],data_loaders[2]]
        else:
            dataloader = [data_loaders[0],data_loaders[1]]
        
        model_fog, model_night, model_rain, model_snow = load_models()
        for dl in dataloader:
            results = []
            for batch_ndx, sample in tqdm(enumerate(dl), total=len(dl)):
                imgs = []
                for img in sample['inputs']:
                    _, current_height, current_width = img.shape
                    if img.dtype != torch.float32:
                        img = img.float()
                    '''# Padding or resizing
                    if current_height < target_height or current_width < target_width:
                        pad_height = max(0, target_height - current_height)
                        pad_width = max(0, target_width - current_width)
                        padding = (0, pad_width, 0, pad_height)  # (left, right, top, bottom)
                        padded_image = F.pad(img, padding, "constant", 0)
                    else:
                        padded_image = img
            
                    if padded_image.shape[1] > target_height or padded_image.shape[2] > target_width:
                        # Resize if larger
                        padded_image = F.interpolate(img.unsqueeze(0), size=(target_height, target_width), mode='bilinear', align_corners=False).squeeze(0)'''
                    
                    rgb_image = img[[2, 1, 0], :, :]
                    mean = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1)
                    std = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1)

                    #Normalize the image
                    normalized_image = (rgb_image - mean) / std
                    imgs.append(normalized_image)

                img = torch.stack(imgs, dim = 0).cuda()

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

                embeds = []

                if (mode == 4):
                    #Step 1: IDASS
                    f1 = fog(img)      
                    f2 = night(img)    
                    f3 = rain(img)     
                    f4 = snow(img)     # torch.Size([batch, 19, 960, 540])
                    
                    e1 = model_fog(img)[3]
                    e2 = model_night(img)[3]
                    e3 = model_rain(img)[3]
                    e4 = model_snow(img)[3]
                    
                    embeds.append(e1)
                    embeds.append(e2)
                    embeds.append(e3)
                    embeds.append(e4)

                    embeds = torch.cat(embeds, dim = 0)    
                    pred = load_gate(embeds)
                    pred = torch.softmax(pred.detach(), dim=1)

                    pred = pred.view(4, img.shape[0], -1)  # Shape: [4, batch_size, num_classes]

                    # Extract individual condition probabilities
                    fog_prob = pred[0]  # Shape: [batch_size, num_classes]
                    night_prob = pred[1]  # Shape: [batch_size, num_classes]
                    rain_prob = pred[2]  # Shape: [batch_size, num_classes]
                    snow_prob = pred[3]

                    # Compute mean probabilities for each class
                    mean_prob_0 = (fog_prob[:, 0] + night_prob[:, 0] + rain_prob[:, 0] +snow_prob[:, 0] ) / 4  # Shape: [batch_size]
                    mean_prob_1 = (fog_prob[:, 1] + night_prob[:, 1] + rain_prob[:, 1] +snow_prob[:, 1]) / 4  # Shape: [batch_size]
                    mean_prob_2 = (fog_prob[:, 2] + night_prob[:, 2] + rain_prob[:, 2]+snow_prob[:, 2]) / 4  # Shape: [batch_size]
                    mean_prob_3 = (fog_prob[:, 3] + night_prob[:, 3] + rain_prob[:, 3]+snow_prob[:, 3]) / 4  # Shape: [batch_size]

                    # Reshape mean probabilities to match the shape of f1, f2, and f3
                    mean_prob_0 = mean_prob_0.view(img.shape[0], 1, 1, 1)  # Shape: [batch_size, 1, 1, 1]
                    mean_prob_1 = mean_prob_1.view(img.shape[0], 1, 1, 1)  # Shape: [batch_size, 1, 1, 1]
                    mean_prob_2 = mean_prob_2.view(img.shape[0], 1, 1, 1)  # Shape: [batch_size, 1, 1, 1]
                    mean_prob_3 = mean_prob_3.view(img.shape[0], 1, 1, 1)  # Shape: [batch_size, 1, 1, 1]

                    # Apply the weighted sum to each environmental effect
                    final_log = f1 * mean_prob_0 + f2 * mean_prob_1 + f3 * mean_prob_2 + f4 * mean_prob_3 

                    gt = gt.squeeze(1)
                    for i in range(f1.shape[0]):
                        area_intersect, area_union, area_pred_label, area_label = intersect_and_union(final_log[i], gt[i])
                        areas = (area_intersect, area_union, area_pred_label, area_label)
                        results.append(areas)
                    
                elif (mode == 3):
                    #Step 1: IDASS
                    f1 = fog(img)      
                    f2 = night(img)    
                    f3 = rain(img)     
                    
                    # Step 2: Gate networks
                    '''gate_fog_output = torch.softmax(load_gate(model_fog(img)[3]) , dim=1)   # torch.Size([batch, 4])
                    gate_night_output = torch.softmax(load_gate(model_night(img)[3]), dim=1)
                    gate_rain_output = torch.softmax(load_gate(model_rain(img)[3])  , dim=1)


                    # Step 3: Compute the mean 
                    mean_o1 = (gate_fog_output[:, 0] + gate_night_output[:, 0] + gate_rain_output[:, 0] )/3.
                    mean_o2 = (gate_fog_output[:, 1] + gate_night_output[:, 1] + gate_rain_output[:, 1])/3.
                    mean_o3 = (gate_fog_output[:, 2] + gate_night_output[:, 2] + gate_rain_output[:, 2] )/3.


                    m_o1 = mean_o1.view(img.shape[0],1,1,1)
                    m_o2 = mean_o2.view(img.shape[0],1,1,1)
                    m_o3 = mean_o3.view(img.shape[0],1,1,1)
                
                    # Step 4: Apply the weighted sum to each environmental effect
                    weighted_f1 = f1 * m_o1
                    weighted_f2 = f2 * m_o2
                    weighted_f3 = f3 * m_o3

                    # Step 5: Sum the weighted environmental effects to obtain the final prediction
                    final_prediction = weighted_f1 + weighted_f2 + weighted_f3 '''

                    
                    e1 = model_fog(img)[3]
                    e2 = model_night(img)[3]
                    e3 = model_rain(img)[3]
                    
                    embeds.append(e1)
                    embeds.append(e2)
                    embeds.append(e3)

                    embeds = torch.cat(embeds, dim = 0)    
                    pred = load_gate(embeds)
                    pred = torch.softmax(pred.detach(), dim=1)

                    pred = pred.view(3, img.shape[0], -1)  # Shape: [3, batch_size, num_classes]

                    # Extract individual condition probabilities
                    fog_prob = pred[0]  # Shape: [batch_size, num_classes]
                    night_prob = pred[1]  # Shape: [batch_size, num_classes]
                    rain_prob = pred[2]  # Shape: [batch_size, num_classes]

                    # Compute mean probabilities for each class
                    mean_prob_0 = (fog_prob[:, 0] + night_prob[:, 0] + rain_prob[:, 0]) / 3  # Shape: [batch_size]
                    mean_prob_1 = (fog_prob[:, 1] + night_prob[:, 1] + rain_prob[:, 1]) / 3  # Shape: [batch_size]
                    mean_prob_2 = (fog_prob[:, 2] + night_prob[:, 2] + rain_prob[:, 2]) / 3  # Shape: [batch_size]

                    # Reshape mean probabilities to match the shape of f1, f2, and f3
                    mean_prob_0 = mean_prob_0.view(img.shape[0], 1, 1, 1)  # Shape: [batch_size, 1, 1, 1]
                    mean_prob_1 = mean_prob_1.view(img.shape[0], 1, 1, 1)  # Shape: [batch_size, 1, 1, 1]
                    mean_prob_2 = mean_prob_2.view(img.shape[0], 1, 1, 1)  # Shape: [batch_size, 1, 1, 1]

                    # Apply the weighted sum to each environmental effect
                    final_log = f1 * mean_prob_0 + f2 * mean_prob_1 + f3 * mean_prob_2 
                    
                    
                    #import pdb;pdb.set_trace()

                    gt = gt.squeeze(1)
                    for i in range(f1.shape[0]):
                        area_intersect, area_union, area_pred_label, area_label = intersect_and_union(final_log[i], gt[i])
                        areas = (area_intersect, area_union, area_pred_label, area_label)
                        results.append(areas)

                else:
                    #Step 1: IDASS
                    f1 = fog(img)      
                    f2 = night(img)  

                    e1 = model_fog(img)[3]
                    e2 = model_night(img)[3]
                    
                    embeds.append(e1)
                    embeds.append(e2)

                    embeds = torch.cat(embeds, dim = 0)    
                    pred = load_gate(embeds)
                    pred = torch.softmax(pred.detach(), dim=1)

                    pred = pred.view(2, img.shape[0], -1)  # Shape: [3, batch_size, num_classes]
                    import pdb;pdb.set_trace()

                    # Extract individual condition probabilities
                    fog_prob = pred[0]  # Shape: [batch_size, num_classes]
                    night_prob = pred[1]  # Shape: [batch_size, num_classes]
          

                    # Compute mean probabilities for each class
                    mean_prob_0 = (fog_prob[:, 0] + night_prob[:, 0] ) / 2  # Shape: [batch_size]
                    mean_prob_1 = (fog_prob[:, 1] + night_prob[:, 1] ) / 2  # Shape: [batch_size]

                    # Reshape mean probabilities to match the shape of f1, f2, and f3
                    mean_prob_0 = mean_prob_0.view(img.shape[0], 1, 1, 1)  # Shape: [batch_size, 1, 1, 1]
                    mean_prob_1 = mean_prob_1.view(img.shape[0], 1, 1, 1)  # Shape: [batch_size, 1, 1, 1]

                    # Apply the weighted sum to each environmental effect
                    final_log = f1 * mean_prob_0 + f2 * mean_prob_1  

                    #import pdb;pdb.set_trace()
                    
                    gt = gt.squeeze(1)
                    for i in range(f1.shape[0]):
                        area_intersect, area_union, area_pred_label, area_label = intersect_and_union(final_log[i], gt[i])
                        areas = (area_intersect, area_union, area_pred_label, area_label)
                        results.append(areas)

            results = tuple(zip(*results))
            assert len(results) == 4
            #print(torch.sum(results[1][0]))
            total_area_intersect = torch.sum(torch.stack(results[0],dim = 0),dim = 0)
            total_area_union = torch.sum(torch.stack(results[1],dim = 0), dim = 0)
            total_iou = total_area_intersect / total_area_union
            miou = total_iou[~torch.isnan(total_iou)].mean()
            print("mIoU:", miou)

    class GatingNet(nn.Module):
        def __init__(self):
            super(GatingNet, self).__init__()
            #self.linear = nn.Linear(261120, 2) # 261120 -> after squeezing he output from 4th block of backbone
            self.conv = nn.Conv2d(512,1,(5,5))
            self.relu = nn.ReLU()
            self.pool = nn.MaxPool2d(2)
            self.linear = nn.Linear(6*13,2)

        def forward(self, x):
            x = self.pool(self.relu(self.conv(x)))
            x = x.view(x.size(0), -1)
            x = self.linear(x)
            return x

    def load_gate(x):
        x = x.cuda()
        pretrained = '/BS/DApt/work/project/segformer_test/work_dirs/gate/save_gate_model_2_classes_v11.pth'
        gate = GatingNet()
        weights = torch.load(pretrained)
        gate.load_state_dict(weights,strict= False)
        gate.eval()
        gate.cuda()

        with torch.no_grad():  # Ensure no gradients are calculated
            x = gate(x)
            return x

    run_infer(mode =2)
    
if __name__ == '__main__':
    main()




