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
import random

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

class ReplayBuffer(Dataset):
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []

    def add_dict(self, data):
        if len(self.buffer) >= self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append(data)

    def sample_images(self, dataloader, num_samples):
        dataset = dataloader.dataset
        # Randomly sample images from the dataloader
        indices = random.sample(range(len(dataset)), num_samples)
        for idx in indices:
            data = dataset[idx]
            img = data['inputs']
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
            
            rgb_image = padded_image[[2, 1, 0], :, :]
            
            mean = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1)
            std = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1)

            #Normalize the image
            normalized_image = (rgb_image - mean) / std
            data['inputs'] = normalized_image
            self.add_dict(data)
            
    def __len__(self):
        return len(self.buffer)
    
    def __getitem__(self, idx):
        return self.buffer[idx]
    

def custom_collate(batch):
    collated_batch = {'inputs': [], 'meta_info': []}
    
    for item in batch:
        collated_batch['inputs'].append(item['inputs'])

        meta_info = {
            'scale_factor': item['data_samples'].metainfo['scale_factor'],
            'ori_shape': item['data_samples'].metainfo['ori_shape'],
            'img_path': item['data_samples'].metainfo['img_path'],
        }
        collated_batch['meta_info'].append(meta_info)
    collated_batch['inputs'] = torch.stack(collated_batch['inputs'], dim=0)
    
    return collated_batch
    
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
        results = []

        if (mode == 2):
            replay_buffer = ReplayBuffer(buffer_size=500)
            replay_buffer.sample_images(data_loaders[0], 50)
            replay_buffer.sample_images(data_loaders[1], 50)

            replay_loader = DataLoader(replay_buffer, batch_size=8, shuffle=True,collate_fn=custom_collate)

            for batch_ndx, sample in enumerate(replay_loader):
                img = sample['inputs'].cuda()
                img_metas = sample['meta_info']

                embeds = []
                with torch.no_grad():
                    fog_embeds = model_fog(img)[3]
                    night_embeds = model_night(img)[3]
                    embeds.append(fog_embeds)
                    embeds.append(night_embeds)
            
                embeds = torch.cat(embeds, dim = 0)
                output_logits = load_gate(embeds)

                labels = torch.zeros(img.size(0)).cuda().long()
                for i in range(img.size(0)):
                    if img_metas[i]['img_path'].find("/fog/") != -1:
                        labels[i] = 0
                    elif img_metas[i]['img_path'].find("/night/") != -1:
                        labels[i] = 1

                labels = torch.cat([labels] * 2, dim=0)

                probabilities = torch.softmax(output_logits, dim=1)
                predictions = torch.argmax(probabilities, dim=1)
                correct_predictions = (predictions == labels).sum().item()
                total_predictions = labels.size(0)
                accuracy = correct_predictions / total_predictions
                results.append(accuracy)
        
        if (mode == 3):
            replay_buffer = ReplayBuffer(buffer_size=500)
            replay_buffer.sample_images(data_loaders[0], 50)
            replay_buffer.sample_images(data_loaders[1], 50)
            replay_buffer.sample_images(data_loaders[2], 50)

            replay_loader = DataLoader(replay_buffer, batch_size=8, shuffle=True,collate_fn=custom_collate)

            for batch_ndx, sample in enumerate(replay_loader):
                img = sample['inputs'].cuda()
                img_metas = sample['meta_info']

                embeds = []
                with torch.no_grad():
                    fog_embeds = model_fog(img)[3]
                    night_embeds = model_night(img)[3]
                    rain_embeds = model_rain(img)[3]
                    embeds.append(fog_embeds)
                    embeds.append(night_embeds)
                    embeds.append(rain_embeds)
            
                embeds = torch.cat(embeds, dim = 0)
                output_logits = load_gate(embeds)

                labels = torch.zeros(img.size(0)).cuda().long()
                for i in range(img.size(0)):
                    if img_metas[i]['img_path'].find("/fog/") != -1:
                        labels[i] = 0
                    elif img_metas[i]['img_path'].find("/night/") != -1:
                        labels[i] = 1
                    elif img_metas[i]['img_path'].find("/rain/") != -1:
                        labels[i] = 2

                labels = torch.cat([labels] * 3, dim=0)

                probabilities = torch.softmax(output_logits, dim=1)
                predictions = torch.argmax(probabilities, dim=1)
                correct_predictions = (predictions == labels).sum().item()
                total_predictions = labels.size(0)
                accuracy = correct_predictions / total_predictions
                results.append(accuracy)
        
        if (mode == 4):
            replay_buffer = ReplayBuffer(buffer_size=500)
            replay_buffer.sample_images(data_loaders[0], 50)
            replay_buffer.sample_images(data_loaders[1], 50)
            replay_buffer.sample_images(data_loaders[2], 50)
            replay_buffer.sample_images(data_loaders[3], 50)

            replay_loader = DataLoader(replay_buffer, batch_size=8, shuffle=True,collate_fn=custom_collate)

            for batch_ndx, sample in enumerate(replay_loader):
                img = sample['inputs'].cuda()
                img_metas = sample['meta_info']

                embeds = []
                with torch.no_grad():
                    fog_embeds = model_fog(img)[3]
                    night_embeds = model_night(img)[3]
                    rain_embeds = model_rain(img)[3]
                    snow_embeds = model_snow(img)[3]
                    embeds.append(fog_embeds)
                    embeds.append(night_embeds)
                    embeds.append(rain_embeds)
                    embeds.append(snow_embeds)
            
                embeds = torch.cat(embeds, dim = 0)
                output_logits = load_gate(embeds)

                labels = torch.zeros(img.size(0)).cuda().long()
                for i in range(img.size(0)):
                    if img_metas[i]['img_path'].find("/fog/") != -1:
                        labels[i] = 0
                    elif img_metas[i]['img_path'].find("/night/") != -1:
                        labels[i] = 1
                    elif img_metas[i]['img_path'].find("/rain/") != -1:
                        labels[i] = 2
                    elif img_metas[i]['img_path'].find("/snow/") != -1:
                        labels[i] = 3

                labels = torch.cat([labels] * 4, dim=0)

                probabilities = torch.softmax(output_logits, dim=1)
                predictions = torch.argmax(probabilities, dim=1)
                correct_predictions = (predictions == labels).sum().item()
                total_predictions = labels.size(0)
                accuracy = correct_predictions / total_predictions
                results.append(accuracy)

        
        results = torch.cat(results)
        print(f'Average accuracy for mode {mode} is {results.mean()}')


    class GatingNet(nn.Module):
        def __init__(self):
            super(GatingNet, self).__init__()
            #self.linear = nn.Linear(261120, 2) # 261120 -> after squeezing he output from 4th block of backbone
            self.m = nn.Conv2d(512,1,(5,5))
            self.relu = nn.ReLU()
            self.pool = nn.MaxPool2d(2)
            self.linear = nn.Linear(6*13, 3)

        def forward(self, x):
            x = self.pool(self.relu(self.m(x)))
            x = x.view(x.size(0), -1)
            x = self.linear(x)
            return x

    def load_gate(x):
        x = x.cuda()
        path = '/BS/DApt/work/project/segformer_test/work_dirs/gate/save_gate_model_3_classes_v11.pth'
        gate = GatingNet()
        weights = torch.load(path)
        gate.load_state_dict(weights,strict= False)
        gate.eval()
        gate.cuda()

        with torch.no_grad():  # Ensure no gradients are calculated
            x = gate(x)
            return x

    run_infer(mode = 2)
    
if __name__ == '__main__':
    main()




