import torch
import torch.nn as nn

from mmseg.registry import MODELS
from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.registry import init_default_scope
init_default_scope('mmseg')
from mmseg.datasets.acdc import ACDCDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Dataset
import random
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import mmengine

target_shape = (960, 540)
target_height = 540
target_width = 960
mean = [123.675, 116.28, 103.53]
std = [58.395, 57.12, 57.375]

# Define the Albumentations transformation
transform = A.Compose([
    A.Resize(height=target_shape[0], width=target_shape[1]),
   A.Normalize(mean=mean, std=std),
    ToTensorV2()
])

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

    check_point = '/BS/DApt/work/project/segformer_test/work_dirs/b5_rain_IDASS/teacher_state_dict.pth'
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


class GatingNet(nn.Module):
    def __init__(self):
        super(GatingNet, self).__init__()
        #self.linear = nn.Linear(261120, 2) # 261120 -> after squeezing he output from 4th block of backbone
        self.m = nn.Conv2d(512,1,(5,5))
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.linear = nn.Linear(6*13, 2)
        
    def forward(self, x):
        #import pdb;pdb.set_trace()
        x = self.pool(self.relu(self.m(x)))
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x
    
    def increase_linear(self):
        if self.linear.out_features < 4:  # Maximum output neurons should be 4
            self.linear = increase_classifier(self.linear)
        
def increase_classifier(linear_layer):
    old_shape = linear_layer.weight.shape
    new_layer = nn.Linear(old_shape[1], old_shape[0] + 1).cuda()

    # Copy weights and bias from the old layer to the new layer
    print('Copying weights and bias from the old layer to the new layer')
    new_layer.weight.data[:old_shape[0], :] = linear_layer.weight.data
    new_layer.bias.data[:old_shape[0]] = linear_layer.bias.data
    
    return new_layer

def load_dataloader():
    dataloaders = []
    #Fog
    dataset_config = '/BS/DApt/work/project/segformer_test/local_config/_base_/datasets/acdc_fog.py'
    cfg = Config.fromfile(dataset_config)
    dl1 = Runner.build_dataloader(cfg.train_dataloader)
    dataloaders.append(dl1)
    #Night
    dataset_config = '/BS/DApt/work/project/segformer_test/local_config/_base_/datasets/acdc_night.py'
    cfg = Config.fromfile(dataset_config)
    dl2 = Runner.build_dataloader(cfg.train_dataloader)
    dataloaders.append(dl2)
    #Rain
    dataset_config = '/BS/DApt/work/project/segformer_test/local_config/_base_/datasets/acdc_rain.py'
    cfg = Config.fromfile(dataset_config)
    dl3 = Runner.build_dataloader(cfg.train_dataloader)
    dataloaders.append(dl3)
    #Snow
    dataset_config = '/BS/DApt/work/project/segformer_test/local_config/_base_/datasets/acdc_snow.py'
    cfg = Config.fromfile(dataset_config)
    dl4 = Runner.build_dataloader(cfg.train_dataloader)
    dataloaders.append(dl4)

    return dataloaders

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
    replay_buffer = ReplayBuffer(buffer_size=800)
    mmengine.runner.set_random_seed(0)
    gate_model = GatingNet().cuda()

    model_fog, model_night, model_rain, model_snow = load_models()
    data_loaders = load_dataloader()

    criterion = nn.CrossEntropyLoss(label_smoothing=0.01)
    optimizer = optim.AdamW(gate_model.parameters(), lr=3e-5,betas=(0.95, 0.999),weight_decay = 0.001)

    print("Add 10 random fog images")
    replay_buffer.sample_images(data_loaders[0], 5)

    for dl in range(1, len(data_loaders)):
        replay_buffer.sample_images(data_loaders[dl], 5)
        replay_loader = DataLoader(replay_buffer, batch_size=4, shuffle=True,collate_fn=custom_collate)

        # training loop for gate
        for epoch in range(50):
            run_loss = 0.0
 
            for batch_ndx, sample in enumerate(replay_loader):
                img = sample['inputs'].cuda()
                img_metas = sample['meta_info']
                embeds = []
                
                optimizer.zero_grad()
                with torch.no_grad():
                    if dl == 1:
                        continue
                        '''fog_embeds = model_fog(img)[3]
                        night_embeds = model_night(img)[3]
                        embeds.append(fog_embeds)
                        embeds.append(night_embeds)'''
                
                    elif dl == 2:
                        continue
                        '''fog_embeds = model_fog(img)[3]
                        night_embeds = model_night(img)[3]
                        rain_embeds = model_rain(img)[3]
                        embeds.append(fog_embeds)
                        embeds.append(night_embeds)
                        embeds.append(rain_embeds)'''
                
                    else:
                        fog_embeds = model_fog(img)[3]
                        night_embeds = model_night(img)[3]
                        rain_embeds = model_rain(img)[3]
                        snow_embeds = model_snow(img)[3]
                        
                        embeds.append(fog_embeds)
                        embeds.append(night_embeds)
                        embeds.append(rain_embeds)
                        embeds.append(snow_embeds)
                
                embeds = torch.cat(embeds, dim = 0) 
                
            
                gate_model.train()
                output = gate_model(embeds) 
                #import pdb;pdb.set_trace()
                labels = torch.zeros(img.size(0)).cuda().long()
                for i in range(img.size(0)):
                    #print(img_metas[i]['img_path'])
                    if img_metas[i]['img_path'].find("/fog/") != -1:
                        labels[i] = 0
                    elif img_metas[i]['img_path'].find("/night/") != -1:
                        labels[i] = 1
                    elif img_metas[i]['img_path'].find("/rain/") != -1:
                        labels[i] = 2
                    elif img_metas[i]['img_path'].find("/snow/") != -1:
                        labels[i] = 3
                
                labels = torch.cat([labels] * (dl + 1), dim=0)
                probabilities = torch.softmax(output, dim=1)
                predictions = torch.argmax(probabilities, dim=1)

                #import pdb;pdb.set_trace()
            
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                run_loss += loss.item()
                if batch_ndx % 10 == 0:
                    print(f'labels: {labels}')
                    print(f'predictions: {predictions}') 
                    correct_predictions = (predictions == labels).sum().item()
                    total_predictions = labels.size(0)
                    accuracy = correct_predictions / total_predictions 
                    print(f'epoch {epoch + 1}, accuracy {accuracy} loss: {run_loss / 5:.4f}')
                    run_loss = 0.0

        save_path = f'/BS/DApt/work/project/segformer_test/work_dirs/gate/save_gate_model_{dl + 1}_classes_v11.pth'
        torch.save(gate_model.state_dict(), save_path)

        print(f'Model saved as {save_path}')
        print("Increase classifier's classes by one.")
        gate_model.increase_linear()
        print("Update classifier")
        optimizer = optim.AdamW(gate_model.parameters(), lr=3e-5,betas=(0.95, 0.999),weight_decay = 0.001)
    

if __name__ == '__main__':
    main()
