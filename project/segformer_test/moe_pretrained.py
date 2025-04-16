import torch
import torch.nn as nn
import numpy as np
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
import timm
import torchvision.transforms as transforms

target_shape = (384,384)
target_height = 384
target_width = 384
mean = [123.675, 116.28, 103.53]
std = [58.395, 57.12, 57.375]

class GatingNet(nn.Module):
    def __init__(self):
        super(GatingNet, self).__init__()
        #self.linear = nn.Linear(261120, 2) # 261120 -> after squeezing he output from 4th block of backbone
        self.backbone = timm.create_model('vit_base_patch16_clip_384.openai_ft_in1k', pretrained=True, 
                                                          num_classes=0, drop_rate=0., drop_path_rate=0., global_pool='')
        self.pool = nn.AdaptiveAvgPool2d(16)
        self.linear = nn.Linear(16*16, 2)

        for n, p in self.backbone.named_parameters():
            p.requires_grad = False
        
    def forward(self, img):
        #import pdb;pdb.set_trace()
        x = self.backbone(img)
        x = self.pool(x)
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
    replay_buffer = ReplayBuffer(buffer_size=400)
    mmengine.runner.set_random_seed(0)
    gate_model = GatingNet().cuda()

    data_loaders = load_dataloader()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(gate_model.parameters(), lr=3e-4,weight_decay = 0.001)

    
    replay_buffer.sample_images(data_loaders[0], 50)

    for dl in range(1, len(data_loaders)):
        print("Added in replay buffer")
        replay_buffer.sample_images(data_loaders[dl], 50)
        replay_loader = DataLoader(replay_buffer, batch_size=16, shuffle=True,collate_fn=custom_collate)
        accuracies = []

        # training loop for gate
        for epoch in range(30):
            run_loss = 0.0
 
            for batch_ndx, sample in enumerate(replay_loader):
                img = sample['inputs'].cuda()
                img_metas = sample['meta_info']
                
                embeds = []
                
                optimizer.zero_grad()

                gate_model.train()
                output = gate_model(img) 
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
                
                #labels = torch.cat([labels] * (dl + 1), dim=0)
                print(labels)
                #import pdb;pdb.set_trace()
            
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                run_loss += loss.item()
                if batch_ndx % 10 == 0:  
                    probabilities = torch.softmax(output, dim=1)
                    predictions = torch.argmax(probabilities, dim=1)
                    print(labels)
                    print(predictions) 
                    correct_predictions = (predictions == labels).sum().item()
                    total_predictions = labels.size(0)
                    accuracy = correct_predictions / total_predictions
                    accuracies.append(accuracy)
                    print(f'epoch {epoch + 1}, accuracy {accuracy} loss: {run_loss / 5:.4f}')
                    run_loss = 0.0
        
        #print(accuracies)
        #import pdb;pdb.set_trace()
        #accuracies = torch.cat(accuracies)
        print(f'Average Training Accuracy {np.mean(accuracies)}')


        save_path = f'/BS/DApt/work/project/segformer_test/work_dirs/gate/clip_{dl + 1}_classes_v2.pth'
        torch.save(gate_model.state_dict(), save_path)

        print(f'Model saved as {save_path}')
        print("Increase classifier's classes by one.")
        gate_model.increase_linear()
        print("Update classifier")
        optimizer = optim.AdamW(gate_model.parameters(), lr=3e-4,weight_decay = 0.001)
    

if __name__ == '__main__':
    main()
