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
from torch.utils.data import Sampler, DataLoader

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


    backbone_fog.cuda()
    backbone_night.cuda()

    backbone_fog.eval()
    backbone_night.eval()


    return backbone_fog,backbone_night


class GatingNet(nn.Module):
    def __init__(self):
        super(GatingNet, self).__init__()
        #self.linear = nn.Linear(261120, 2) # 261120 -> after squeezing he output from 4th block of backbone
        self.conv1 = nn.Conv2d(512, 64, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        
        self.conv2 = nn.Conv2d(64, 3, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(3)
        
        # Adjust the dimensions accordingly to match the flattened size after pooling
        self.fc1 = nn.Linear(3 * 6 * 3, 128) 
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 2)

        self.initialize_weights()
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d) or isinstance(m,nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        #import pdb;pdb.set_trace()
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = x.reshape(x.size(0), -1)
        x = self.dropout(self.fc1(x))
        x = self.fc2(x)
        return x
    

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

    return dataloaders

class ReplayBuffer(Dataset):
    def __init__(self,dataloader1, dataloader2,num_samples=100):
        self.dl1 = dataloader1
        self.dl2 = dataloader2
        dataset1 = self.dl1.dataset
        dataset2 = self.dl2.dataset
        indices1 = random.sample(range(len(dataset1)), num_samples)
        indices2 = random.sample(range(len(dataset2)), num_samples)

        self.buffer = []
        for id1,id2 in zip(indices1,indices2):
            data1 = dataset1[id1]
            data2 = dataset2[id2]
            self.buffer.append(data1)
            self.buffer.append(data2)

    def __getitem__(self, idx):
        data = self.buffer[idx]
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

        return data
            
    def __len__(self):
        return len(self.buffer)

    
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

class BalancedSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        
        self.fog_indices = [i for i, data in enumerate(dataset.buffer) if data['data_samples'].img_path.find("fog") != -1]
        self.night_indices = [i for i, data in enumerate(dataset.buffer) if data['data_samples'].img_path.find("night") != -1]
        #import pdb;pdb.set_trace()

    def __iter__(self):
        num_batches = len(self.dataset) // self.batch_size

        random.shuffle(self.fog_indices)
        random.shuffle(self.night_indices)

        indices = []
        for _ in range(num_batches):
            fog_batch = random.sample(self.fog_indices, self.batch_size // 2)
            night_batch = random.sample(self.night_indices, self.batch_size // 2)
            batch = fog_batch + night_batch

            random.shuffle(batch)
            indices.extend(batch)

        random.shuffle(indices)
        
        return iter(indices)

    def __len__(self):
        return len(self.dataset)


def main():
    
    mmengine.runner.set_random_seed(0)
    gate_model = GatingNet().cuda()

    model_fog, model_night = load_models()
    data_loaders = load_dataloader()
    replay_buffer = ReplayBuffer(data_loaders[0],data_loaders[1])
    sampler = BalancedSampler(replay_buffer, batch_size=16)
    replay_loader = DataLoader(replay_buffer, batch_size=16, sampler=sampler, collate_fn=custom_collate)

    #import pdb;pdb.set_trace()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(gate_model.parameters(), lr=1e-3)
      
        # training loop for gate
    for epoch in range(50):
        run_loss = 0.0
 
        for batch_ndx, sample in enumerate(replay_loader):
            img = sample['inputs'].cuda()
            img_metas = sample['meta_info']
            embeds = []
                
            optimizer.zero_grad()
            with torch.no_grad():
                    
                fog_embeds = model_fog(img)[3]
                night_embeds = model_night(img)[3]
                embeds.append(fog_embeds)
                embeds.append(night_embeds)
      
            embeds = torch.cat(embeds, dim = 0) 
                
            gate_model.train()
            output = gate_model(embeds) 
                #import pdb;pdb.set_trace()
            labels = torch.zeros(img.size(0)).cuda().long()
            for i in range(img.size(0)):
                if img_metas[i]['img_path'].find("/fog/") != -1:
                    labels[i] = 0
                elif img_metas[i]['img_path'].find("/night/") != -1:
                    labels[i] = 1
               
            #import pdb;pdb.set_trace()
            labels = torch.cat([labels]*2 , dim=0)
            #import pdb;pdb.set_trace()
            loss = criterion(output, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(gate_model.parameters(), max_norm=1.0)

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
                print(f'epoch {epoch + 1}, accuracy {accuracy} loss: {run_loss / 5:.4f}')
                run_loss = 0.0

    save_path = f'/BS/DApt/work/project/segformer_test/work_dirs/gate/NightvsFog_100.pth'
    torch.save(gate_model.state_dict(), save_path)

    print(f'Model saved as {save_path}')

if __name__ == '__main__':
    main()
