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
from torch.cuda.amp import autocast, GradScaler
import cv2

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



def slide_inference_miou(preds: torch.Tensor, gt: torch.Tensor, crop_size=(1024, 1024), stride=(768, 768), num_classes=19):
    h_stride, w_stride = stride
    h_crop, w_crop = crop_size
    batch_size, num_classes, h_img, w_img = preds.shape  # Unpack the shape

    # Calculate the number of horizontal and vertical sliding windows
    h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
    w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1

    # Initialize accumulators
    preds_accum = torch.zeros((batch_size, num_classes, h_img, w_img), dtype=torch.float32, device=preds.device)
    count_mat = torch.zeros((batch_size, 1, h_img, w_img), dtype=torch.float32, device=preds.device)

    # Slide over the image to accumulate predictions
    for h_idx in range(h_grids):
        for w_idx in range(w_grids):
            y1 = h_idx * h_stride
            x1 = w_idx * w_stride
            y2 = min(y1 + h_crop, h_img)
            x2 = min(x1 + w_crop, w_img)
            y1 = max(y2 - h_crop, 0)
            x1 = max(x2 - w_crop, 0)

            preds_patch = preds[:, :, y1:y2, x1:x2]
            count_mat[:, :, y1:y2, x1:x2] += 1
            preds_accum[:, :, y1:y2, x1:x2] += preds_patch

    # Normalize the accumulated predictions
    preds_accum /= count_mat

    # Convert logits to class predictions
    preds_final = torch.argmax(preds_accum, dim=1)  # Shape: [batch_size, h_img, w_img]

    # Calculate mIoU
    results = []
    for i in range(batch_size):
        area_intersect, area_union, _, _ = intersect_and_union(preds_final[i], gt[i], num_classes)
        results.append((area_intersect, area_union))

    # Sum up all intersections and unions
    total_area_intersect = torch.sum(torch.stack([r[0] for r in results]), dim=0)
    total_area_union = torch.sum(torch.stack([r[1] for r in results]), dim=0)

    # Calculate the IoU for each class
    total_iou = total_area_intersect / total_area_union
    miou = total_iou[~torch.isnan(total_iou)].mean()

    return miou

def intersect_and_union(preds: torch.Tensor, labels: torch.Tensor):
    #Calculate Intersection and Union for a batch of predictions and ground truths.
    num_classes = preds.size(1)
    
    pred_label= torch.argmax(preds, dim=0)  # Convert softmax scores to class predictions
    x = pred_label

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

class GaussianNoise(torch.nn.Module):
    def __init__(self, mean=0., std=0.7):
        super().__init__()
        self.std = std
        self.mean = mean

    def forward(self, img):
        noise = torch.randn(img.size()) * self.std + self.mean
        noise = noise.to(img.device)
        return img + noise

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class MultiScaleAug(object):
    def __init__(self, scale_factors=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]):
        self.scale_factors = scale_factors

    def __call__(self, img):
        c, h, w = img.shape
        augmented_imgs = []

        for scale in self.scale_factors:
            # Scale the image
            scaled_h, scaled_w = int(h * scale), int(w * scale)
            scaled_img = F.interpolate(img.unsqueeze(0), size=(scaled_h, scaled_w), mode='bilinear', align_corners=False).squeeze(0)
            
            # Add the scaled image
            augmented_imgs.append(scaled_img)

        return augmented_imgs

def augs1():
    train_aug = transforms.Compose([
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15)])
    return train_aug

def augs2():
    train_aug = transforms.Compose([
        transforms.GaussianBlur(kernel_size=(55,93), sigma= [0.1, 0.9])])
    return train_aug

def augs3():
    train_aug = transforms.Compose([
        GaussianNoise()])
    return train_aug

class CoTTA:
    def __init__(self, t_model, s_model, model_anchor, model_state, data_loader, val_dataloader, num_iterations, target_height, target_width, learning_rate, save_dir, weight_decay=0.01, alpha_teacher=0.01):
        self.t_model = t_model
        self.s_model = s_model
        self.model_anchor = model_anchor
        self.model_state = model_state
        self.data_loader = data_loader
        self.val_dataloader = val_dataloader
        self.num_iterations = num_iterations
        self.target_height = target_height
        self.target_width = target_width
        self.learning_rate = learning_rate
        self.save_dir = save_dir
        self.weight_decay = weight_decay
        self.alpha_teacher = alpha_teacher
        self.scaler = GradScaler()

        self._freeze_model(self.t_model, requires_grad=False)
        self._freeze_model(self.s_model, requires_grad=True)

        self.optimizer = self.configure_optimizer()
        self.scheduler = self.configure_scheduler()
        
        os.makedirs(self.save_dir, exist_ok=True)

        self.first_batches = self._fetch_first_batches()

        self.transform1 = augs1()  
        self.transform2 = augs2()
        self.transform3 = augs3()
        self.multi_scale_aug = MultiScaleAug()

    def _freeze_model(self, model, requires_grad):
        for param in model.parameters():
            param.requires_grad = requires_grad

    def _fetch_first_batches(self):
        first_batches = []
        for i, sample in enumerate(self.data_loader):
            if i < 2:
                first_batches.append(sample)
            else:
                break
        return first_batches
    
    def generate_pseudo_labels(self, imgs, batch_img_metas):
        with torch.no_grad():
            all_preds = []
            for img in imgs:
                # Apply color jitter
                jittered_img = self.transform1(img.unsqueeze(0)).squeeze(0)

                # Apply multi-scale augmentation
                scaled_images = self.multi_scale_aug(jittered_img)
                preds = []
                
                for scaled_img in scaled_images:
                    scaled_img = scaled_img.unsqueeze(0)  # Add batch dimension
                    outputs = self.t_model.module.encode_decode(scaled_img, batch_img_metas)
                    
                    # Resize the outputs to the target size (540x960)
                    outputs_resized = F.interpolate(outputs, size=(540,960), mode='bilinear', align_corners=False)
                    
                    # Compute softmax probabilities
                    #probs = F.softmax(outputs_resized, dim=1)
                    preds.append(outputs_resized)
    
                # Average the predictions
                averaged_pred = torch.mean(torch.stack(preds), dim=0)
                outputs_ema = F.softmax(averaged_pred, dim=1)
                max_confidences, pseudo_label = torch.max(outputs_ema, dim=1)
                
                # Apply confidence thresholding
                confidence_threshold = 0.962
                mask2 = max_confidences >= confidence_threshold
                pseudo_label[~mask2] = -1234  # Set low-confidence predictions to -1 (ignore index)

                all_preds.append(pseudo_label)

            # Combine pseudo-labels from different images
            pseudo_labels = torch.stack(all_preds)
            #import pdb;pdb.set_trace()
            return pseudo_labels

    def configure_optimizer(self):

        param_dict = {pn: p for pn, p in self.s_model.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {'params': decay_params, 'weight_decay': self.weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=self.learning_rate, betas=(0.9, 0.999),eps=1e-8)
        return optimizer

    def configure_scheduler(self):
        def lr_lambda(iteration):
            warmup_iters = 2000
            if iteration < warmup_iters:
                return iteration / warmup_iters
            return max(0.0, (self.num_iterations - iteration) / (self.num_iterations - warmup_iters))
        
        scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        return scheduler

    def update_ema_variables(self,iteration):
        t_model_state_dict = self.t_model.state_dict()

        self.alpha_teacher = min(1 - 1 / (iteration + 1),  self.alpha_teacher)
        
        for name, param in self.s_model.named_parameters():
            if name in t_model_state_dict:
                ema_param = t_model_state_dict[name]
                ema_param.copy_(self.alpha_teacher * ema_param + (1 - self.alpha_teacher) * param.data)
            else:
                print(f"Warning: {name} not found in teacher model's state_dict")
    
    def check_teacher_gradients(self):
        for param in self.t_model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                print(f"Teacher model parameter {param.name} has non-zero gradient!")

    def train(self):
        '''wandb.init(project="CoTTA_v1", config={
            "num_iterations": self.num_iterations,
            "target_height": self.target_height,
            "target_width": self.target_width,
            "learning_rate": self.learning_rate
        })'''

        #config = wandb.config

        iteration = 0
        best_miou = 0.0

        while iteration < self.num_iterations:
            for batch_ndx, sample in enumerate(self.first_batches):
                if iteration >= self.num_iterations:
                    break
                            
                img_paths = [s.img_path for s in sample['data_samples']] 
                imgs = self.preprocess_images(img_paths)
                #import pdb;pdb.set_trace()

                batch_img_metas = [
                    dict(
                        ori_shape=imgs.shape[2:],
                        img_shape=imgs.shape[2:],
                        pad_shape=imgs.shape[2:],
                        padding_size=[0, 0, 0, 0]
                    )
                ] * imgs.shape[0]

                anchor = F.softmax(self.model_anchor.encode_decode(imgs, batch_img_metas), dim=1)
                anchor_prob = anchor.max(1)[0].detach()
                mask = anchor_prob >= 0.96 #in code paper #96

                
                mask = mask.detach()
                    #print(mask.sum()/(mask.view(-1).shape[0]))
                #import pdb;pdb.set_trace()


                ## Teacher Predictions with transformed images
                with torch.no_grad():
                    x = self.t_model.module.encode_decode(self.transform1(imgs), batch_img_metas)
                    y = self.t_model.module.encode_decode(self.transform2(imgs), batch_img_metas)
                    z = self.t_model.module.encode_decode(self.transform3(imgs), batch_img_metas)
                    
                    outputs_ema = (x + y + z  )/3.

                    outputs_ema = F.softmax(x, dim=1) 
                    # Get maximum confidence scores and pseudo-labels
                    max_confidences, pseudo_label = torch.max(outputs_ema, dim=1)
                    # Apply confidence thresholding
                    confidence_threshold = 0.85
                    mask2 = max_confidences >= confidence_threshold   
                    # Set low-confidence predictions to -1 (ignore index)
                    pseudo_label[~mask2] = -1234


                    #pseudo_label = self.generate_pseudo_labels(imgs, batch_img_metas)
                    #pseudo_label = torch.squeeze(pseudo_label,1)
                    standard_ema = self.t_model.module.encode_decode(imgs, batch_img_metas)
                    standard_ema = F.softmax(standard_ema, dim=1)
                    max_c, _ = torch.max(standard_ema, dim=1)
                    y = max_c.mean(dim=0)

                    if y.max() < 0.80:
                        print(max_c)

                    standard_ema= torch.argmax(standard_ema, dim=1)
                    mask = mask.detach()
                    t_pred = torch.where(mask, standard_ema, pseudo_label) ## check output

                ## Student prediction
                with amp.autocast():
                    s_logits = self.s_model.module.encode_decode(imgs, batch_img_metas)
                    #import pdb;pdb.set_trace()
                    loss = nn.CrossEntropyLoss(ignore_index=-1234,label_smoothing = 0.1)(s_logits,t_pred)

                #loss = self.compute_student_loss(imgs, batch_img_metas, t_pred)

                # Check if any gradients are set for the teacher model
                #self.check_teacher_gradients()

                # Backward pass and optimizer step
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                #import pdb;pdb.set_trace()

                # Check again after backward pass
                #self.check_teacher_gradients()

                # Scheduler and EMA update
                self.scheduler.step()
                self.update_ema_variables(iteration)

                if iteration > 1:
                    self.stochastic_restoration()

                #wandb.log({"train_loss": loss.item(), "iteration": iteration})

                if iteration % 100 == 0:
                    val_loss, miou = self.validate_model()
                    #wandb.log({"val_loss": val_loss, "val_miou": miou, "iteration": iteration})
                    self.save_model(miou, iteration)
                    print(f"Iteration {iteration}, Train_Loss: {loss.item():.4f}, Val_Loss: {val_loss:.4f}, Val_miou: {miou:.4f}")

                if iteration % 50 == 0:
                    print(f"Iteration {iteration}, Train_Loss: {loss.item():.4f}")

                iteration += 1

        print("Training completed. Best model saved with miou:", best_miou)
        #wandb.finish()

    def preprocess_images(self, paths):
        img_paths = paths
        new_size = (960,540) 
        full_image = []
        for imgs in img_paths:
            # Read and convert the image
            img = cv2.imread(imgs)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            resized_image = cv2.resize(img, new_size)
            resized_image = resized_image.astype(np.float32) 
            resized_image = torch.tensor(resized_image).permute(2, 0, 1)  
            mean = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1)
            std = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1)
            normalized_image = (resized_image - mean) / std
            full_image.append(normalized_image)

        # Stack all images into a single tensor and move to GPU if available
        full_image = torch.stack(full_image, dim=0).cuda()

        return full_image

    def stochastic_restoration(self):
        for n, p in self.s_model.named_parameters():
            if any(x in n for x in ['weight', 'bias']) and p.requires_grad:
                parts = n.split('.')
                new_n = '.'.join(parts[1:])  # Needed when using multiple GPUs
                mask = (torch.rand(p.shape) < 0.01).float().cuda()
                with torch.no_grad():
                    p.data = self.model_state[f"{new_n}"].cuda() * mask + p * (1.0 - mask)

    def validate_model(self):
        # Unpack models and validation loader
        model = self.s_model
        x_model = self.t_model
        val_loader = self.val_dataloader

        val_loss = 0
        total_batches = len(val_loader)
        model.eval()
        x_model.eval()

        results = []
        for batch_ndx, sample in enumerate(val_loader):
            imgs = []
            for img in sample['inputs']:
                _, current_height, current_width = img.shape
                if img.dtype != torch.float32:
                    img = img.float()
                rgb_image = img[[2, 1, 0], :, :]
                
                mean = torch.tensor([123.675, 116.28, 103.53], device=rgb_image.device).view(3, 1, 1)
                std = torch.tensor([58.395, 57.12, 57.375], device=rgb_image.device).view(3, 1, 1)

                # Normalize the image
                normalized_image = (rgb_image - mean) / std
                #normalized_image = F.interpolate(normalized_image.unsqueeze(0), size=(600,960), mode='bilinear', align_corners=False)
                imgs.append(normalized_image)

            img = torch.stack(imgs, dim=0).cuda()
            img = torch.squeeze(img,1)
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
                final_predictionx = F.interpolate(final_prediction, size=(1024, 1024), mode='nearest')
                t_pred = torch.argmax(final_prediction, dim=1)
                s_logits = x_model.module.encode_decode(img, batch_img_metas)

            gts = []
            for gt in sample['data_samples']:
                gt = gt._gt_sem_seg.data
                gt = gt.float()
                gts.append(gt)
            
            gt = torch.stack(gts, dim=0).cuda()
            #
            gt = F.interpolate(gt, size=(1024, 1024), mode='nearest')
            
            gt = gt.squeeze(0)
            gt = gt.to(dtype=torch.int64)
            gt = gt.squeeze(1)


            #import pdb;pdb.set_trace()
            for i in range(final_predictionx.shape[0]):
                area_intersect, area_union, area_pred_label, area_label = intersect_and_union(
                    final_predictionx[i], gt[i]
                )
                areas = (area_intersect, area_union, area_pred_label, area_label)
                results.append(areas)

            loss = nn.CrossEntropyLoss()(s_logits, t_pred)
            val_loss += loss.item()
            
        results = tuple(zip(*results))
        assert len(results) == 4
        total_area_intersect = torch.sum(torch.stack(results[0],dim = 0),dim = 0)
        total_area_union = torch.sum(torch.stack(results[1],dim = 0), dim = 0)
        total_iou = total_area_intersect / total_area_union
        miou = total_iou[~torch.isnan(total_iou)].mean()
        #import pdb;pdb.set_trace
        #miou = slide_inference_miou(final_prediction, gt, crop_size=(1024, 1024), stride=(768, 768), num_classes=19)
        
        return val_loss / total_batches, miou

    def save_model(self, miou, iteration):
        # Create the save directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)

        # Define the model file path
        model_path = os.path.join(self.save_dir, f'fog_{miou:.4f}_{iteration}.pth')

        # Save the teacher model state dictionary
        torch.save(self.t_model.state_dict(), model_path)

        # Print a message indicating the model has been saved
        print(f"Iteration {iteration}, New best validation miou: {miou:.4f}. Model saved to {model_path}")

def main():
    config_path = '/BS/DApt/work/project/segformer_test/local_config/my_models/b5_fog_nlora.py'
    target_height = 540
    target_width = 960

    # Load data loaders, models, and weights
    data_loaders, val_dataloaders = load_dataloader()
    t_model, s_model, weights = load_models(config_path)
    
    # Create the anchor model (deepcopy of student model)
    model_anchor = deepcopy(s_model)
    
    # Parallelize models for multi-GPU training
    s_model = nn.DataParallel(s_model).cuda()
    t_model = nn.DataParallel(t_model).cuda()

    # Create an instance of the CoTTA class
    cotta_trainer = CoTTA(
        t_model=t_model,
        s_model=s_model,
        model_anchor=model_anchor,
        model_state=weights,
        data_loader=data_loaders[0],
        val_dataloader=val_dataloaders[0],
        num_iterations=10000,
        target_height=target_height,
        target_width=target_width,
        learning_rate=(6e-6)/4,
        save_dir="/BS/DApt/work/project/segformer_test/work_dirs/cottav5/night"
    )

    # Start training
    cotta_trainer.train()

if __name__ == '__main__':
    main()

