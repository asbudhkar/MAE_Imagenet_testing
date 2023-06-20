import os
import argparse
import math
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Resize, ToTensor, Compose, Normalize
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from model import *
from utils import setup_seed
from torch.utils.data import Subset

import numpy as np
import pandas as pd
import shutil, time, os, requests, random, copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_device_batch_size', type=int, default=256)
    parser.add_argument('--base_learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--total_epoch', type=int, default=20)
    parser.add_argument('--warmup_epoch', type=int, default=5)
    parser.add_argument('--pretrained_model_path', type=str, default='vit-t-mae_224_imgnet.pt')
    parser.add_argument('--output_model_path', type=str, default='vit-t-classifier-from_scratch_224_imgnet.pt')

    args = parser.parse_args()

    setup_seed(args.seed)

    batch_size = args.batch_size
    load_batch_size = min(args.max_device_batch_size, batch_size)

    assert batch_size % load_batch_size == 0
    steps_per_update = batch_size // load_batch_size


    def set_seed(seed = 16):
        np.random.seed(seed)
        torch.manual_seed(seed)

    import os
    from torch.utils.data import Dataset
    from PIL import Image
    import json

    class ImageNetKaggle(Dataset):
        def __init__(self, root, split, transform=None):
            self.samples = []
            self.targets = []
            self.transform = transform
            self.syn_to_class = {}
            with open(os.path.join(root, "imagenet_class_index.json"), "rb") as f:
                        json_file = json.load(f)
                        for class_id, v in json_file.items():
                            self.syn_to_class[v[0]] = int(class_id)
            with open(os.path.join(root, "ILSVRC2012_val_labels.json"), "rb") as f:
                        self.val_to_syn = json.load(f)
            samples_dir = os.path.join(root, "ILSVRC/Data/CLS-LOC", split)
            for entry in os.listdir(samples_dir):
                if split == "train":
                    syn_id = entry
                    target = self.syn_to_class[syn_id]
                    syn_folder = os.path.join(samples_dir, syn_id)
                    for sample in os.listdir(syn_folder):
                        sample_path = os.path.join(syn_folder, sample)
                        self.samples.append(sample_path)
                        self.targets.append(target)
                elif split == "val":
                    syn_id = self.val_to_syn[entry]
                    target = self.syn_to_class[syn_id]
                    sample_path = os.path.join(samples_dir, entry)
                    self.samples.append(sample_path)
                    self.targets.append(target)
        def __len__(self):
                return len(self.samples)
        def __getitem__(self, idx):
                x = Image.open(self.samples[idx]).convert("RGB")
                if self.transform:
                    x = self.transform(x)
                return x, self.targets[idx]


    from torch.utils.data import DataLoader
    from torchvision import transforms
    import torch
    import torchvision
    from tqdm import tqdm

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    val_transform = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )
    dataset = ImageNetKaggle("/data/abudhkar/imagenet/", "val", val_transform)
    indices = np.arange(len(dataset))
    train_indices, test_indices = train_test_split(indices, train_size=1000, stratify=dataset.targets)

    # Warp into Subsets and DataLoaders
    dataset = Subset(dataset, train_indices)

    val_dataloader = DataLoader(
                dataset,
                batch_size=4, 
                num_workers=2, 
                shuffle=True,
                drop_last=True,
                pin_memory=True
            )

    train_transform = transforms.Compose(
                [  
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )
    dataset = ImageNetKaggle("/data/abudhkar/imagenet/", "train", train_transform)
    # Split the indices in a stratified way to get imagenet subset to reduce computation time
    indices = np.arange(len(dataset))
    train_indices, test_indices = train_test_split(indices, train_size=10*1000, stratify=dataset.targets)

    # Warp into Subsets and DataLoaders
    dataset = Subset(dataset, train_indices)

    train_dataloader = DataLoader(
                dataset,
                batch_size=4, 
                num_workers=2, 
                shuffle=True,
                drop_last=True,
                pin_memory=True
            )
            
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.pretrained_model_path is not None:
        model = MAE_ViT()
        model = nn.DataParallel(model,device_ids=[2,3])
        model.load_state_dict(torch.load(args.pretrained_model_path))
        writer = SummaryWriter(os.path.join('logs', 'cifar10', 'pretrain-cls'))
    else:
        model = MAE_ViT()
        writer = SummaryWriter(os.path.join('logs', 'cifar10', 'scratch-cls'))
    
    # Linear probing
    model.module.encoder.requires_grad_(False)
    model = ViT_Classifier(model.module.encoder, num_classes=1000).to(device)
    model = model.to(device)
    
    # Parallely train pn 2 gpus
    model = nn.DataParallel(model,device_ids=[0,1])
    loss_fn = torch.nn.CrossEntropyLoss()
    acc_fn = lambda logit, label: torch.mean((logit.argmax(dim=-1) == label).float())

    optim = torch.optim.AdamW(model.parameters(), lr=args.base_learning_rate * args.batch_size / 256, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    lr_func = lambda epoch: min((epoch + 1) / (args.warmup_epoch + 1e-8), 0.5 * (math.cos(epoch / args.total_epoch * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func, verbose=True)

    best_val_acc = 0
    step_count = 0
    optim.zero_grad()
    for e in tqdm(range(args.total_epoch)):
        model.train()
        losses = []
        acces = []
        running_corrects = 0
        for img, label in tqdm(iter(train_dataloader)):
            step_count += 1
            img = img.to(device)
            label = label.to(device)
            logits = model(img)
            _, preds = torch.max(logits, 1)
          
            running_corrects += torch.sum(preds == label.data)
            loss = loss_fn(logits, label)
            acc = acc_fn(logits, label)
            loss.backward()
            if step_count % steps_per_update == 0:
                optim.step()
                optim.zero_grad()
            losses.append(loss.item())
            acces.append(acc.item())
        epoch_acc = running_corrects / len(train_dataloader)
        print("Train Epoch acc",epoch_acc)
        lr_scheduler.step()
        avg_train_loss = sum(losses) / len(losses)
        avg_train_acc = sum(acces) / len(acces)
        print(f'In epoch {e}, average training loss is {avg_train_loss}, average training acc is {avg_train_acc}.')

        model.eval()
        with torch.no_grad():
            losses = []
            acces = []
            running_corrects = 0
            for img, label in tqdm(iter(val_dataloader)):
                img = img.to(device)
                label = label.to(device)
                logits = model(img)
                
                _, preds = torch.max(logits, 1)
                running_corrects += torch.sum(preds == label.data)
                
                loss = loss_fn(logits, label)
                acc = acc_fn(logits, label)
                losses.append(loss.item())
                acces.append(acc.item())
            avg_val_loss = sum(losses) / len(losses)
            avg_val_acc = sum(acces) / len(acces)
            print("Val Epoch acc",running_corrects/len(val_dataloader))
            print(f'In epoch {e}, average validation loss is {avg_val_loss}, average validation acc is {avg_val_acc}.')  

        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            print(f'saving best model with acc {best_val_acc} at {e} epoch!')       
            torch.save(model.state_dict(), args.output_model_path)

        writer.add_scalars('cls/loss', {'train' : avg_train_loss, 'val' : avg_val_loss}, global_step=e)
        writer.add_scalars('cls/acc', {'train' : avg_train_acc, 'val' : avg_val_acc}, global_step=e)