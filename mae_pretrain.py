import os
import argparse
import math
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Resize, ToTensor, Compose, Normalize
from tqdm import tqdm

from model import *
from utils import setup_seed
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--max_device_batch_size', type=int, default=512)
    parser.add_argument('--base_learning_rate', type=float, default=1.5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--mask_ratio', type=float, default=0.75)
    parser.add_argument('--total_epoch', type=int, default=2000)
    parser.add_argument('--warmup_epoch', type=int, default=400)
    parser.add_argument('--model_path', type=str, default='vit-t-mae_224_imgnet_vit-base.pt')

    args = parser.parse_args()

    setup_seed(args.seed)

    batch_size = args.batch_size
    load_batch_size = min(args.max_device_batch_size, batch_size)

    assert batch_size % load_batch_size == 0
    steps_per_update = batch_size // load_batch_size

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
    
    # Get subset of imagenet to reduce computation time
    train_indices, test_indices = train_test_split(indices, train_size=1000, stratify=dataset.targets)

    # Warp into Subsets and DataLoaders
    dataset = Subset(dataset, train_indices)

    val_dataloader = DataLoader(
                dataset,
                batch_size=2, 
                num_workers=1, 
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
    indices = np.arange(len(dataset))
    
    # Get subset of imagenet to reduce computation time
    train_indices, test_indices = train_test_split(indices, train_size=80*1000, stratify=dataset.targets)

    # Warp into Subsets and DataLoaders
    dataset = Subset(dataset, train_indices)

    train_dataloader = DataLoader(
                dataset,
                batch_size=2, # may need to reduce this depending on your GPU 
                num_workers=1, # may need to reduce this depending on your num of CPUs and RAM
                shuffle=True,
                drop_last=True,
                pin_memory=True
            )
    writer = SummaryWriter(os.path.join('logs', 'cifar10', 'mae-pretrain'))
    device = 'cuda:2' if torch.cuda.is_available() else 'cpu'

    model = MAE_ViT(mask_ratio=args.mask_ratio).to(device)
    
    # Parallely train on multiple gpus
    model = torch.nn.DataParallel(model,device_ids=[2,3])
    optim = torch.optim.AdamW(model.parameters(), lr=args.base_learning_rate * args.batch_size / 256, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    lr_func = lambda epoch: min((epoch + 1) / (args.warmup_epoch + 1e-8), 0.5 * (math.cos(epoch / args.total_epoch * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func, verbose=True)

    step_count = 0
    optim.zero_grad()
    for e in range(args.total_epoch):
        model.train()
        losses = []
        for img, label in tqdm(iter(train_dataloader)):
            step_count += 1
            img = img.to(device)
            predicted_img, mask = model(img)
            loss = torch.mean((predicted_img - img) ** 2 * mask) / args.mask_ratio
            loss.backward()
            if step_count % steps_per_update == 0:
                optim.step()
                optim.zero_grad()
            losses.append(loss.item())
        lr_scheduler.step()
        avg_loss = sum(losses) / len(losses)
        writer.add_scalar('mae_loss', avg_loss, global_step=e)
        print(f'In epoch {e}, average traning loss is {avg_loss}.')

        torch.save(model.state_dict(), args.model_path)