# triplet_resnet_train.py
import os, math, random
from typing import List, Tuple, Dict
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import models, transforms

# ---------------------------
# 1) Dataset: 返回 (image, label)
# ---------------------------
class YourDataset(Dataset):
    """
    假设 self.items = [(img_path, class_id), ...]
    你可以改造成从标注文件/目录结构读取。
    """
    def __init__(self, items: List[Tuple[str, int]], train=True, img_size=224):
        self.items = items
        norm = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        if train:
            self.tf = transforms.Compose([
                transforms.Resize(int(img_size*1.1)),
                transforms.RandomResizedCrop(img_size, scale=(0.7,1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.2,0.2,0.2,0.1),
                transforms.ToTensor(),
                norm
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize(int(img_size*1.1)),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                norm
            ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, label = self.items[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.tf(img)
        return img, label

# ---------------------------
# 2) Balanced PK Sampler
#    每个 mini-batch 选 P 个类，每类 K 张图
# ---------------------------
class PKSampler(Sampler[List[int]]):
    def __init__(self, labels: List[int], P: int, K: int):
        self.P, self.K = P, K
        self.labels = np.array(labels)
        self.index_by_label: Dict[int, List[int]] = {}
        for i, y in enumerate(self.labels):
            self.index_by_label.setdefault(int(y), []).append(i)
        self.labels_set = list(self.index_by_label.keys())
        # 过滤样本不足 K 的类（也可以在采样时放回）
        self.labels_set = [c for c in self.labels_set if len(self.index_by_label[c]) >= K]
        if len(self.labels_set) < P:
            raise ValueError(f"Not enough classes with >= {K} images. Found {len(self.labels_set)}.")

    def __iter__(self):
        # 无限迭代直到 DataLoader 用完
        while True:
            random.shuffle(self.labels_set)
            for i in range(0, len(self.labels_set), self.P):
                batch_labels = self.labels_set[i:i+self.P]
                if len(batch_labels) < self.P:  # 丢弃不完整 batch 的类组
                    continue
                batch_indices = []
                for c in batch_labels:
                    idxs = self.index_by_label[c]
                    batch_indices.extend(random.sample(idxs, self.K))
                yield batch_indices

    def __len__(self):
        # PyTorch 需要定义长度，用近似值：总样本数 / (P*K)
        return len(self.labels) // (self.P * self.K)

# ---------------------------
# 3) Model: ResNet backbone -> 128D embedding (L2 normalized)
# ---------------------------
class EmbedHead(nn.Module):
    def __init__(self, in_dim=512, embed_dim=128, bn=True):
        super().__init__()
        layers = [nn.Linear(in_dim, embed_dim)]
        if bn:
            layers.append(nn.BatchNorm1d(embed_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.net(x)
        x = F.normalize(x, p=2, dim=1)  # L2 normalize for cosine distance
        return x

class ResNetEmbedding(nn.Module):
    def __init__(self, backbone='resnet18', embed_dim=128, pretrained=True):
        super().__init__()
        assert backbone in ['resnet18','resnet34','resnet50','resnet101']
        if backbone == 'resnet18':
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
            feat_dim = 512
        elif backbone == 'resnet34':
            self.backbone = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
            feat_dim = 512
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
            feat_dim = 2048
        else:
            self.backbone = models.resnet101(weights=models.ResNet101_Weights.DEFAULT if pretrained else None)
            feat_dim = 2048

        # 去掉原 fc，保留全局池化
        self.backbone.fc = nn.Identity()
        self.embed = EmbedHead(in_dim=feat_dim, embed_dim=embed_dim, bn=True)

    def forward(self, x):
        feats = self.backbone(x)           # [B, feat_dim]
        emb = self.embed(feats)            # [B, embed_dim], L2-normalized
        return emb

# ---------------------------
# 4) Batch-hard Triplet Loss
#    参考: "In Defense of the Triplet Loss for Person Re-Identification"
# ---------------------------
def pairwise_distances(embeddings: torch.Tensor, squared=False):
    # embeddings: [B, D], 已 L2 normalize，可用余弦→欧氏距离关系
    dot = embeddings @ embeddings.t()                 # [B,B]
    # 欧氏距离: ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b; 归一化后≈ 2 - 2*cos
    # 这里更通用地从点积构建欧氏距离
    sq_norm = torch.diag(dot)
    dist = sq_norm.unsqueeze(1) - 2*dot + sq_norm.unsqueeze(0)
    dist = torch.clamp(dist, min=0.0)
    if not squared:
        # 为数值稳定，对零对角加 eps 再 sqrt
        mask = (dist == 0.0).float()
        dist = dist + mask * 1e-16
        dist = torch.sqrt(dist)
        dist = dist * (1.0 - mask)
    return dist

def batch_hard_triplet_loss(emb: torch.Tensor, labels: torch.Tensor, margin=0.3, squared=False):
    """
    emb: [B, D], labels: [B]
    对每个 anchor，选 hardest positive & hardest negative (batch 内)
    """
    B = emb.size(0)
    dists = pairwise_distances(emb, squared=squared)  # [B,B]

    labels = labels.unsqueeze(1)  # [B,1]
    is_pos = (labels == labels.t()).float()
    is_neg = 1.0 - is_pos
    # 排除自身
    is_pos = is_pos - torch.eye(B, device=emb.device)

    # hardest positive: 最大正样本距离
    pos_dists = dists * is_pos + (1.0 - is_pos) * (-1e9)
    hardest_pos = pos_dists.max(dim=1).values

    # hardest negative: 最小负样本距离
    neg_dists = dists * is_neg + (1.0 - is_neg) * (1e9)
    hardest_neg = neg_dists.min(dim=1).values

    loss = F.relu(hardest_pos - hardest_neg + margin)
    return loss.mean(), hardest_pos.mean().item(), hardest_neg.mean().item()

# ---------------------------
# 5) 简单评估: Recall@1 (最近邻)
# ---------------------------
def compute_embeddings(model, loader, device):
    model.eval()
    all_emb, all_y = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            emb = model(imgs)
            all_emb.append(emb.cpu())
            all_y.append(labels)
    return torch.cat(all_emb, 0), torch.cat(all_y, 0)

def recall_at_1(emb, labels):
    # 余弦相似进行最近邻检索（排除自身）
    emb = F.normalize(emb, p=2, dim=1)
    sim = emb @ emb.t()  # [N,N]
    N = emb.size(0)
    sim.fill_diagonal_(-1)  # 排除自身
    nn_idx = sim.argmax(dim=1)
    preds = labels[nn_idx]
    r1 = (preds == labels).float().mean().item()
    return r1

# ---------------------------
# 6) 训练脚本
# ---------------------------
def train(
    train_items: List[Tuple[str,int]],
    val_items: List[Tuple[str,int]] = None,
    backbone='resnet18',
    embed_dim=128,
    img_size=224,
    P=8, K=4,                # 一个 batch = P*K
    epochs=20,
    lr=3e-4,
    weight_decay=1e-4,
    margin=0.3,
    num_workers=4,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    train_ds = YourDataset(train_items, train=True, img_size=img_size)
    train_labels = [y for _, y in train_items]
    sampler = PKSampler(train_labels, P=P, K=K)
    train_loader = DataLoader(
        train_ds, batch_sampler=sampler, num_workers=num_workers, pin_memory=True
    )

    val_loader = None
    if val_items:
        val_ds = YourDataset(val_items, train=False, img_size=img_size)
        val_loader = DataLoader(val_ds, batch_size=128, shuffle=False,
                                num_workers=num_workers, pin_memory=True)

    model = ResNetEmbedding(backbone=backbone, embed_dim=embed_dim, pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.startswith('cuda')))
    best_r1 = 0.0

    for epoch in range(1, epochs+1):
        model.train()
        losses, pos_mean, neg_mean = [], [], []
        steps_per_epoch = len(train_loader)
        for step, (imgs, labels) in enumerate(train_loader, start=1):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=(device.startswith('cuda'))):
                emb = model(imgs)
                loss, p_mean, n_mean = batch_hard_triplet_loss(emb, labels, margin=margin)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            losses.append(loss.item())
            pos_mean.append(p_mean)
            neg_mean.append(n_mean)

            # 为了演示，跑够一个 epoch 的 batch 数即可
            if step >= steps_per_epoch:
                break

        scheduler.step()

        msg = (f"Epoch {epoch:02d}/{epochs} | "
               f"Loss {np.mean(losses):.4f} | "
               f"pos_d {np.mean(pos_mean):.3f} | "
               f"neg_d {np.mean(neg_mean):.3f} | "
               f"lr {scheduler.get_last_lr()[0]:.2e}")
        print(msg)

        if val_loader:
            with torch.no_grad():
                emb, y = compute_embeddings(model, val_loader, device)
                r1 = recall_at_1(emb, y)
                print(f"  Val Recall@1: {r1*100:.2f}%")
                if r1 > best_r1:
                    best_r1 = r1
                    os.makedirs("checkpoints", exist_ok=True)
                    torch.save(model.state_dict(), "checkpoints/best_triplet_resnet.pth")
                    print("  (✓) Saved best model.")

    # 训练结束保存最终权重
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/final_triplet_resnet.pth")
    print("Training done. Weights saved to checkpoints/")

# ---------------------------
# 7) 示例入口
# ---------------------------
if __name__ == "__main__":
    # 这里示例构造假的路径和标签。实际使用时替换为你的数据。
    # 比如：train_items = [("data/cat_01/img0001.jpg", 0), ("data/cat_01/img0002.jpg", 0), ("data/cat_02/img0001.jpg", 1), ...]
    train_items = []  # TODO: 填入你的 (path, class_id)
    val_items = []    # 可选
    if len(train_items) == 0:
        raise RuntimeError("Please fill train_items with your (image_path, class_id).")
    train(
        train_items=train_items,
        val_items=val_items if len(val_items) > 0 else None,
        backbone='resnet18',   # 可改 'resnet50'
        embed_dim=128,
        img_size=224,
        P=8, K=4,              # batch size = 32
        epochs=20,
        lr=3e-4,
        weight_decay=1e-4,
        margin=0.3,
        num_workers=4
    )
