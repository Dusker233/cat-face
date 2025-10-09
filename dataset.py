from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

from typing import List, Tuple


class ImageDataset(Dataset):
    def __init__(
        self, items: List[Tuple[str, str]], train: bool = True, image_size: int = 224
    ):
        self.items = items
        norm = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        if train:
            self.tf = transforms.Compose(
                [
                    transforms.Resize(int(image_size * 1.1)),
                    transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
                    transforms.ToTensor(),
                    norm,
                ]
            )
        else:
            self.tf = transforms.Compose(
                [
                    transforms.Resize(int(image_size * 1.1)),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    norm,
                ]
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, label = self.items[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.tf(img)
        return img, label
