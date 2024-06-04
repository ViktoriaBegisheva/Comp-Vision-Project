import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np


class CustomDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.img_names = [f for f in os.listdir(img_dir) if
                          f.endswith('.jpg') and os.path.exists(os.path.join(label_dir, f.replace('.jpg', '.txt')))]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name.replace('.jpg', '.txt'))

        image = Image.open(img_path).convert("RGB")

        with open(label_path, 'r') as f:
            labels = f.readlines()

        labels = [list(map(float, label.strip().split())) for label in labels]
        labels = np.array(labels)

        if self.transform:
            image = self.transform(image)

        return image, labels

