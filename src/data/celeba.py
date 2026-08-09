"""CelebA dataset — images + 40 binary attributes."""

import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class CelebADataset(Dataset):
    def __init__(self, data_dir, split="train", image_size=64, crop_size=148):
        self.img_dir = os.path.join(data_dir, "img_align_celeba", "img_align_celeba")
        if not os.path.exists(self.img_dir):
            self.img_dir = os.path.join(data_dir, "img_align_celeba")

        self.split = split
        self.attrs, self.filenames = self._load_data(data_dir, split)

        self.transform = transforms.Compose([
            transforms.CenterCrop(crop_size),
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])

    def _load_data(self, data_dir, split):
        split_map = {"train": 0, "val": 1, "test": 2}

        # Try CSV format (Kaggle) first, then TXT format (original)
        attr_csv = os.path.join(data_dir, "list_attr_celeba.csv")
        attr_txt = os.path.join(data_dir, "list_attr_celeba.txt")
        part_csv = os.path.join(data_dir, "list_eval_partition.csv")
        part_txt = os.path.join(data_dir, "list_eval_partition.txt")

        if os.path.exists(attr_csv):
            attrs_df = pd.read_csv(attr_csv)
            filenames = attrs_df["image_id"].values
            attr_values = attrs_df.drop("image_id", axis=1).values
        else:
            attrs_df = pd.read_csv(attr_txt, sep=r"\s+", skiprows=1)
            filenames = attrs_df.index.values
            attr_values = attrs_df.values

        if os.path.exists(part_csv):
            part_df = pd.read_csv(part_csv)
            partitions = part_df["partition"].values
        else:
            part_df = pd.read_csv(part_txt, sep=r"\s+", header=None, names=["filename", "partition"])
            partitions = part_df["partition"].values

        # Convert {-1, 1} to {0, 1}
        attr_values = (attr_values + 1) // 2

        mask = partitions == split_map[split]
        return attr_values[mask], filenames[mask]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.filenames[idx])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        attrs = torch.tensor(self.attrs[idx], dtype=torch.float32)
        return image, attrs


def get_dataloaders(data_dir, batch_size=128, image_size=64, crop_size=148, num_workers=2):
    train_set = CelebADataset(data_dir, split="train", image_size=image_size, crop_size=crop_size)
    val_set = CelebADataset(data_dir, split="val", image_size=image_size, crop_size=crop_size)
    test_set = CelebADataset(data_dir, split="test", image_size=image_size, crop_size=crop_size)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader