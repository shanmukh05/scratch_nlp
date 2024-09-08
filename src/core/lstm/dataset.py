import cv2
import torch
import logging
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

class LSTMDataset(Dataset):
    def __init__(self, paths, transforms, tokens=None):
        self.paths = paths
        self.tokens = tokens
        self.transforms = transforms

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        image = cv2.imread(self.paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)["image"]
        if self.tokens is None:
            return image
        
        tokens_ = self.tokens[idx]
        return image, tokens_
    
def create_dataloader(paths, tokens=None, transforms=None, val_split=0.2, batch_size=32, seed=2024, data_type="train"):
    if data_type == "train":
        train_paths, val_paths, train_tokens, val_tokens = train_test_split(paths, tokens, test_size=val_split, random_state=seed)

        train_ds = LSTMDataset(train_paths, transforms[0], train_tokens)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=1, pin_memory=True)

        val_ds = LSTMDataset(val_paths, transforms[1], val_tokens)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=1, pin_memory=True)
        return train_loader, val_loader
    else:
        test_ds = LSTMDataset(paths, transforms)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=1, pin_memory=False)
        return test_loader