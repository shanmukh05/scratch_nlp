import cv2
import torch
import logging
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

class LSTMDataset(Dataset):
    def __init__(self, paths, transforms, tokens=None):
        """
        _summary_

        :param paths: _description_
        :type paths: _type_
        :param transforms: _description_
        :type transforms: _type_
        :param tokens: _description_, defaults to None
        :type tokens: _type_, optional
        """        
        self.paths = paths
        self.tokens = tokens
        self.transforms = transforms

    def __len__(self):
        """
        _summary_

        :return: _description_
        :rtype: _type_
        """        
        return len(self.paths)
    
    def __getitem__(self, idx):
        """
        _summary_

        :param idx: _description_
        :type idx: _type_
        :return: _description_
        :rtype: _type_
        """        
        image = cv2.imread(self.paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)["image"]
        if self.tokens is None:
            return image
        
        tokens_ = self.tokens[idx]
        return image, tokens_
    
def create_dataloader(paths, tokens=None, transforms=None, val_split=0.2, batch_size=32, seed=2024, data_type="train"):
    """
    _summary_

    :param paths: _description_
    :type paths: _type_
    :param tokens: _description_, defaults to None
    :type tokens: _type_, optional
    :param transforms: _description_, defaults to None
    :type transforms: _type_, optional
    :param val_split: _description_, defaults to 0.2
    :type val_split: float, optional
    :param batch_size: _description_, defaults to 32
    :type batch_size: int, optional
    :param seed: _description_, defaults to 2024
    :type seed: int, optional
    :param data_type: _description_, defaults to "train"
    :type data_type: str, optional
    :return: _description_
    :rtype: _type_
    """    
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