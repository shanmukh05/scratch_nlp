import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader


def create_dataloader(X, y=None, val_split=0.2, batch_size=32, seed=2024, data_type="train"):
    if data_type == "train":
        train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=val_split, random_state=seed)

        train_ds = TensorDataset(torch.Tensor(train_X), torch.Tensor(train_y))
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=1, pin_memory=True)

        val_ds = TensorDataset(torch.Tensor(val_X), torch.Tensor(val_y))
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=1, pin_memory=True)
        return train_loader, val_loader
    else:
        test_ds = TensorDataset(torch.Tensor(X))
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=1, pin_memory=False)
        return test_loader