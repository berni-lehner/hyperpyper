import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader



class GenericDataset(Dataset):
    """
    A generic PyTorch Dataset class for working with a list of files.

    Args:
    files (list): List of file paths or data items.
    transforms (callable, optional): A function/transform to apply to the data. Default: None.

    Returns:
    None
    """
    def __init__(self, files, transforms=None):
        self.files = files
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    
    def __getitem__(self, index):
        item = self.files[index]
        
        if self.transforms is not None:
            item = self.transforms(item)
        
        sample = {'item': item, 'file': [str(self.files[index])]}

        return sample
