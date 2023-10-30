import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from ..utils.PathList import PathList


class GenericDataset(Dataset):
    """
    A generic PyTorch Dataset class for working with a list of files.

    Args:
    files (list): List of file paths or data items.
    transforms (callable, optional): A function/transform to apply to the data. Default: None.

    Returns:
    None
    """
    def __init__(self, files, transforms=None, root=None):
        self.files = PathList(files)
        self.full_files = self.files
        self.transforms = transforms

        # Add prefix to all files if necessary
        if root is not None:
            self.full_files = root / self.files


    def __len__(self):
        return len(self.files)

    
    def __getitem__(self, index):
        item = self.full_files[index]

        if self.transforms is not None:
            item = self.transforms(item)

        return {'item': item, 'file': [str(self.files[index].as_posix())]}
