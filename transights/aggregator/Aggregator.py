import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from ..dataset import GenericDataset
from ..utils import Pickler


class DataLoaderAggregator:
    """
    Aggregates mini-batches from a DataLoader to create a full batch.

    Args:
    data_loader (torch.utils.data.DataLoader): A DataLoader containing mini-batches.

    Attributes:
    data_loader (torch.utils.data.DataLoader): The input DataLoader.
    _full_batch (dict): The aggregated full batch.

    Methods:
    collate_fn(batch_list): Aggregates a list of mini-batches into a full batch.
    fit(): Alias for transform().
    fit_transform(): Alias for transform().
    transform(): Aggregates mini-batches and returns the full batch.
    """
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self._full_batch = None


    def collate_fn(self, batch_list):
        """
        Aggregates a list of mini-batches into a full batch.

        Args:
        batch_list (list): A list of dictionaries, each representing a mini-batch.

        Returns:
        dict: The aggregated full batch.
        """
        full_batch = {}

        # Separate items from the batch_list
        for batch in batch_list:
            for key, value in batch.items():
                if key not in full_batch:
                    full_batch[key] = []

                # Check if the values are list of tuples (specifically, filenames)
                if isinstance(value[0], tuple):
                    # Flatten the list of tuples and append to the full_batch
                    full_batch[key].extend(sum(value, ()))
                else:
                    # Append other items directly to the full_batch
                    full_batch[key].append(value)

        # Process each key-value pair in the full_batch dictionary
        for key, value in full_batch.items():
            # Check if the values are of type Tensor, Numpy array, or list of Tensors
            if isinstance(value[0], torch.Tensor):
                full_batch[key] = torch.cat(value, dim=0)
            elif isinstance(value[0], np.ndarray):
                full_batch[key] = np.concatenate(value, axis=0)
            elif isinstance(value[0], list) and isinstance(value[0][0], torch.Tensor):
                full_batch[key] = [torch.cat(v, dim=0) for v in zip(*value)]

        return full_batch

    
    def fit(self, cache_file=None):
        """
        Alias for transform(). Aggregates mini-batches and returns the full batch.

        Returns:
        dict: The aggregated full batch.
        """
        return self.transform(cache_file)

    
    def fit_transform(self, cache_file=None):
        """
        Alias for transform(). Aggregates mini-batches and returns the full batch.

        Returns:
        dict: The aggregated full batch.
        """
        return self.transform(cache_file)

    
    def transform(self, cache_file=None):
        """
        Aggregates mini-batches and returns the full batch.

        Returns:
        dict: The aggregated full batch.
        """
        if cache_file is not None:
            if Path(cache_file).exists():
                self._full_batch = Pickler.load_data(cache_file)
            else:
                self._full_batch = self.__transform()
                Pickler.save_data(self._full_batch, cache_file)
        else:
            self._full_batch = self.__transform()
            
        return self._full_batch        
 
            
    def __transform(self):
        # Avoid unnecessary transforms
        if self._full_batch is None:
            mini_batches = []

            # Iterate over the DataLoader and aggregate all mini batches
            for batch in self.data_loader:
                mini_batches.append(batch)

            # Turn the list of mini batches into a full batch
            self._full_batch = self.collate_fn(mini_batches)

        return self._full_batch        


class DataAggregator(DataLoaderAggregator):
    """
    Aggregates data from a list of files using a GenericDataset and DataLoader.

    Args:
    files (list): List of file paths or data items.
    transforms (callable): A function/transform to apply to the data.
    batch_size (int, optional): Batch size for DataLoader. Default: 8.

    Methods:
    collate_fn(batch_list): Aggregates a list of mini-batches into a full batch.
    fit(): Alias for transform().
    fit_transform(): Alias for transform().
    transform(): Aggregates mini-batches and returns the full batch.
    """

    def __init__(self, files, transforms, batch_size=8):
        """
        Initialize the DataAggregator.

        Args:
        files (list): List of file paths or data items.
        transforms (callable): A function/transform to apply to the data.
        batch_size (int, optional): Batch size for DataLoader. Default: 8.
        """
        data_set = GenericDataset(files, transforms)
        data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=False)
        super().__init__(data_loader)


class DataSetAggregator(DataLoaderAggregator):
    """
    Aggregates data from a given dataset using DataLoader.

    Args:
    data_set (torch.utils.data.Dataset): The input dataset.
    batch_size (int, optional): Batch size for DataLoader. Default: 8.

    Methods:
    collate_fn(batch_list): Aggregates a list of mini-batches into a full batch.
    fit(): Alias for transform().
    fit_transform(): Alias for transform().
    transform(): Aggregates mini-batches and returns the full batch.
    """

    def __init__(self, data_set, batch_size=8):
        """
        Initialize the DataSetAggregator.

        Args:
        data_set (torch.utils.data.Dataset): The input dataset.
        batch_size (int, optional): Batch size for DataLoader. Default: 8.
        """
        data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=False)
        super().__init__(data_loader)
            