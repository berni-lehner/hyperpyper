import os
import pickle
import hashlib
import torch
import torch.nn as nn
import numpy as np
import PIL
from PIL import Image
from pathlib import Path
from typing import Union, List, Dict, Any


class FlattenArray:
    def __call__(self, img):
        if len(img.shape) == 2:
            # Single-channel image: Flatten directly
            return img.flatten()
        else:
            # Multi-channel image: Flatten each channel and concatenate
            return np.concatenate([channel.flatten() for channel in img], axis=0)
    
    def __repr__(self):
        return self.__class__.__name__ + '()'
    
#TODO: test
class FlattenList:
    def __call__(self, img_list):
        flattened_list = []
        for img in img_list:
            if isinstance(img, list):
                flattened_list.extend([item for sublist in img for item in sublist])
            else:
                flattened_list.extend(img.flatten())
                
        return np.array(flattened_list)

    def __repr__(self):
        return self.__class__.__name__ + '()'

    
class ReshapeArray:
    def __init__(self, shape):
        self.shape = shape

    def __call__(self, X):
        X_reshaped = np.reshape(X, self.shape)
        
        return X_reshaped

    def __repr__(self):
        return f"{self.__class__.__name__}(shape={self.shape})"
    

class ToArgMax:
    """
    """
    def __init__(self, axis=None):
        self.axis = axis

    def __call__(self, X):        
        return X.argmax(self.axis).item()
    
    def __repr__(self):
        return f"{self.__class__.__name__}(axis={self.axis})"


class ToLabel:
    """
    """
    def __init__(self, labels=None):
        self.labels = labels

    def __call__(self, X):        
        return self.labels[X]
    
    def __repr__(self):
        return f"{self.__class__.__name__}(labels={self.labels})"

# TODO: clean up type checks (Image, PIL.Image)
class DebugTransform:
    """
    A utility class for debugging transformations by printing input arguments and their characteristics.

    Args:
    transform (callable): The transformation function to be debugged.

    Methods:
    __call__(*args, **kwargs): Calls the transformation function and prints debug information about the input arguments.
    """

    def __init__(self, transform):
        """
        Initialize the DebugTransform.

        Args:
        transform (callable): The transformation function to be debugged.
        """
        self.transform = transform

    def __call__(self, *args, **kwargs):
        """
        Calls the transformation function and prints debug information about the input arguments.

        Args:
        *args: Positional arguments passed to the transformation function.
        **kwargs: Keyword arguments passed to the transformation function.

        Returns:
        Any: The result of the transformation function.
        """
        print(f"Transform: {self.transform.__class__.__name__}")
        print("Input Arguments:")
        
        # Process args
        for i, arg in enumerate(args):
            print(f"Argument {i+1}: {type(arg)}")
            
            if isinstance(arg, list):
                print(f"  Length: {len(arg)}")
                for j, item in enumerate(arg):
                    print(f"  Item {j+1}: {type(item)}")
                    if isinstance(item, (torch.Tensor, np.ndarray)):
                        print(f"    Shape: {item.shape}")
                    elif isinstance(item, Image):
                        print(f"    Size: {item.size}")
            elif isinstance(arg, (torch.Tensor, np.ndarray)):
                print(f"  Shape: {arg.shape}")
            elif isinstance(arg, PIL.Image.Image):
                print(f"  Channels: {len(arg.getbands())}, Size: {arg.size}")
        
        # Process kwargs
        for k, v in kwargs.items():
            print(f"Argument '{k}': {type(v)}")
            if isinstance(v, (torch.Tensor, np.ndarray)):
                print(f"  Shape: {v.shape}")
            elif isinstance(v, PIL.Image.Image):
                print(f"  Channels: {len(v.getbands())}, Size: {v.size}")
                #print(f"  Size: {v.size}")
        
        return self.transform(*args, **kwargs)
    

class CachingTransform:
    """
    A utility class for applying a transformation with caching of results.

    Args:
    transform (callable): The transformation function to be cached.
    cache_folder (str): The folder to store cached results.
    verbose (bool, optional): If True, print caching and loading information. Default: False.

    Methods:
    __call__(data): Applies the transformation with caching and returns the result.
    _get_cache_filename(data): Generates a unique cache filename based on the data.
    """

    def __init__(self, transform, cache_folder, verbose=False):
        """
        Initialize the CachingTransform.

        Args:
        transform (callable): The transformation function to be cached.
        cache_folder (str): The folder to store cached results.
        verbose (bool, optional): If True, print caching and loading information. Default: False.
        """
        self.transform = transform
        self.cache_folder = cache_folder
        self.verbose = verbose

        os.makedirs(cache_folder, exist_ok=True)

    def __call__(self, data):
        """
        Applies the transformation with caching and returns the result.

        Args:
        data: The input data to be transformed.

        Returns:
        Any: The result of the transformation.
        """
        cache_file = self._get_cache_filename(data)
        if os.path.exists(cache_file):
            if self.verbose:
                print(f"Loading cached result ...")
            # Load the cached result if it exists
            with open(cache_file, 'rb') as f:
                result = pickle.load(f)
        else:
            if self.verbose:
                print(f"Caching result ...")

            # Compute the result using the transform and store it in cache
            result = self.transform(data)
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)

        return result

    def _get_cache_filename(self, data):
        """
        Generates a unique cache filename based on the data.

        Args:
        data: The input data to be transformed.

        Returns:
        str: The cache filename.
        """
        # Generate a unique cache filename based on the transform name and data hash
        t_name = type(self.transform).__name__

        # Compute the hash of the binary representation
        if isinstance(data, np.ndarray):
            hash_value = hashlib.md5(data.tobytes()).hexdigest()
        elif isinstance(data, torch.Tensor):
            hash_value = hashlib.sha256(data.cpu().detach().numpy().tobytes()).hexdigest()
        else:
            raise ValueError("Invalid input type. Expected np.ndarray or torch.Tensor.")

        filename = f"{t_name}__{hash_value}.pkl"
        if self.verbose:
            print(filename)

        return os.path.join(self.cache_folder, filename)


class FeatureUnion:
    """A custom feature union to pass an arbitrary input to multiple transforms.
    
    The input is passed to all the provided transforms, and the results are returned
    as a dictionary with the transform names as keys.
    
    Args:
        transforms (List[Any]): A list of transforms to apply to the input.
    """
    def __init__(self, transforms: List[Any]):
        self.transforms = transforms


    def __call__(self, input_data: Any) -> Dict[str, Any]:
        """Apply all transforms to the input data and return the aggregated results.
        
        Args:
            input_data (Any): The input to be passed to the transforms.
        
        Returns:
            Dict[str, Any]: A dictionary of extracted features, where the keys are 
                the string representation of the transforms and the values are the 
                transformed results.
        """
        features = {}

        # Apply each transform to the input data and aggregate the results
        for transform in self.transforms:
            feature_name = str(transform)
            features[feature_name] = transform(input_data)
        
        return features

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(transforms={self.transforms})"