import os
import pickle
import hashlib
import torch
import torch.nn as nn
import numpy as np
import PIL
from PIL import Image
from pathlib import Path
from typing import Union, List


class TensorToNumpy(object):
    def __call__(self, X):
        return X.numpy()
    
    def __repr__(self):
        return self.__class__.__name__ + '()'


class PILToNumpy:
    def __call__(self, img):
        result = np.array(img)
            
        return result

    def __repr__(self):
        return self.__class__.__name__ + '()'



class PILtoHist:
    def __init__(self, bins=256):
        self.bins = bins

    def __call__(self, img):
        # Check the number of channels
        if img.mode == 'RGB':
            num_channels = 3
        elif img.mode == 'L':
            num_channels = 1
        else:
            raise ValueError("Input image mode must be 'RGB' or 'L' (grayscale).")

        histograms = []

        if num_channels == 1:
            # For grayscale images, compute a single histogram
            grayscale_data = np.array(img)
            histogram, _ = np.histogram(grayscale_data, bins=self.bins, range=(0, 256))
            histograms.append(histogram)
        elif num_channels == 3:
            # For RGB images, compute histograms for each channel
            for channel in range(3):
                channel_data = np.array(img)[:, :, channel]
                histogram, _ = np.histogram(channel_data, bins=self.bins, range=(0, 256))
                histograms.append(histogram)

        return np.array(histograms)

    def __repr__(self):
        return f"{self.__class__.__name__}(bins={self.bins})"

       
    
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
        return self.__class__.__name__ + '()'

    
class NumpyToPIL:
    def __call__(self, X):
        img = Image.fromarray(X)
        
        return img
    
    def __repr__(self):
        return self.__class__.__name__ + '()'
    
    
class ProjectTransform:
    """
    A data transformation class that applies a given data projector to transform input data to a lower dimension.

    Parameters:
    -----------
    projector : object
        A dimensionality reduction model or function that implements a 'fit_transform' method to project input data.
        Compatible dimensionality reducers include UMAP, t-SNE, Random Projections, and others following scikit-learn's API.

    Methods:
    --------
    __call__(X)
        Apply the data projection to input data X.

    Attributes:
    -----------
    projector : object
        The dimensionality reduction model used for transforming data.

    Notes:
    ------
    Some dimensionality reducers like UMAP and t-SNE require training before being used with ProjectTransform.
    They must be trained using their 'fit' or 'fit_transform' methods on the entire dataset, as they cannot be trained with mini-batches.
    """
    def __init__(self, projector):
        self.projector = projector

    def __call__(self, X):
        X_trans = self.projector.transform(X)
        
        return X_trans
    
    def __repr__(self):
        return self.__class__.__name__ + '()'
    
    
class FileToPIL:
    def __init__(self, mode: str='RGB'):
        self.mode = mode

    #@log_full
    def __call__(self, file):
        """
        Load and return a PIL image.

        Args:
            file: A single file path.

        Returns:
            A PIL image.

        Examples:
            # Load a single image
            image = File2PIL().transform('path/to/image.jpg')

        """
        if isinstance(file, (str, Path)):
            result = Image.open(str(file)).convert(self.mode)
        else:
            raise ValueError("Invalid input type. Expected str, Path, or list of str/Path.")
            
        return result
 

    def __repr__(self):
        return self.__class__.__name__ + '()'
    

class DummyPIL:
    def __init__(self, dummy=None):
        self.dummy = dummy

        if self.dummy is None:
            array = np.arange(0, 1024, 1, np.uint8)
            array = np.reshape(array, (32, 32))
            self.dummy = Image.fromarray(array)

    #@log_full
    def __call__(self, file):
        """
        Create and return a dummy PIL image.


        Returns:
            A PIL image.

        """            
        return self.dummy
 

    def __repr__(self):
        return self.__class__.__name__ + '()'


class FlattenTensor:
    def __call__(self, tensor):
        return tensor.view(-1)
    
    def __repr__(self):
        return self.__class__.__name__ + '()'
    
    
class ToDevice:
    def __init__(self, device=None):
        if not device:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.device = device

        
    def __call__(self, data):
        return data.to(self.device)

    
    def __repr__(self):
        return self.__class__.__name__ + '()'
    
    
import torch
import torch.nn as nn

class PyTorchEmbedding:
    """
    A utility class for extracting embeddings from a PyTorch model.

    Args:
    model (torch.nn.Module): The PyTorch model from which embeddings will be extracted.
    device (str or torch.device, optional): The device to use for computation. If None, GPU is used if available. Default: None.
    from_layer (int, optional): The index of the starting layer for slicing the model. Default: None.
    to_layer (int, optional): The index of the ending layer for slicing the model. Default: None.

    Methods:
    __slice_model(model, from_layer, to_layer): Slices the model to retain layers from `from_layer` to `to_layer`.
    __auto_slice(model): Automatically removes all linear layers from the end of the model.
    __call__(img): Extracts embeddings from the input image using the configured model.
    """

    def __init__(self, model, device=None, from_layer=None, to_layer=None):
        """
        Initialize the PyTorchEmbedding.

        Args:
        model (torch.nn.Module): The PyTorch model from which embeddings will be extracted.
        device (str or torch.device, optional): The device to use for computation. If None, GPU is used if available. Default: None.
        from_layer (int, optional): The index of the starting layer for slicing the model. Default: None.
        to_layer (int, optional): The index of the ending layer for slicing the model. Default: None.
        """
        if from_layer or to_layer:
            self.model = self.__slice_model(model, from_layer, to_layer)
        else:
            self.model = self.__auto_slice(model)

        # Set the module in evaluation mode
        self.model.eval()

        if not device:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

    def __slice_model(self, model, from_layer=None, to_layer=None):
        """
        Slices the model to retain layers from `from_layer` to `to_layer`.

        Args:
        model (torch.nn.Module): The PyTorch model to be sliced.
        from_layer (int, optional): The index of the starting layer for slicing. Default: None.
        to_layer (int, optional): The index of the ending layer for slicing. Default: None.

        Returns:
        torch.nn.Module: The sliced model.
        """
        # Make model iterable
        mdl = nn.Sequential(*list(model.children()))

        return mdl[from_layer:to_layer]

    def __auto_slice(self, model):
        """
        Automatically removes all linear layers from the end of the model.

        Args:
        model (torch.nn.Module): The PyTorch model to be sliced.

        Returns:
        torch.nn.Module: The model with linear layers removed from the end.
        """
        # Make model iterable
        mdl = nn.Sequential(*list(model.children()))

        last_linear_layer_idx = None

        # Figure out how many linear layers are at the end
        for i, layer in reversed(list(enumerate(mdl))):
            if isinstance(layer, torch.nn.modules.linear.Linear):
                last_linear_layer_idx = i
            else:
                break

        return mdl[:last_linear_layer_idx]

    def __call__(self, img):
        """
        Extract embeddings from the input image using the configured model.

        Args:
        img (torch.Tensor): The input image tensor.

        Returns:
        torch.Tensor: The extracted embeddings.
        """
        if img.dim() != 4:
            # Add a batch dimension to the image tensor
            img = img.unsqueeze(0)

        # Pass the image through the model and get the embeddings
        with torch.no_grad():
            embeddings = self.model(img).detach()  # TODO: Not sure if we even need detach() here

        return embeddings
    
    def __repr__(self):
        return self.__class__.__name__ + '()'


class PyTorchOutput:
    """
    A utility class for obtaining model outputs from a PyTorch model.

    Args:
    model (torch.nn.Module): The PyTorch model to obtain outputs from.
    device (str or torch.device, optional): The device to use for computation. If None, GPU is used if available. Default: None.

    Methods:
    __call__(img): Passes an image through the model and returns the resulting output.
    """

    def __init__(self, model, device=None):
        """
        Initialize the PyTorchOutput.

        Args:
        model (torch.nn.Module): The PyTorch model to obtain outputs from.
        device (str or torch.device, optional): The device to use for computation. If None, GPU is used if available. Default: None.
        """
        self.model = model

        # Set the module in evaluation mode
        self.model.eval()

        if not device:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

    def __call__(self, img):
        """
        Passes an image through the model and returns the resulting output.

        Args:
        img (torch.Tensor): The input image tensor.

        Returns:
        torch.Tensor: The output tensor produced by the model.
        """
        if img.dim() != 4:
            # Add a batch dimension to the image tensor
            img = img.unsqueeze(0)

        # Pass the image through the model and get the output
        with torch.no_grad():
            result = self.model(img).detach()  # TODO: Not sure if we even need detach() here

        return result


    
    def __repr__(self):
        return self.__class__.__name__ + '()'
    
    
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
                print(f"  Size: {v.size}")
        
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
    
    