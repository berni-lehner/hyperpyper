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


class FileToPIL:
    def __init__(self, mode: str='RGB'):
        self.mode = mode

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

    def __call__(self, file):
        """
        Create and return a dummy PIL image.


        Returns:
            A PIL image.

        """            
        return self.dummy
 

    def __repr__(self):
        return self.__class__.__name__ + '()'


class PILToNumpy:
    def __call__(self, img):
        result = np.array(img)
            
        return result

    def __repr__(self):
        return self.__class__.__name__ + '()'


class NumpyToPIL:
    def __call__(self, X):
        img = Image.fromarray(X)
        
        return img
    
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
