import os
import io
import random
import pickle
import hashlib
import cv2
import torch
import torch.nn as nn
import numpy as np
import PIL
from PIL import Image
from pathlib import Path
from typing import Union, List


class FileToPIL:
    """
    Transform to load a PIL image from a given file.
    """
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
        return f"{self.__class__.__name__}(mode={self.mode})"
    

class DummyPIL:
    """
    Transform to create a dummy image.
    """
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
    """
    Transform to extract the colot histogram of the input image.
    """
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


class PILTranspose:
    """
    Transform to transpose the input image.
    """
    def __call__(self, img):
        """
        Transpose the input image.

        Args:
            img (PIL Image): Input image.

        Returns:
            PIL Image: Transposed image.
        """
        return img.transpose(Image.TRANSPOSE)        

    def __repr__(self):
        return self.__class__.__name__ + '()'



class BWToRandColor(object):
    """
    Transform to replace black and white pixels with random colors in an RGB image.
    """
    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Apply the transformation to the input image.

        Parameters:
            img (PIL.Image.Image): Input grayscale image in "RGB" mode.

        Returns:
            PIL.Image.Image: Transformed image with replaced black and white pixels.
        """
        if img.mode != "RGB":
            raise ValueError("Input image mode must be 'RGB'.")            
        else:
            img_array = np.array(img)

            # Generate random colors for black and white replacement
            color1 = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            color2 = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

            # Create a mask for black pixels
            black_mask = (img_array[:, :, 0] == 0) & (img_array[:, :, 1] == 0) & (img_array[:, :, 2] == 0)

            # Create a mask for white pixels
            white_mask = (img_array[:, :, 0] == 255) & (img_array[:, :, 1] == 255) & (img_array[:, :, 2] == 255)

            # Replace black pixels with random color1
            img.paste(color1, None, mask=Image.fromarray(black_mask))

            # Replace white pixels with random color2
            img.paste(color2, None, mask=Image.fromarray(white_mask))

            # Convert the array back to a PIL image
            img = Image.fromarray(np.uint8(img))

        return img


    def __repr__(self):
        return self.__class__.__name__ + '()'



class GrayToRandColor(object):
    """
    Transform to interpolate colors based on grayscale intensity in an RGB image.
    """
    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Apply the transformation to the input image.

        Parameters:
            img (PIL.Image.Image): Input grayscale image in "RGB" mode.

        Returns:
            PIL.Image.Image: Transformed image with colors interpolated based on grayscale intensity.
        """
        if img.mode != "RGB":
            raise ValueError("Input image mode must be 'RGB'.")            
        else:
            img_array = np.array(img)

            # Generate random colors for smooth fade
            color1 = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            color2 = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

            # Calculate grayscale intensity
            gray_intensity = np.mean(img_array, axis=-1)

            # Normalize the intensity to range [0, 1]
            normalized_intensity = gray_intensity / 255.0

            # Interpolate between color1 and color2 based on intensity
            interpolated_colors = (
                (1 - normalized_intensity) * color1[0] + normalized_intensity * color2[0],
                (1 - normalized_intensity) * color1[1] + normalized_intensity * color2[1],
                (1 - normalized_intensity) * color1[2] + normalized_intensity * color2[2]
            )

            # Replace pixels with interpolated colors
            img_array[:, :, 0] = interpolated_colors[0]
            img_array[:, :, 1] = interpolated_colors[1]
            img_array[:, :, 2] = interpolated_colors[2]

            # Convert the array back to a PIL image
            img = Image.fromarray(np.uint8(img_array))

        return img


    def __repr__(self):
        return self.__class__.__name__ + '()'



class RandomPixelInvert:
    """
    Transform to randomly invert a proportion of pixels in an image.
    """
    def __init__(self, p: Union[int, float] = 0.1):
        """
        Initialize the RandomPixelInvert.

        Parameters:
            p (Union[int, float]): Proportion of pixels to invert, ranging from 0 to 1.
                If an integer is provided, it represents the exact number of pixels to invert.
                If a float is provided, it represents the proportion of pixels to invert relative to the total number of pixels in the image.
        """
        self.num_pixels_to_invert = 0
        if isinstance(p, int):
            self.num_pixels_to_invert = p

        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Apply the random invert transformation to the input image.

        Parameters:
            image (PIL.Image): Input image.

        Returns:
            PIL.Image: Image with randomly inverted pixels.
        """
        if img.mode != "RGB":
            raise ValueError("Input image mode must be 'RGB'.") 
        else:
            image_array = np.array(img)

            if not self.num_pixels_to_invert:
                # Determine the number of pixels to invert
                self.num_pixels_to_invert = int(self.p * image_array.size / 3)  # Divide by 3 for RGB channels

            random_indices = np.random.choice(image_array.size // 3, size=self.num_pixels_to_invert, replace=False)

            # Separate channels
            channels = [image_array[:, :, i] for i in range(3)]

            # Invert selected pixels for each channel
            for channel in channels:
                channel.flat[random_indices] = 255 - channel.flat[random_indices]

            # Merge channels back
            inverted_image_array = np.stack(channels, axis=-1)

            # Convert the array back to PIL image
            img = Image.fromarray(inverted_image_array)

        return img

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p})"



class PixelInvert:
    """
    Transform to invert a given set of pixels in an image.
    """
    def __init__(self, pos=[0]):
        """
        Initialize the PixelInvert.

        Parameters:
        """
        self.pos = pos

    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Apply the random invert transformation to the input image.

        Parameters:
            image (PIL.Image): Input image.

        Returns:
            PIL.Image: Image with randomly inverted pixels.
        """
        if img.mode != "RGB":
            raise ValueError("Input image mode must be 'RGB'.") 
        else:
            image_array = np.array(img)

            indices = np.array(self.pos)

            # Separate channels
            channels = [image_array[:, :, i] for i in range(3)]

            # Invert selected pixels for each channel
            for channel in channels:
                channel.flat[indices] = 255-channel.flat[indices]

            # Merge channels back
            inverted_image_array = np.stack(channels, axis=-1)

            # Convert the array back to PIL image
            img = Image.fromarray(inverted_image_array)

        return img

    def __repr__(self):
        return f"{self.__class__.__name__}(pos={self.pos})"


# TODO: set values at once, not for each channel separately; test with pos list;
class PixelSet:
    """
    Transform to set the color of a given set of pixels in an image.
    """
    def __init__(self, pos=[0], color=(0,0,0)):
        """
        Initialize the PixelSet.
        """
        self.pos = pos
        self.color = color

    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Apply the transformation to the input image.

        Parameters:
            image (PIL.Image): Input image.

        Returns:
            PIL.Image: Image with pixels set to a specific color.
        """
        if img.mode != "RGB":
            raise ValueError("Input image mode must be 'RGB'.") 
        else:
            image_array = np.array(img)

            indices = np.array(self.pos)

            # Separate channels
            channels = [image_array[:,:,i] for i in range(3)]

            # Set color for selected pixels for each channel
            for channel in channels:
                channel.flat[indices] = self.color

            # Merge channels back
            inverted_image_array = np.stack(channels, axis=-1)

            # Convert the array back to PIL image
            img = Image.fromarray(inverted_image_array)

        return img

    def __repr__(self):
        return f"{self.__class__.__name__}(pos={self.pos}, color={self.color})"

# TODO: add parameters for each side to swithc on/off
class DrawFrame:
    """
    Draw a frame around the outer edges of the image.
    """
    def __init__(self, width=1, color=(0,0,0)):
        """
        Initialize the DrawFrame transformation.

        Args:
            width (int): Width of the frame.
            color (tuple): RGB values of the frame color.
        """
        self.width = width
        self.color = color

    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Apply the transformation to the input image.

        Parameters:
            img (PIL.Image): Input image.

        Returns:
            PIL.Image: Image with a frame drawn around its outer edges.
        """
        if img.mode != "RGB":
            raise ValueError("Input image mode must be 'RGB'.")
        else:
            img_array = np.array(img)

            # Draw the frame on the top and bottom edges
            img_array[:self.width, :] = self.color
            img_array[-self.width:, :] = self.color

            # Draw the frame on the left and right edges
            img_array[:, :self.width] = self.color
            img_array[:, -self.width:] = self.color

            # Convert the modified array back to PIL image
            img = Image.fromarray(img_array)

        return img

    def __repr__(self):
            return f"{self.__class__.__name__}(width={self.width}, color={self.color})"



class JPEGCompressionTransform:
    """
    Transform to compress an image (PIL image or torch tensor) as JPEG with specified compression quality.
    """
    def __init__(self, quality: int=75):
        """
        Initialize the JPEGCompressionTransform.

        Parameters:
            quality (int): Compression quality level, ranging from 0 to 100.
                0 represents the lowest quality and highest compression,
                while 100 represents the highest quality and lowest compression.
        """
        self.quality = int(quality)

    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Apply JPEG compression to the input image.

        Parameters:
            image (PIL.Image or torch.Tensor): Input image.

        Returns:
            torch.Tensor: Compressed image as a PNG-like tensor.
        """
        # Create an in-memory stream to hold the compressed image
        with io.BytesIO() as output:
            # Save the image to the stream with JPEG format and specified quality
            img.save(output, format='JPEG', quality=self.quality)
            # Rewind the stream to the beginning
            output.seek(0)
            # Open the compressed image from the stream in RGB mode
            img = Image.open(output).convert('RGB')
        return img

    def __repr__(self):
        return f"{self.__class__.__name__}(quality={self.quality})"




class WEBPCompressionTransform:
    """
    Transform to compress an image (PIL image or torch tensor) as WEBP with specified compression quality.
    """
    def __init__(self, quality: int=80):
        """
        Initialize the WEBPCompressionTransform.

        Parameters:
            quality (int): Compression quality level, ranging from 0 to 100.
                0 represents the lowest quality and highest compression,
                while 100 represents the highest quality and lowest compression.
        """
        self.quality = int(quality)

    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Apply WEBP compression to the input image.

        Parameters:
            image (PIL.Image or torch.Tensor): Input image.

        Returns:
            torch.Tensor: Compressed image as a PNG-like tensor.
        """
        # Create an in-memory stream to hold the compressed image
        with io.BytesIO() as output:
            # Save the image to the stream with JPEG format and specified quality
            img.save(output, format='WEBP', quality=self.quality, lossless=False)
            # Rewind the stream to the beginning
            output.seek(0)
            # Open the compressed image from the stream in RGB mode
            img = Image.open(output).convert('RGB')
        return img

    def __repr__(self):
        return f"{self.__class__.__name__}(quality={self.quality})"        

# TODO: JPEG 2000
#image.save(output, format='JP2', quality_mode='dB', quality_layers=[80])  # Example with quality level 80 dB



class PILAxisSwap(object):
    """
    Transform to randomly swap full rows or columns in the given image along the specified axis.
    """
    def __init__(self, axis=0, n_swaps=1):
        """
        Initialize the transform.

        Args:
            axis (int): Axis along which to swap rows or columns (0 for rows, 1 for columns, default is 0).
            n_swaps (int): The number of rows/columns to be swapped (default is 1).
        """
        self.axis = axis
        self.n_swaps = n_swaps

    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Apply the transformation to the input image.

        Args:
            img (PIL.Image.Image): The input PIL image.

        Returns:
            PIL.Image.Image: The modified PIL image with swapped rows/columns.
        """
        if not isinstance(img, Image.Image):
            raise TypeError("Input 'img' must be a PIL Image object.")
        
        # Convert to a numpy array
        image_array = np.array(img)
        
        if self.axis not in [0, 1]:
            raise ValueError("axis must be 0 for rows or 1 for columns")
        
        # Get the total number of rows or columns
        num_elements = image_array.shape[self.axis]
        
        if self.n_swaps * 2 > num_elements:
            raise ValueError("num_swaps is too large for the size of the image along the specified axis")

        # Pre-sample all necessary pairs of indices
        indices = list(range(num_elements))
        random.shuffle(indices)
        swap_pairs = [(indices[i], indices[i + 1]) for i in range(0, self.n_swaps * 2, 2)]
        
        # Perform the swaps
        for idx1, idx2 in swap_pairs:
            if self.axis == 0:  # Swap rows
                image_array[[idx1, idx2], :] = image_array[[idx2, idx1], :]
            elif self.axis == 1:  # Swap columns
                image_array[:, [idx1, idx2]] = image_array[:, [idx2, idx1]]
        
        # Convert the modified numpy array back to a PIL image
        modified_image = Image.fromarray(image_array)
        
        return modified_image

    def __repr__(self):
        return self.__class__.__name__ + f'(axis={self.axis}, num_swaps={self.n_swaps})'



class PILToEdgeAngleHist(object):
    """
    Transform to detect predominant edge angle using gradient orientation histogram.
    """
    def __init__(self, bins=10, startangle=0, counterclock=True, threshold=0.0, ksize=5):
        """
        Initialize the transform.

        Args:
            bins (int): Number of bins for the histogram.
            startangle (int): The offset angle to start the histogram.
            counterclock (bool): Direction of angle mapping. True for counterclockwise, False for clockwise.
            threshold (float): Proportion of gradients with respect to magnitude to be kept, between 0.0 and 1.0.
            ksize (int): Size of the extended Sobel kernel.
        """
        self.bins = bins
        self.startangle = startangle
        self.counterclock = counterclock
        self.threshold = threshold
        self.ksize = ksize

    def __call__(self, img: Image.Image) -> np.ndarray:
        """
        Apply the transformation to the input image.

        Args:
            img (PIL.Image.Image): The input PIL image.

        Returns:
            np.ndarray: The histogram of edge orientations.
        """
        if not isinstance(img, Image.Image):
            raise TypeError("Input 'img' must be a PIL Image object.")
        if not isinstance(self.startangle, int):
            raise ValueError("Starting angle 'startangle' must be an integer.")
        if not 0 <= self.startangle <= 359:
            raise ValueError("Starting angle 'startangle' value must be between 0 and 359.")
        if not isinstance(self.counterclock, bool):
            raise ValueError("Parameter 'counterclock' must be a boolean.")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("Threshold value must be between 0.0 and 1.0.")

        # Convert to grayscale
        image_array = np.array(img.convert('L'))

        # Compute gradients
        grad_x = cv2.Sobel(image_array, cv2.CV_64F, 1, 0, ksize=self.ksize)
        grad_y = cv2.Sobel(image_array, cv2.CV_64F, 0, 1, ksize=self.ksize)

        # Compute gradient magnitude and orientation
        magnitude = cv2.magnitude(grad_x, grad_y)
        orientation = cv2.phase(grad_x, grad_y, angleInDegrees=True)

        # Apply offset and direction to orientation and 
        # Collapse angles that are 180 degrees apart into the same bins
        if self.counterclock:
            orientation = (360-orientation+self.startangle) % 180
        else:
            orientation = (orientation+self.startangle) % 180

        # Mask out magnitudes below the relative threshold
        normalized_magnitude = magnitude/np.max(magnitude)
        valid_mask = normalized_magnitude>=self.threshold
        filtered_orientation = orientation[valid_mask]

        # Create histogram of orientations
        histogram, _ = np.histogram(filtered_orientation, bins=self.bins)

        return histogram

    def __repr__(self):
        return (self.__class__.__name__ + 
                f'(bins={self.bins}, startangle={self.startangle}, '
                f'counterclock={self.counterclock}, threshold={self.threshold}, ksize={self.ksize})')
