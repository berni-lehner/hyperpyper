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
    Transform to extract the color histogram of the input image.
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


def fft2d_magnitude_spec(img: Image.Image) -> np.ndarray:
    """Compute the 2D FFT magnitude spectrum of a grayscale or RGB PIL image.

    Args:
        img (Image.Image): Input image (grayscale or RGB).

    Returns:
        np.ndarray: The magnitude spectrum of the image. For RGB images, returns
            a 3D array where each channel has its own magnitude spectrum.
    """
    # Convert to numpy array
    img_np = np.array(img)

    if img.mode == 'L':
        # Grayscale image
        fft_transform = np.fft.fft2(img_np)
        fft_shifted = np.fft.fftshift(fft_transform)
        magnitude_spec = np.log(np.abs(fft_shifted) + 1)  # Add 1 to avoid log(0)
        return magnitude_spec

    elif img.mode == 'RGB':
        # RGB image, process each channel separately
        magnitude_specs = []
        for channel in range(3):
            fft_transform = np.fft.fft2(img_np[:, :, channel])
            fft_shifted = np.fft.fftshift(fft_transform)
            magnitude_spec = np.log(np.abs(fft_shifted) + 1)
            magnitude_specs.append(magnitude_spec)
        
        # Stack the magnitude spectra for the three channels
        return np.stack(magnitude_specs, axis=-1)

    else:
        raise ValueError("Input image mode must be 'RGB' or 'L' (grayscale).")


class PILtoMagnSpectrum:
    """A transform to compute the magnitude spectrum of a grayscale or RGB PIL image.

    This class can be used in a torchvision.transforms.Compose pipeline.

    Attributes:
        None
    """
    
    def __call__(self, img: Image.Image) -> np.ndarray:
        """Apply the transform to a PIL image.

        Args:
            img (Image.Image): The input image.

        Returns:
            np.ndarray: The magnitude spectrum of the image.
        """
        return fft2d_magnitude_spec(img)

    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'


def radial_average(magnitude_spectrum: np.ndarray) -> np.ndarray:
    """Compute the radial average of a 2D Fourier magnitude spectrum.

    The radial average computes the mean values of the magnitude spectrum along 
    concentric circles around the center of the spectrum, providing a 1D 
    representation of the spectrum.

    Args:
        magnitude_spectrum (np.ndarray): The 2D Fourier magnitude spectrum of an image.
            Should be a 2D array.

    Returns:
        np.ndarray: The 1D radially averaged spectrum.
    """
    if magnitude_spectrum.ndim != 2:
        raise ValueError("Input magnitude_spectrum must be a 2D array.")

    # Get the shape of the input image
    rows, cols = magnitude_spectrum.shape
    center_row, center_col = rows // 2, cols // 2

    # Create a grid of distances from the center
    y, x = np.ogrid[:rows, :cols]
    distance_from_center = np.sqrt((x - center_col)**2 + (y - center_row)**2)

    # Sort distances into integer bins (radial distances)
    distance_bins = np.round(distance_from_center).astype(int)

    # Compute the radial mean: group by distance and average
    radial_avg = np.bincount(distance_bins.ravel(), magnitude_spectrum.ravel()) / np.bincount(distance_bins.ravel())

    return radial_avg


class RadialAvgSpectrum:
    """A transform to compute the radial average of a 2D Fourier magnitude spectrum.

    If the input is a grayscale (2D) magnitude spectrum, it computes the radial average
    directly. If the input is an RGB (3D) magnitude spectrum, it computes the radial
    average for each channel.
    """
    def __call__(self, magnitude_spectrum: np.ndarray) -> np.ndarray:
        """Apply the radial average transform to a magnitude spectrum.

        Args:
            magnitude_spectrum (np.ndarray): Input magnitude spectrum. Can be a 2D
                array for grayscale images or a 3D array for RGB images.

        Returns:
            np.ndarray: The radially averaged spectrum. If the input is grayscale,
                returns a 1D array. If the input is RGB, returns a 2D array where
                each row is the radial average for one channel.
        """
        if magnitude_spectrum.ndim == 2:
            # Grayscale image
            return radial_average(magnitude_spectrum)
        elif magnitude_spectrum.ndim == 3 and magnitude_spectrum.shape[2] == 3:
            # RGB image, process each channel separately
            radial_averages = []
            for channel in range(3):
                radial_avg = radial_average(magnitude_spectrum[:, :, channel])
                radial_averages.append(radial_avg)
            return np.array(radial_averages)
        else:
            raise ValueError("Input must be a 2D grayscale spectrum or a 3D RGB spectrum.")

    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'

    
def spectral_centroid(magnitude_spectrum: np.ndarray) -> float:
    """Compute the spectral centroid of a 1D magnitude spectrum.

    The spectral centroid is the "center of mass" of the spectrum, representing the 
    average frequency weighted by magnitude.

    Args:
        magnitude_spectrum (np.ndarray): The 1D Fourier magnitude spectrum.

    Returns:
        float: The spectral centroid, a weighted mean of the frequency values.

    Raises:
        ValueError: If the input magnitude_spectrum is not a 1D array.
    """
    if magnitude_spectrum.ndim != 1:
        raise ValueError("Input magnitude_spectrum must be a 1D array.")

    frequencies = np.arange(len(magnitude_spectrum))
    centroid = np.sum(frequencies * magnitude_spectrum) / np.sum(magnitude_spectrum)

    return centroid


class SpectralCentroid:
    """A transform to compute the spectral centroid of a Fourier magnitude spectrum.

    If the input is a grayscale (1D) magnitude spectrum, it computes the spectral 
    centroid directly. If the input is an RGB (2D) magnitude spectrum, it computes 
    the spectral centroid for each channel separately.
    """

    def __call__(self, magnitude_spectrum: np.ndarray) -> np.ndarray:
        """Apply the spectral centroid transform to a magnitude spectrum.

        Args:
            magnitude_spectrum (np.ndarray): Input magnitude spectrum. Can be a 1D
                array for grayscale images or a 2D array for RGB images.

        Returns:
            np.ndarray: The spectral centroid. If the input is grayscale, returns a 
                single float value. If the input is RGB, returns a 1D array with the 
                spectral centroid for each channel.
        """
        if magnitude_spectrum.ndim == 1:
            # Grayscale image spectrum
            return spectral_centroid(magnitude_spectrum)
        elif magnitude_spectrum.ndim == 2 and magnitude_spectrum.shape[0] == 3:
            # RGB image spectrum, process each channel separately
            centroids = []
            for channel in range(3):
                centroid = spectral_centroid(magnitude_spectrum[:, channel])
                centroids.append(centroid)
            return np.array(centroids)
        else:
            raise ValueError("Input must be a 1D grayscale spectrum or a 2D RGB spectrum.")

    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'


def spectral_bandwidth(magnitude_spectrum: np.ndarray) -> float:
    """Compute the spectral bandwidth of a 1D magnitude spectrum.

    Spectral bandwidth is the weighted standard deviation of frequencies 
    around the spectral centroid, where the weights are the magnitudes 
    of the frequencies.

    Args:
        magnitude_spectrum (np.ndarray): The 1D Fourier magnitude spectrum.

    Returns:
        float: The spectral bandwidth.

    Raises:
        ValueError: If the input magnitude_spectrum is not a 1D array.
    """
    if magnitude_spectrum.ndim != 1:
        raise ValueError("Input magnitude_spectrum must be a 1D array.")

    frequencies = np.arange(len(magnitude_spectrum))
    centroid = spectral_centroid(magnitude_spectrum)

    # Compute the spectral bandwidth (weighted standard deviation)
    bandwidth = np.sqrt(np.sum((frequencies - centroid)**2 * magnitude_spectrum) / np.sum(magnitude_spectrum))

    return bandwidth


class SpectralBandwidth:
    """A transform to compute the spectral bandwidth of a Fourier magnitude spectrum.

    If the input is a grayscale (1D) magnitude spectrum, it computes the spectral 
    bandwidth directly. If the input is an RGB (2D) magnitude spectrum, it computes 
    the spectral bandwidth for each channel separately.
    """
    def __call__(self, magnitude_spectrum: np.ndarray) -> np.ndarray:
        """Apply the spectral bandwidth transform to a magnitude spectrum.

        Args:
            magnitude_spectrum (np.ndarray): Input magnitude spectrum. Can be a 1D
                array for grayscale images or a 2D array for RGB images.

        Returns:
            np.ndarray: The spectral bandwidth. If the input is grayscale, returns a 
                single float value. If the input is RGB, returns a 1D array with the 
                spectral bandwidth for each channel.
        """
        if magnitude_spectrum.ndim == 1:
            # Grayscale image
            return spectral_bandwidth(magnitude_spectrum)
        elif magnitude_spectrum.ndim == 2 and magnitude_spectrum.shape[1] == 3:
            # RGB image, process each channel separately
            bandwidths = []
            for channel in range(3):
                bandwidth = spectral_bandwidth(magnitude_spectrum[:, channel])
                bandwidths.append(bandwidth)
            return np.array(bandwidths)
        else:
            raise ValueError("Input must be a 2D grayscale spectrum or a 3D RGB spectrum.")

    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'

from scipy.stats import linregress

def spectral_slope(magnitude_spectrum: np.ndarray) -> float:
    """Compute the spectral slope of a 1D magnitude spectrum using linear regression.

    The spectral slope quantifies the rate of change in magnitude as a function 
    of frequency.

    Args:
            magnitude_spectrum (np.ndarray): Input magnitude spectrum. Can be a 2D
                array for grayscale images or a 3D array for RGB images.

    Returns:
        float: The slope of the linear regression on the magnitude spectrum.

    Raises:
        ValueError: If the input magnitude_spectrum is not a 1D array.
    """
    if magnitude_spectrum.ndim != 1:
        raise ValueError("Input magnitude_spectrum must be a 1D array.")

    frequencies = np.arange(len(magnitude_spectrum))

    # Linear regression on the magnitude spectrum
    slope, _, _, _, _ = linregress(frequencies[1:], magnitude_spectrum[1:])
    
    return slope


class SpectralSlope:
    """A transform to compute the spectral slope of a Fourier magnitude spectrum.

    If the input is a grayscale (1D) magnitude spectrum, it computes the spectral 
    slope directly. If the input is an RGB (2D) magnitude spectrum, it computes 
    the spectral slope for each channel separately.
    """
    def __call__(self, magnitude_spectrum: np.ndarray) -> np.ndarray:
        """Apply the spectral slope transform to a magnitude spectrum.

        Args:
            magnitude_spectrum (np.ndarray): Input magnitude spectrum. Can be a 2D
                array for grayscale images or a 3D array for RGB images.

        Returns:
            np.ndarray: The spectral slope. If the input is grayscale, returns a 
                single float value. If the input is RGB, returns a 1D array with the 
                spectral slope for each channel.
        """
        if magnitude_spectrum.ndim == 1:
            # Grayscale image
            return spectral_slope(magnitude_spectrum)
        elif magnitude_spectrum.ndim == 2 and magnitude_spectrum.shape[1] == 3:
            # RGB image, process each channel separately
            slopes = []
            for channel in range(3):
                slope = spectral_slope(magnitude_spectrum[:, channel])
                slopes.append(slope)
            return np.array(slopes)
        else:
            raise ValueError("Input must be a 2D grayscale spectrum or a 3D RGB spectrum.")

    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'


def roll_off_point(magnitude_spectrum: np.ndarray, threshold: float = 0.85) -> int:
    """Compute the roll-off point of a 1D magnitude spectrum.

    The roll-off point is the frequency index where the cumulative sum of the 
    magnitude spectrum reaches a specified percentage (threshold) of the total energy. 

    Args:
        magnitude_spectrum (np.ndarray): The 1D Fourier magnitude spectrum. It should
            be a 1D array, representing the magnitude spectrum of a grayscale image or 
            a single channel from an RGB image.
        threshold (float, optional): The fraction of the total energy to consider for 
            the roll-off point. Default is 0.85 (85%).

    Returns:
        int: The frequency index where the cumulative energy surpasses the 
            threshold percentage of the total energy.

    Raises:
        ValueError: If the input magnitude_spectrum is not a 1D array.
    """
    if magnitude_spectrum.ndim != 1:
        raise ValueError("Input magnitude_spectrum must be a 1D array.")

    total_energy = np.sum(magnitude_spectrum)
    cumulative_energy = np.cumsum(magnitude_spectrum)

    roll_off = np.argmax(cumulative_energy >= threshold*total_energy)

    return roll_off


class SpectralRollOff:
    """A transform to compute the roll-off point of a Fourier magnitude spectrum.

    The roll-off point represents the frequency index where the cumulative energy 
    surpasses a certain percentage (threshold) of the total energy. 

    If the input is a grayscale (1D) magnitude spectrum, it computes the roll-off 
    point directly. If the input is an RGB (2D) magnitude spectrum, it computes the 
    roll-off point for each channel separately.
    """

    def __init__(self, threshold: float = 0.85):
        """
        Initialize the transform.

        Args:
            threshold (float): The fraction of total energy to compute the roll-off point.
                Default is 0.85 (85% of total energy).
        """
        self.threshold = threshold

    def __call__(self, magnitude_spectrum: np.ndarray) -> np.ndarray:
        """Apply the roll-off point transform to a magnitude spectrum.

        Args:
            magnitude_spectrum (np.ndarray): Input magnitude spectrum. Can be a 1D 
                array for grayscale images or a 2D array for RGB images (3 channels).

        Returns:
            np.ndarray: The roll-off point(s). If the input is grayscale, returns a 
                single integer value. If the input is RGB, returns a 1D array with 
                the roll-off point for each channel.
        """
        if magnitude_spectrum.ndim == 1:
            # Grayscale image
            return roll_off_point(magnitude_spectrum, self.threshold)
        elif magnitude_spectrum.ndim == 2 and magnitude_spectrum.shape[1] == 3:
            # RGB image, process each channel separately
            roll_offs = []
            for channel in range(3):
                roll_off = roll_off_point(magnitude_spectrum[:, channel], self.threshold)
                roll_offs.append(roll_off)
            return np.array(roll_offs)
        else:
            raise ValueError("Input must be a 1D grayscale spectrum or a 2D RGB spectrum with 3 channels.")

    def __repr__(self) -> str:
        return (self.__class__.__name__ + f'(threshold={self.threshold})')
