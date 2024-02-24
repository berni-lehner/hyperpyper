import numpy as np
from typing import List, Tuple, Union, Optional
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


class ImageFilePlotter:
    def __init__(self, files,
                title_labels = None,
                title_prefix = '',
                title_postfix = '',
                mode: str='RGB',
                grid_layout: bool = True,
                rotate: bool = False,
                frame: Optional[bool] = True,
                title: Union[str, None] = None,
                title_fontsize: Union[int, None] = None, 
                figsize: Union[Tuple[float, float], None] = None):
        """
        Draws the images corresponding to the given files in the specified layout.

        Args:
           files (List[str]): List of file paths for the images to be plotted.
            title_labels (bool, optional): If True, display labels as titles for each sample. Default is False.
            title_prefix (str): Prefix string to be added to each title.
            title_postfix (str): Postfix string to be added to each title.
            mode (str): Mode for opening images. Default is 'RGB'.
            grid_layout (bool): Whether to display the subplots as a grid.
            rotate (bool): Whether to rotate the layout by 90 degrees.
            frame (bool, optional): If True, display frames around the images. Default is True.
            title (str): Super title for the entire plot.
            title_fontsize (int): Fontsize for the super title.
            figsize (tuple): The size of the entire figure in inches (width, height).
        """
        self.files = files
        self.title_labels = title_labels
        self.title_prefix = title_prefix
        self.title_postfix = title_postfix
        self.mode = mode
        self.n_samples: int = len(files)
        self.grid_layout: bool = grid_layout
        self.rotate: bool = rotate
        self.frame: bool = frame
        self.title: Union[str, None] = title
        self.title_fontsize: Union[int, None] = title_fontsize
        self.figsize: Union[Tuple[float, float], None] = figsize


    def plot(self) -> Figure:
        """
        Draws the images corresponding to the files.
        """
        # Create the layout of subplots
        if self.grid_layout:
            fig, axes = self.create_subplots()
        else:
            if self.rotate:
                fig, axes = plt.subplots(nrows=1, ncols=self.n_samples)
            else:
                fig, axes = plt.subplots(nrows=self.n_samples, ncols=1)

        # Flatten the axes array for easier indexing
        axes = axes.flatten()

        for i in range(self.n_samples):
            ax = axes[i]

            img = Image.open(str(self.files[i])).convert(self.mode)
            ax.imshow(img)

            # Format and set title string
            if self.title_labels:
                lbl = self.title_labels[i]
                if(isinstance(lbl, (float, np.float16, np.float32))):
                    lbl = f"{self.title_labels[i]:.3f}"
                ax.set_title(f"{self.title_prefix}{lbl}{self.title_postfix}")

            ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

            if not self.frame:
                ax.axis("off")

        if self.title is not None:
            plt.suptitle(self.title, fontsize=self.title_fontsize)

        fig.tight_layout()

        return fig


    def create_subplots(self):
        """
        Creates subplots in a grid layout.

        This function calculates the number of rows and columns based on the desired number of subplots.
        The subplots are then created using the `subplots` function from Matplotlib.

        Returns:
            matplotlib.figure.Figure: The generated figure object.
            numpy.ndarray: Flattened array of axes objects representing the subplots.
        """
        square_len = np.sqrt(self.n_samples)
        # we sometimes need an additional row depending on the rotation and the number of subplots
        row_appendix = int(bool(np.remainder(self.n_samples,square_len))*self.rotate)

        nrows = int(square_len) + row_appendix
        ncols = int((self.n_samples+nrows-1) // nrows)
        
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=self.figsize)

        # Flatten the axes array for easier indexing
        axes = axes.flatten()

        # Remove axis and ticks for empty subplots
        for i in range(self.n_samples, nrows * ncols):
            axes[i].axis('off')
        
        return fig, axes