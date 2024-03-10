import numpy as np
from typing import List, Tuple, Union, Optional
from PIL import Image
from matplotlib.figure import Figure

from ..utils import SubplotPlotter

class MultiImagePlotter(SubplotPlotter):
    def __init__(self,
                images,
                titles = [],
                title_prefix = '',
                title_postfix = '',
                layout: str = 'auto',
                rotate: bool = False,
                frame: Optional[bool] = True,
                suptitle: Union[str, None] = None,
                suptitle_fontsize: Union[int, None] = None,
                suptitle_fontweight: Union[str, None] = None,
                figsize: Union[Tuple[float, float], None] = None):
        """
        Draws the given images in the specified layout.

        Args:
            images (List[PIL]): List of images to be plotted.
            titles (list, optional): If given, display list items as titles for each sample. Default is empty.
            title_prefix (str): Prefix string to be added to each title.
            title_postfix (str): Postfix string to be added to each title.
            layout (str): The layout mode for arranging subplots. Options are 'auto', 'grid', and 'vector'.
            rotate (bool): Whether to rotate the layout by 90 degrees.
            frame (bool, optional): If True, display frames around the images. Default is True.
            suptitle (str): Super title for the entire plot.
            suptitle_fontsize (int): Fontsize for the super title.
            suptitle_fontweight (str): Font weight for the super title.
            figsize (tuple): The size of the entire figure in inches (width, height).
        """
        self.images = images
        self.titles = titles
        self.title_prefix = title_prefix
        self.title_postfix = title_postfix
        self.frame: bool = frame

        super().__init__(n_subplots=len(images),
            layout=layout,
            rotate=rotate,
            suptitle=suptitle,
            suptitle_fontsize=suptitle_fontsize,
            suptitle_fontweight=suptitle_fontweight,
            figsize=figsize)


    def plot(self) -> Figure:
        """
        Draws the images.
        """
        for i in range(self.n_subplots):
            ax = self.axes[i]

            img = self.images[i]
            ax.imshow(img)

            # Format and set title string
            if len(self.titles) > 0:
                lbl = self.titles[i]
                if(isinstance(lbl, (float, np.float16, np.float32))):
                    lbl = f"{self.titles[i]:.3f}"
                ax.set_title(f"{self.title_prefix}{lbl}{self.title_postfix}")

            ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

            if not self.frame:
                ax.axis("off")

        super()._finish_plot()

        return self.fig
