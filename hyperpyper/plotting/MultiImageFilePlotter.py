from typing import List, Tuple, Union, Optional
from pathlib import Path
from PIL import Image


from ..plotting import MultiImagePlotter

class MultiImageFilePlotter(MultiImagePlotter):
    def __init__(self,
                files,
                mode: str = 'RGB',
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
        Draws the given image files in the specified layout.

        Args:
            
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
        self.files = files
        self.mode = mode
        self.images = []
        self._load_images()


        super().__init__(
            images=self.images,
            titles=titles,
            title_prefix=title_prefix,
            title_postfix=title_postfix,
            layout=layout,
            rotate=rotate,
            frame=frame,
            suptitle=suptitle,
            suptitle_fontsize=suptitle_fontsize,
            suptitle_fontweight=suptitle_fontweight,
            figsize=figsize)


    def _load_images(self) -> None:
        """
        Load images from a list of filenames.
        """
        for file in self.files:
            file_path = Path(file) if isinstance(file, str) else file
            image = Image.open(file_path).convert(self.mode)
            self.images.append(image)