
from typing import List, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


class SubplotPlotter:
    def __init__(self,
                n_subplots: int,
                layout: str = 'auto',
                rotate: bool = False,
                suptitle: Union[str, None] = None,
                suptitle_fontsize: Union[int, None] = None,
                suptitle_fontweight: Union[str, None] = None,
                figsize: Union[Tuple[float, float], None] = None):
        """
        Constructs a SubplotPlotter object.

        Parameters:
            n_subplots (int): The number of subplots.
            layout (str): The layout mode for arranging subplots. Options are 'auto', 'grid', and 'vector'.
            rotate (bool): Whether to rotate the layout by 90 degrees.
            suptitle (str): Super title for the entire plot.
            suptitle_fontsize (int): Fontsize for the super title.
            suptitle_fontweight (str): Font weight for the super title.
            figsize (tuple): The size of the entire figure in inches (width, height).
        """
        self.n_subplots: int = n_subplots
        self.layout = layout
        self.rotate: bool = rotate
        self.suptitle: Union[str, None] = suptitle
        self.suptitle_fontsize: Union[int, None] = suptitle_fontsize
        self.suptitle_fontweight: Union[str, None] = suptitle_fontweight
        self.figsize: Union[Tuple[float, float], None] = figsize

        self.fig = None
        self.axes = None

        self.n_rows = None
        self.n_cols = None

        if self.n_subplots < 2: 
            raise ValueError(f"Invalid value for parameter n_subplots={self.n_subplots}.")

        self._create_subplots()            


    def _finish_plot(self) -> None:
        """
        Finishes plot setup by setting the super title and adjusting layout.
        """
        if self.suptitle is not None:
            plt.suptitle(self.suptitle, fontsize=self.suptitle_fontsize, fontweight=self.suptitle_fontweight)
            
        self.fig.tight_layout()


    def _create_subplots(self) -> None:
        """
        Creates subplot axes correponding to the given parameters.
        """
        if self.layout == 'grid':
            self._create_grid()
        elif self.layout == 'vector':
            self._create_vector()
        elif self.layout == 'auto':
            self._create_autogrid()
        else:
            raise ValueError(f"Invalid value for parameter layout={self.layout}.")

        if self.rotate:
            tmp = self.n_rows
            self.n_rows = self.n_cols
            self.n_cols = tmp

        self.fig, self.axes = plt.subplots(nrows=self.n_rows,
            ncols=self.n_cols,
            figsize=self.figsize)

        self._finish_subplots()


    def _finish_subplots(self) -> None:
        """
        Performs post-processing on subplots.
        """
        # Flatten the axes array for easier indexing
        self.axes = self.axes.flatten()

        # Remove axis and ticks for empty subplots
        for i in range(self.n_subplots, self.n_rows * self.n_cols):
            self.axes[i].axis('off')


    def _create_vector(self) -> None:
        """
        Creates subplot parameters aligned with a single axis.
        """
        self.n_rows = 1
        self.n_cols = self.n_subplots


    def _create_grid(self) -> None:
        """
        Creates subplot parameters strictly in a grid layout.
        """
        self.n_rows = self._find_smallest_square(self.n_subplots)
        self.n_cols = self.n_rows


    def _create_autogrid(self) -> None:
        """
        Creates subplot parameters in a grid layout while avoiding empty subplots.
        For example, 8 subplots will result in a 4 by 2 subplot matrix instead of a 3 by 3 with an empty subplot.
        """
        square_len = np.sqrt(self.n_subplots)

        self.n_rows = int(square_len)
        self.n_cols = int((self.n_subplots+self.n_rows-1) // self.n_rows)


    def _find_smallest_square(self, n):
        """
        Find the smallest square length to fit n elements.
        """
        smallest_square = int(np.ceil(np.sqrt(n))) ** 2
        return int(np.sqrt(smallest_square))
