
from typing import List, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


class SubplotPlotter:
    def __init__(self,
                n_subplots: int,
                grid_layout: bool = True,
                rotate: bool = False,
                title: Union[str, None] = None,
                title_fontsize: Union[int, None] = None,
                title_fontweight: Union[str, None] = None,
                figsize: Union[Tuple[float, float], None] = None):
        """
        Constructs a SubplotPlotter object.

        Parameters:
            grid_layout (bool): Whether to display the subplots as a grid.
            rotate (bool): Whether to rotate the layout by 90 degrees.
            title (str): Super title for the entire plot.
            title_fontsize (int): Fontsize for the super title.
            title_fontweight (str): Font weight for the super title.
            figsize (tuple): The size of the entire figure in inches (width, height).
        """
        self.n_subplots: int = n_subplots
        self.grid_layout: bool = grid_layout
        self.rotate: bool = rotate
        self.title: Union[str, None] = title
        self.title_fontsize: Union[int, None] = title_fontsize
        self.title_fontweight: Union[str, None] = title_fontweight
        self.figsize: Union[Tuple[float, float], None] = figsize

        self.fig = None
        self.axes = None

        if self.n_subplots < 2: 
            raise ValueError(f"Invalid value for parameter n_subplots={self.n_subplots}.")

        self.create_subplots()            


#    def plot(self):
#        """
#        Draws the figures in corresponding subplots.
#        """
#        self.fig, self.axes = create_subplots()




    def plot_finish(self) -> None:
        if self.title is not None:
            plt.suptitle(self.title, fontsize=self.title_fontsize, fontweight=self.title_fontweight)
            
        self.fig.tight_layout()


    def create_subplots(self) -> None:
        # Create the layout of subplots
        if self.grid_layout:
            self.create_grid()
        else:
            if self.rotate:
                nrows = 1
                ncols = self.n_subplots
            else:
                nrows = self.n_subplots
                ncols = 1
            self.fig, self.axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=self.figsize)

            # Flatten the axes array for easier indexing
            self.axes = self.axes.flatten()


    def create_grid(self) -> None:
        """
        Creates subplots in a grid layout.

        This function calculates the number of rows and columns based on the desired number of subplots.
        The subplots are then created using the `subplots` function from Matplotlib.

        Returns:
            matplotlib.figure.Figure: The generated figure object.
            numpy.ndarray: Flattened array of axes objects representing the subplots.
        """
        square_len = np.sqrt(self.n_subplots)
        # we sometimes need an additional row depending on the rotation and the number of subplots
        row_appendix = int(bool(np.remainder(self.n_subplots,square_len))*self.rotate)

        nrows = int(square_len) + row_appendix
        ncols = int((self.n_subplots+nrows-1) // nrows)
        
        self.fig, self.axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=self.figsize)

        # Flatten the axes array for easier indexing
        self.axes = self.axes.flatten()

        # Remove axis and ticks for empty subplots
        for i in range(self.n_subplots, nrows * ncols):
            self.axes[i].axis('off')