
from typing import List, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure


class MultiFigurePlotter:
    def __init__(self, figures: List[Figure], grid_layout: bool = True, rotate: bool = False, 
                 title: Union[str, None] = None, title_fontsize: Union[int, None] = None, 
                 figsize: Union[Tuple[float, float], None] = None):
        """
        Constructs a MultiFigurePlotter object.

        Parameters:
            figures (list): List of matplotlib.figure.Figure objects to be plotted.
            grid_layout (bool): Whether to display the subplots as a grid.
            rotate (bool): Whether to rotate the layout by 90 degrees.
            title (str): Super title for the entire plot.
            title_fontsize (int): Fontsize for the super title.
            figsize (tuple): The size of the entire figure in inches (width, height).
        """
        self.figures: List[Figure] = figures
        self.n_subplots: int = len(figures)
        self.grid_layout: bool = grid_layout
        self.rotate: bool = rotate
        self.title: Union[str, None] = title
        self.title_fontsize: Union[int, None] = title_fontsize
        self.figsize: Union[Tuple[float, float], None] = figsize


    def plot(self) -> Figure:
        """
        Draws the figures in corresponding subplots.
        """
        # Create the layout of subplots
        if self.grid_layout:
            fig, axes = self.create_subplots()
        else:
            if self.rotate:
                fig, axes = plt.subplots(nrows=1, ncols=self.n_subplots)
            else:
                fig, axes = plt.subplots(nrows=self.n_subplots, ncols=1)

        # copy content into subplot axes
        for i,f in enumerate(self.figures):
            self.ax2ax(source_ax=f.get_axes()[0], target_ax=axes[i])

        if self.title is not None:
            plt.suptitle(self.title, fontsize=self.title_fontsize)
            
        fig.tight_layout()

        return fig  


    def create_subplots(self) -> Tuple[Figure, np.ndarray]:
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
        
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=self.figsize)

        # Flatten the axes array for easier indexing
        axes = axes.flatten()

        # Remove axis and ticks for empty subplots
        for i in range(self.n_subplots, nrows * ncols):
            axes[i].axis('off')
        
        return fig, axes


    def ax2ax(self, source_ax, target_ax) -> None:
        """
        Reproduces the contents of one Matplotlib axis onto another axis.

        Parameters:
            source_ax (matplotlib.axes.Axes): The source axis from which the content will be copied.
            target_ax (matplotlib.axes.Axes): The target axis where the content will be reproduced.

        Returns:
            None
        """
        # Reproduce line plots
        for line in source_ax.get_lines():
            target_ax.plot(line.get_xdata(),
                        line.get_ydata(),
                        label=line.get_label(),
                        color=line.get_color(),
                        linestyle=line.get_linestyle(),
                        linewidth=line.get_linewidth(),
                        marker=line.get_marker(),
                        markeredgecolor=line.get_markeredgecolor(),
                        markeredgewidth=line.get_markeredgewidth(),
                        markerfacecolor=line.get_markerfacecolor(),
                        markersize=line.get_markersize(),
                        )

        # Reproduce rectangles (histogram bars)
        for artist in source_ax.__dict__['_children']:
            if isinstance(artist, patches.Rectangle):
                rect = artist
                # Retrieve properties of each rectangle and reproduce it on the target axis
                target_ax.add_patch(patches.Rectangle((rect.get_x(), rect.get_y()),
                                                    rect.get_width(),
                                                    rect.get_height(),
                                                    edgecolor=rect.get_edgecolor(),
                                                    facecolor=rect.get_facecolor(),
                                                    linewidth=rect.get_linewidth(),
                                                    linestyle=rect.get_linestyle()
                                                    ))

        # Reproduce collections (e.g., LineCollection)
        for collection in source_ax.collections:
            if isinstance(collection, plt.collections.LineCollection):
                lc = plt.collections.LineCollection(segments=collection.get_segments(),
                                                    label=collection.get_label(),
                                                    color=collection.get_color(),
                                                    linestyle=collection.get_linestyle(),
                                                    linewidth=collection.get_linewidth(),
                                                )
                target_ax.add_collection(lc)

        # Reproduce axis limits and aspect ratio
        target_ax.set_xlim(source_ax.get_xlim())
        target_ax.set_ylim(source_ax.get_ylim())
        target_ax.set_aspect(source_ax.get_aspect())

        # Reproduce axis labels
        target_ax.set_xlabel(source_ax.get_xlabel())
        target_ax.set_ylabel(source_ax.get_ylabel())
        
        # Reproduce title
        target_ax.set_title(source_ax.get_title())

        # Reproduce legend
        handles, labels = source_ax.get_legend_handles_labels()
        target_ax.legend(handles, labels)
