import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.colors import Colormap
from typing import Union, List

class HistogramPlotter:
    def __init__(self,
                 X: Union[np.ndarray, List[np.ndarray]],
                 X_labels: Union[None, str, List[str]] = None,
                 bins: Union[int, str, List[float]] = 'auto',
                 density: bool = False,
                 colormap: Union[str, Colormap, None] = None,
                 colors: Union[None, List[str]] = None):
        """
        Initialize the HistogramPlotter object.

        Parameters:
        - X (Union[np.ndarray, List[np.ndarray]]): Single or list of 1D measurements to be turned into histograms.
        - X_labels (Union[None, str, List[str]], optional): Label or list of labels corresponding to each dataset in X.
        - bins (Union[int, str, List[float]], optional): Specification of histogram bins. Default is 'auto'.
        - density (bool, optional): If True, the histogram represents a probability density. Default is False.
        - colormap (Union[str, Colormap, None], optional): Name of the colormap or a colormap object itself.
        - colors (Union[None, List[str]], optional): List of colors corresponding to each dataset in X.
        """
        self.X = X
        self.X_labels = X_labels
        self.bins = bins
        self.density = density
        self.colormap = colormap
        self.colors = colors
        self.color_iter = None

        if (self.colormap is not None) and (self.colors is not None):
            raise ValueError("Both 'colormap' and 'colors' parameters cannot be provided simultaneously. "
                             "Please choose either colormap or colors.")

        if not isinstance(self.X, list):
            self.X = [self.X]

        if isinstance(self.X_labels, str):
            self.X_labels = [self.X_labels]

        if self.X_labels is not None and len(self.X) != len(self.X_labels):
            raise ValueError("Length of X and X_labels should match")

        if self.colors is not None:
            if len(self.colors) != len(self.X):
                raise ValueError("Length of 'colors' should match the number of datasets in X")
            self.color_iter = iter(self.colors)

        if self.colormap is None:
            self.colormap = plt.rcParams['axes.prop_cycle'].by_key()['color']
        elif isinstance(self.colormap, str):
            self.colormap = plt.get_cmap(self.colormap)


    def plot(self) -> Figure:
        """
        Plot histograms for single or multiple datasets for visual comparison.

        Returns:
        - Figure: Matplotlib figure object containing the histogram plot.
        """
        combined_data = np.concatenate(self.X)
        common_bins = np.histogram_bin_edges(combined_data, bins=self.bins)

        fig, ax = plt.subplots()
        for i, data in enumerate(self.X):
            label = None if self.X_labels is None else self.X_labels[i]
            color = None

            if self.colors is not None:
                color = next(self.color_iter)
            else:
                if isinstance(self.colormap, list):
                    color = self.colormap[i]
                else:
                    color = self.colormap(i/len(self.X))

            ax.hist(data, bins=common_bins, alpha=0.5, label=label, color=color, density=self.density)

        ylbl = "Relative Frequency" if self.density else "Frequency"
        ax.set_ylabel(ylbl)

        if self.X_labels is not None:
            ax.legend()

        return fig
