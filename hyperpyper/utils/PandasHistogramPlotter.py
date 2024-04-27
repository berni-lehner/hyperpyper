import numpy as np
import pandas as pd
from typing import List, Tuple, Union

from matplotlib.colors import Colormap

from ..utils import HistogramPlotter

class PandasHistogramPlotter(HistogramPlotter):
    def __init__(self,
        df: pd.DataFrame,
        columns: List[str],
        labels: Union[None, str, List[str]] = 'auto',
        bins: Union[int, str, List[float]] = 'auto',
        density: bool = False,
        colormap: Union[str, Colormap, None] = None,
        colors: Union[None, List[str]] = None,
        figsize: Union[Tuple[float, float], None] = None):
        """
        Initialize the PandasHistogramPlotter object.

        Parameters:
            df (pd.DataFrame): Pandas DataFrame.
            columns (List[str]): The columns of the DataFrame that will be analysed.
            labels (Union[None, str, List[str]], optional): Label or list of labels for the legend. Default is 'auto' and uses the columns.
            bins (Union[int, str, List[float]], optional): Specification of histogram bins. Default is 'auto'.
            density (bool, optional): If True, the histogram represents a probability density. Default is False.
            colormap (Union[str, Colormap, None], optional): Name of the colormap or a colormap object itself.
            colors (Union[None, List[str]], optional): List of colors corresponding to each dataset in X.
            figsize (tuple): The size of the entire figure in inches (width, height).
        """
        self.columns = columns
        self.df = df

        # Set legend entries to columns
        if labels == 'auto':
            labels = self.columns

        # Extract data into a list of arrays
        X = []
        for col in self.columns:
            X.append(self.df[col].values)
            
        super().__init__(
            X=X,
            X_labels=labels,
            bins=bins,
            density=density,
            colormap=colormap,
            colors=colors,
            figsize=figsize)