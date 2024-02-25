import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import List, Tuple, Union, Optional

class DatasetRandSamplePlotter:
    def __init__(self, dataset: Union[Dataset, List[Dataset]],
                n_samples: int = 16,
                grid_layout: bool = True, rotate: bool = False, title_label: Optional[bool] = False,
                frame: Optional[bool] = True,
                title: Union[str, None] = None, title_fontsize: Union[int, None] = None, 
                figsize: Union[Tuple[float, float], None] = None):
        """
        Constructs a MultiFigurePlotter object.

        Parameters:
            dataset (Union[torch.utils.data.Dataset, List[torch.utils.data.Dataset]]): Single or list of PyTorch datasets.
            n_samples (int): Number of samples to plot.
            grid_layout (bool): Whether to display the subplots as a grid.
            rotate (bool): Whether to rotate the layout by 90 degrees.
            title_label (bool, optional): If True, display labels as titles for each sample. Default is False.
            frame (bool, optional): If True, display frames around the images. Default is True.
            title (str): Super title for the entire plot.
            title_fontsize (int): Fontsize for the super title.
            figsize (tuple): The size of the entire figure in inches (width, height).
        """
        self.dataset: Dataset = dataset
        self.n_samples: int = n_samples
        self.grid_layout: bool = grid_layout
        self.rotate: bool = rotate
        self.title_label: bool = title_label
        self.frame: bool = frame
        self.title: Union[str, None] = title
        self.title_fontsize: Union[int, None] = title_fontsize
        self.figsize: Union[Tuple[float, float], None] = figsize


    def plot(self) -> Figure:
        """
        Draws the samples from the Dataset(s).
        """
        if isinstance(self.dataset, list):
            fig = self.plot_random_samples_multi()
        else:
            fig = self.plot_random_samples_single()

        if self.title is not None:
            plt.suptitle(self.title, fontsize=self.title_fontsize)

        fig.tight_layout()

        return fig


    def plot_random_samples_single(self) -> Figure:
        """
        Draws the samples from the Dataset.
        """
        # Randomly sample indices
        n_images = len(self.dataset)
        sampled_indices = np.random.permutation(n_images)[:self.n_samples]

        # Extract images and labels
        sampled_images = [self.dataset[i][0] for i in sampled_indices]
        if self.title_label:
            sampled_labels = [self.dataset[i][1] for i in sampled_indices]

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
            ax.imshow(np.transpose(sampled_images[i], (1, 2, 0)))

            if self.title_label:
                ax.set_title(f"{sampled_labels[i]}")

            ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

            if not self.frame:
                ax.axis("off")

        return fig

    def plot_random_samples_multi(self) -> Figure:
        """
        Plot random samples from multiple PyTorch datasets.

        Returns:
            Figure: The generated matplotlib figure containing the plotted samples.
        """
        n_datasets = len(self.dataset)

        # Create the layout of subplots
        if self.rotate:
            fig, axes = plt.subplots(nrows=self.n_samples, ncols=n_datasets, figsize=self.figsize)
        else:
            fig, axes = plt.subplots(nrows=n_datasets, ncols=self.n_samples, figsize=self.figsize)


        if n_datasets < 2:
            axes = np.expand_dims(axes, axis=0)  # Add an extra dimension for consistency

        for dataset_index, ds in enumerate(self.dataset):
            # Randomly sample indices
            n_images = len(ds)
            sampled_indices = np.random.permutation(n_images)[:self.n_samples]

            # Extract images and labels
            sampled_images = [ds[i][0] for i in sampled_indices]
            if self.title_label:
                sampled_labels = [ds[i][1] for i in sampled_indices]


            for i in range(self.n_samples):
                if self.rotate:
                    ax = axes[i, dataset_index]
                else:
                    ax = axes[dataset_index, i]
                ax.imshow(np.transpose(sampled_images[i], (1, 2, 0)))

                if self.title_label:
                    ax.set_title(f"{sampled_labels[i]}")

                ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

                if not self.frame:
                    ax.axis("off")

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