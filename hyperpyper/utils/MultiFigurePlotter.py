
from typing import List, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.transforms import BlendedGenericTransform


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

        if self.n_subplots < 2: 
            raise ValueError(f"Invalid value for parameter n_subplots={self.n_subplots}.")

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
            # TODO: there might be a better way to check if a line was plotted with axvline or axhline
            if not isinstance(line.get_transform(), BlendedGenericTransform):
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
            # Reproduce lines plotted with axvline or axhline
            else:
                x_data = line.get_xdata()
                y_data = line.get_ydata()

                if all(x==x_data[0] for x in x_data):
                    target_ax.axvline(x=x_data[0],
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
                if all(y==y_data[0] for y in y_data):
                    target_ax.axhline(y=y_data[0],
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
            #print(f"collection detected with type: {type(collection)}")
            if isinstance(collection, plt.collections.LineCollection):
                lc = plt.collections.LineCollection(segments=collection.get_segments(),
                                                    label=collection.get_label(),
                                                    color=collection.get_color(),
                                                    linestyle=collection.get_linestyle(),
                                                    linewidth=collection.get_linewidth(),
                                                    )
                target_ax.add_collection(lc)

        # Reproduce images
        for img in source_ax.images:
            target_ax.imshow(img.get_array(),
                cmap=img.get_cmap(),
                interpolation=img.get_interpolation(),
                extent=img.get_extent())

        # Reproduce text objects
        for text_obj in source_ax.texts:
            x, y = text_obj.get_position()
            target_ax.text(x=x, y=y,
                s=text_obj.get_text(),
                color=text_obj.get_color(),
                fontsize=text_obj.get_fontsize(),
                fontweight=text_obj.get_fontweight(),
                ha=text_obj.get_horizontalalignment(),
                va=text_obj.get_verticalalignment())

        # Reproduce ticks
        tick_params = source_ax.yaxis.get_tick_params(which='major')
        target_ax.yaxis.set_tick_params(which='major',
            left=tick_params['left'],
            right=tick_params['right'],
            labelleft=tick_params['labelleft'],
            labelright=tick_params['labelright'],
            gridOn=tick_params['gridOn'],
        )
        tick_params = source_ax.xaxis.get_tick_params(which='major')
        target_ax.xaxis.set_tick_params(which='major',
            left=tick_params['left'],
            right=tick_params['right'],
            labelleft=tick_params['labelleft'],
            labelright=tick_params['labelright'],
            gridOn=tick_params['gridOn'],
        )
        tick_params = source_ax.yaxis.get_tick_params(which='minor')
        target_ax.yaxis.set_tick_params(which='minor',
            left=tick_params['left'],
            right=tick_params['right'],
            labelleft=tick_params['labelleft'],
            labelright=tick_params['labelright'],
            gridOn=tick_params['gridOn'],
        )
        tick_params = source_ax.xaxis.get_tick_params(which='minor')
        target_ax.xaxis.set_tick_params(which='minor',
            left=tick_params['left'],
            right=tick_params['right'],
            labelleft=tick_params['labelleft'],
            labelright=tick_params['labelright'],
            gridOn=tick_params['gridOn'],
        )
        target_ax.set_xticks(source_ax.get_xticks())
        target_ax.set_yticks(source_ax.get_yticks())

        # Reproduce tick labels
        lbls = source_ax.get_xticklabels()
        if len(lbls) > 0:
            lbl = lbls[0]
            target_ax.set_xticklabels(lbls,
                fontproperties=lbl.get_font_properties(),
                color=lbl.get_color(),
                va=lbl.get_va(),
                ha=lbl.get_ha(),
                rotation=lbl.get_rotation(),
            )
        lbls = source_ax.get_yticklabels()
        if len(lbls) > 0:
            lbl = lbls[0]
            target_ax.set_yticklabels(lbls,
                fontproperties=lbl.get_font_properties(),
                color=lbl.get_color(),
                va=lbl.get_va(),
                ha=lbl.get_ha(),
                rotation=lbl.get_rotation(),
            )

        # Reproduce axis visibility
        target_ax.xaxis.set_visible(source_ax.xaxis.get_visible())
        target_ax.yaxis.set_visible(source_ax.yaxis.get_visible())
        if not source_ax.axison:
            target_ax.set_axis_off()

        # Reproduce axis limits
        target_ax.set_xlim(source_ax.get_xlim())
        target_ax.set_ylim(source_ax.get_ylim())

        # Reproduce aspect ratio
        target_ax.set_aspect(source_ax.get_aspect())

        # Reproduce axis labels
        lbl = source_ax.xaxis.label
        target_ax.set_xlabel(xlabel=source_ax.get_xlabel(),
            fontproperties=lbl.get_font_properties(),
            color=lbl.get_color(),
            va=lbl.get_va(),
            ha=lbl.get_ha(),
            rotation=lbl.get_rotation()
        )        
        lbl = source_ax.yaxis.label
        target_ax.set_ylabel(ylabel=source_ax.get_ylabel(),
            fontproperties=lbl.get_font_properties(),
            color=lbl.get_color(),
            va=lbl.get_va(),
            ha=lbl.get_ha(),
            rotation=lbl.get_rotation()
        )
        
        # Reproduce title
        target_ax.set_title(label=source_ax.get_title(),
            fontproperties=source_ax.title.get_font_properties(),
            color=source_ax.title.get_color(),
            va=source_ax.title.get_va(),
            ha=source_ax.title.get_ha(),
            rotation=source_ax.title.get_rotation()
        )

        # Reproduce legend
        if source_ax.get_legend():
            handles, labels = source_ax.get_legend_handles_labels()
            target_ax.legend(handles, labels)
