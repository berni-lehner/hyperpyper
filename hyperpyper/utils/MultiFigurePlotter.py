
from typing import List, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection, PathCollection, PolyCollection
from matplotlib.figure import Figure
from matplotlib.artist import Artist
from matplotlib.transforms import BlendedGenericTransform # to check if a line was plotted with axvline or axhline

from ..utils import SubplotPlotter

class MultiFigurePlotter(SubplotPlotter):
    def __init__(self,
                figures: List[Figure],
                layout: str = 'auto',
                rotate: bool = False,
                suptitle: Union[str, None] = None,
                suptitle_fontsize: Union[int, None] = None,
                suptitle_fontweight: Union[str, None] = None,
                figsize: Union[Tuple[float, float], None] = None):
        """
        Constructs a MultiFigurePlotter object.

        Parameters:
            figures (list): List of matplotlib.figure.Figure objects to be plotted.
            layout (str): The layout mode for arranging subplots. Options are 'auto', 'grid', and 'vector'.
            rotate (bool): Whether to rotate the layout by 90 degrees.
            suptitle (str): Super title for the entire plot.
            suptitle_fontsize (int): Fontsize for the super title.
            suptitle_fontweight (str): Font weight for the super title.
            figsize (tuple): The size of the entire figure in inches (width, height).
        """
        self.figures: List[Figure] = figures

        super().__init__(n_subplots=len(figures),
            layout=layout,
            rotate=rotate,
            suptitle=suptitle,
            suptitle_fontsize=suptitle_fontsize,
            suptitle_fontweight=suptitle_fontweight,
            figsize=figsize)

    def plot(self) -> Figure:
        """
        Draws the figures in corresponding subplots.
        """
        # copy content into subplot axes
        for i,f in enumerate(self.figures):
            self.ax2ax(source_ax=f.get_axes()[0], target_ax=self.axes[i])

        super()._finish_plot()

        return self.fig  


    def _recreate_PolyCollection(self, artist: Artist) -> PolyCollection:
        """
        Reproduce a PolyCollection artist with the same properties.

        Parameters:
            artist (matplotlib.artist.Artist): The PolyCollection artist to be reproduced.

        Returns:
            matplotlib.collections.PolyCollection: The reproduced PolyCollection with the same properties.

        Raises:
            ValueError: If the input artist is not a PolyCollection.
        """
        if not isinstance(artist, PolyCollection):
            raise ValueError(f"Input artist is not a PolyCollection, but {type(artist)}.")

        vertices = artist.get_paths()[0].vertices
        facecolor = artist.get_facecolor()[0]
        edgecolor = artist.get_edgecolor()[0]
        alpha = artist.get_alpha()
        linewidth = artist.get_linewidth()
        linestyle = artist.get_linestyle()
        # For some reason, linestyle (the dashes) gets modifed by linewidth, and we need to compensate
        if linestyle[0][1] is not None:
            linestyle = (linestyle[0][0], linestyle[0][1]/linewidth)

        # Create a PolyCollection with the same properties
        poly_collection = PolyCollection([vertices],
                facecolors=facecolor,
                edgecolors=edgecolor,
                alpha=alpha,
                linestyle=linestyle,
                linewidth=linewidth)

        return poly_collection


    def ax2ax(self, source_ax, target_ax) -> None:
        """
        Reproduces the contents of one Matplotlib axis onto another axis.

        Parameters:
            source_ax (matplotlib.axes.Axes): The source axis from which the content will be copied.
            target_ax (matplotlib.axes.Axes): The target axis where the content will be reproduced.

        Returns:
            None
        """
        # Reproduce background color
        target_ax.set_facecolor(source_ax.get_facecolor())

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
            # Reproduce PolyCollection (plotted with fill_between())
            if isinstance(artist, PolyCollection):
                for artist in source_ax.get_children():
                    if isinstance(artist, PolyCollection):
                        poly_collection = self._recreate_PolyCollection(artist)
                        target_ax.add_collection(poly_collection)

        # Reproduce collections (e.g., LineCollection)
        for collection in source_ax.collections:
            if isinstance(artist, PolyCollection):
                print("collections.PolyCollection")

            #print(f"collection detected with type: {type(collection)}")
            # Reproduce lines
            if isinstance(collection, LineCollection):
                lc = LineCollection(segments=collection.get_segments(),
                    label=collection.get_label(),
                    color=collection.get_color(),
                    linestyle=collection.get_linestyle(),
                    linewidth=collection.get_linewidth(),
                    )
                target_ax.add_collection(lc)
            # Reproduce scatterplot
            if isinstance(collection, PathCollection):
                x_data = [d[0] for d in collection.get_offsets().data]
                y_data = [d[1] for d in collection.get_offsets().data]

                # Reproduce properties of the original scatterplot
                properties = {
                        'color': collection.get_facecolor()[0],
                        'edgecolor': collection.get_edgecolor()[0],
                        'alpha': collection.get_alpha(),
                        'linestyle': collection.get_linestyle(),
                        'linewidth': collection.get_linewidth(),
                        's': collection.get_sizes()[0],                        
                    }
                scatter = target_ax.scatter(x_data, y_data, **properties)
                # Reproduce remaining properties of marker
                scatter.set_paths(collection.get_paths())

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

            new_handles = []
            for handle in handles:
                if isinstance(handle, PolyCollection):
                    # Create a new PolyCollection with the same properties
                    new_handle = self._recreate_PolyCollection(handle)
                    new_handles.append(new_handle)
                else:
                    new_handles.append(handle)
            target_ax.legend(new_handles, labels)

            # Iterate over text elements in the legend and set font properties in the target legend
            for source_text, target_text in zip(source_ax.get_legend().get_texts(), target_ax.get_legend().get_texts()):
                font_properties = source_text.get_font_properties()
                target_text.set_fontproperties(font_properties)
