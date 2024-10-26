
from typing import List, Tuple, Union
import inspect
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection, PathCollection, PolyCollection, QuadMesh
from matplotlib.figure import Figure
from matplotlib.artist import Artist
from matplotlib.transforms import BlendedGenericTransform # to check if a line was plotted with axvline or axhline

from ..plotting import SubplotPlotter

class MultiFigurePlotter(SubplotPlotter):
    def __init__(self,
                figures: List[Figure],
                layout: str = 'auto',
                rotate: bool = False,
                suptitle: Union[str, None] = None,
                suptitle_fontsize: Union[int, None] = None,
                suptitle_fontweight: Union[str, None] = None,
                figsize: Union[Tuple[float, float], None] = None,
                facecolor = None):
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
            figsize=figsize,
            facecolor=facecolor)

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
        edgecolor = artist.get_edgecolor()
        if edgecolor.size > 0:
            edgecolor = edgecolor[0]
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


    def _copy_instance(self, obj):
        """Creates a copy of the given object by extracting its parameters.

        Args:
            obj: The object to be copied.

        Returns:
            A new instance of the same class as the input object with the same
            parameters.
        """
        cls = obj.__class__
        init_method = cls.__init__
        signature = inspect.signature(init_method)
        parameters = signature.parameters

        # Initialize containers for args and kwargs
        args = []
        kwargs = {}

        # Extract parameter values from the object
        for param_name, param in parameters.items():
            if param_name == 'self':
                continue
            try:
                value = getattr(obj, param_name)
            except AttributeError:
                # Handle missing attributes gracefully
                continue
            
            # Classify the parameter into args or kwargs
            if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
                args.append(value)
            elif param.kind == inspect.Parameter.KEYWORD_ONLY:
                kwargs[param_name] = value
            elif param.kind == inspect.Parameter.VAR_POSITIONAL:
                args.extend(value)
            elif param.kind == inspect.Parameter.VAR_KEYWORD:
                kwargs.update(value)

        # Create a new instance with the same parameters
        if kwargs:
            new_instance = cls(*args, **kwargs)
        else:
            new_instance = cls(*args)
        
        return new_instance


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
            # Reproduce pie chart
            if isinstance(artist, patches.Wedge):
                # TODO: does it make sense to implement a custom deepcopy (deepcopy does not work)?
#                wedge = self._copy_instance(artist)
#                target_ax.add_patch(wedge)
                wedge = artist
                target_ax.add_patch(
                    patches.Wedge(
                        wedge.center,
                        wedge.r,
                        wedge.theta1,
                        wedge.theta2,
                        width=wedge.width,
                        edgecolor=wedge.get_edgecolor(),
                        facecolor=wedge.get_facecolor(),
                        linewidth=wedge.get_linewidth(),
                        linestyle=wedge.get_linestyle(),
                    )
                )
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
                        'alpha': collection.get_alpha(),
                        'linestyle': collection.get_linestyle(),
                        'linewidth': collection.get_linewidth(),
                        's': collection.get_sizes()[0],
                        'zorder': collection.get_zorder(),
                }
                # Extract the individual color data
                c = collection.get_array()
                if c is None: # Same color for all markers
                    properties['color'] = collection.get_facecolor()[0]
                    properties['edgecolor'] = collection.get_edgecolor()[0]
                else: # Individual colors for all markers
                    properties['edgecolor'] = collection.get_edgecolor()
                    properties['cmap'] = collection.get_cmap()
                    properties['c'] = c
                    
                scatter = target_ax.scatter(x_data, y_data, **properties)
                # Reproduce remaining properties of marker
                scatter.set_paths(collection.get_paths())

            # Reproduce QuadMesh (pcolormesh)
            if isinstance(collection, QuadMesh):
                mesh = collection

                target_mesh = target_ax.pcolormesh(mesh.get_array(),
                                     cmap=mesh.cmap,
                                     norm=mesh.norm,
                                     edgecolors=mesh.get_edgecolors(),
                                     linewidths=mesh.get_linewidths(),
                                     shading=mesh._shading,
                                     alpha=mesh.get_alpha(),
                                     zorder=mesh.get_zorder())
                # Reproduce coordinate grid
                target_mesh._coordinates = mesh._coordinates

                # Reproduce the colorbar 
                if mesh.colorbar:
                    colorbar = mesh.colorbar
                    target_ax.get_figure().colorbar(mesh, ax=target_ax)

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
