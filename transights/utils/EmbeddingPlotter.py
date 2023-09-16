import numpy as np
from pathlib import Path
from PIL import Image

import plotly.express as px
import plotly.graph_objs as go
import plotly.express as px

import ipywidgets as widgets



class EmbeddingPlotter:
    def __init__(self, data, color=None, hover_name=None, file_list=None, width=800, height=600):
        self.data = data
        self.color = color
        self.hover_name = hover_name
        self.file_list = file_list
        self.width = width
        self.height = height


    def load_image(self, filename: str):
        with open(filename, "rb") as f:
                im = f.read()
        return im        


    def plot(self):
        if self.data.shape[1] == 2:
            if self.file_list is None:
                return self._plot_2d()
            else:
                return self._plot_2d_thumb()
        elif self.data.shape[1] == 3:
            if self.file_list is None:
                return self._plot_3d()
            else:
                return self._plot_3d_thumb()
        else:
            raise ValueError("Data dimension should be 2 or 3 for scatterplot.")


    def _update_2d_lookandfeel(self, fig):
        fig.update_layout(width=self.width, height=self.height)
        fig.update_traces(marker=dict(size=5, line=dict(color='black', width=0.5)),
                            selector=dict(mode='markers'))
        fig.update_layout(legend= {'itemsizing': 'constant'})

        return fig


    def _update_3d_lookandfeel(self, fig):
        # update look and feel...
        fig.update_layout(width=self.width, height=self.height)
        fig.update_layout(legend= {'itemsizing': 'constant'})
        fig.update_layout(title_text='Embedding',
                        showlegend=True,
                        legend=dict(orientation="h", yanchor="top", y=0, xanchor="center", x=0.5),
                        scene_camera=dict(up=dict(x=0, y=0, z=1), 
                                                center=dict(x=0, y=0, z=-0.1),
                                                eye=dict(x=1.5, y=-1.4, z=0.5)),
                                                margin=dict(l=0, r=0, b=0, t=0),
                        scene = dict(xaxis=dict(backgroundcolor='white',
                                                color='black',
                                                gridcolor='#f0f0f0',
                                                title_font=dict(size=10),
                                                tickfont=dict(size=10),
                                                ),
                                    yaxis=dict(backgroundcolor='white',
                                                color='black',
                                                gridcolor='#f0f0f0',
                                                title_font=dict(size=10),
                                                tickfont=dict(size=10),
                                                ),
                                    zaxis=dict(backgroundcolor='lightgrey',
                                                color='black', 
                                                gridcolor='#f0f0f0',
                                                title_font=dict(size=10),
                                                tickfont=dict(size=10),
                                                )))
        # ...and marker size
        fig.update_traces(marker=dict(size=3, line=dict(color='black', width=0.1)))

        return fig


    def _plot_2d(self):
        fig = px.scatter(x=self.data[:, 0],
                        y=self.data[:, 1],
                        color=self.color,
                        hover_name=self.hover_name)

        fig = self._update_2d_lookandfeel(fig)

        return fig


    def _plot_3d(self):
        fig = px.scatter_3d(x=self.data[:, 0], y=self.data[:, 1], z=self.data[:, 2], color=self.color, hover_name=self.hover_name)
        fig = self._update_3d_lookandfeel(fig)

        return fig


    def _plot_2d_thumb(self):
        def update(trace, points, state):
            if not points.point_inds:
                return
            
            ind = points.point_inds[0]
            fname = trace['hovertext'][ind]
            img.value = self.load_image(Path(fname))

        fig = self._plot_2d()

        img = widgets.Image(format='png', width=128)
        # TODO: initialize with dummy ; why is this not working?
        #dummy = Image.new('RGBA', size=(32, 32), color=(128, 128, 128))
        #img.value = memoryview(np.array(dummy))
        #img.value = self.load_image(self.file_list[0])
        
        fig = go.FigureWidget(fig)

        # Register callback for all plots (each color is a plot of its own)
        for f in fig.data:
            f.on_hover(update)

        layout = widgets.Layout(
            width='100%',
            height='',
            flex_flow='row',
            display='flex'
        )

        return widgets.Box([fig, widgets.VBox([widgets.Label(), img, widgets.Label()])], layout=layout)


    def _plot_3d_thumb(self):
        def update(trace, points, state):
            if not points.point_inds:
                return
            
            ind = points.point_inds[0]
            fname = trace['hovertext'][ind]
            img.value = self.load_image(Path(fname))

        fig = self._plot_3d()

        img = widgets.Image(format='png', width=128)
        # TODO: initialize with dummy ; why is this not working?
        #dummy = Image.new('RGBA', size=(32, 32), color=(128, 128, 128))
        #img.value = memoryview(np.array(dummy))
        #img.value = self.load_image(self.file_list[0])
        
        fig = go.FigureWidget(fig)

        # Register callback for all plots (each color is a plot of its own)
        for f in fig.data:
            f.on_hover(update)

        layout = widgets.Layout(
            width='100%',
            height='',
            flex_flow='row',
            display='flex'
        )

        return widgets.Box([fig, widgets.VBox([widgets.Label(), img, widgets.Label()])], layout=layout)        