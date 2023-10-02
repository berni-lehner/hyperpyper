import numpy as np
from pathlib import Path
from PIL import Image

import plotly.express as px
import plotly.graph_objs as go
import plotly.express as px

import ipywidgets as widgets

from scipy.stats import gaussian_kde


class EmbeddingPlotter:
    def __init__(self, data, color=None, hover_name=None, file_list=None, width=800, height=600, fix_axis=True, axis_margin=1.1):
        self.data = data
        self.color = color
        self.file_list = file_list

        if file_list is not None:
            # we use hover_name for files, which has to be a string representation, and not a pathlib.Path
            self.hover_name = [str(f) for f in file_list]
            if hover_name is not None:
                raise ValueError("file_list and hover_name not supported at the same time.")
        else:
            self.hover_name = hover_name



        self.width = width
        self.height = height
        self.fix_axis = fix_axis
        self.axis_margin = axis_margin

        self._fig = None
        self._min_vals = []
        self._max_vals = []

        if fix_axis:
            self._min_vals = np.min(data, axis=0)
            self._max_vals = np.max(data, axis=0)

        # Color according to the density
        if type(self.color) == str:
            if self.color == 'kde':
                self.color = self._estimate_density()
            else:
                raise ValueError("Unknown color mode. Supported modes currently are: 'kde'.")


    def _estimate_density(self):
        # estimate kde
        kde = gaussian_kde(self.data.T)

        # evaluate on data
        density = kde.evaluate(self.data.T)
        
        return density

    #todo:
    #def get_scatterplot_fig():

        

    def load_image(self, filename: str):
        with open(filename, "rb") as f:
                im = f.read()
        return im        


    def plot(self):
        if self.data.shape[1] == 2:
            if self.file_list is None:
                return self._plot_2d()
            else:
                return self._plot_thumb(self._plot_2d)
        elif self.data.shape[1] == 3:
            if self.file_list is None:
                return self._plot_3d()
            else:
                return self._plot_thumb(self._plot_3d)
        else:
            raise ValueError("Data dimension should be 2 or 3 for scatterplot.")


    def _update_2d_lookandfeel(self):
        if self._fig is None:
            raise ValueError("Figure needs to be created beforehand.")

        self._fig.update_layout(width=self.width, height=self.height)
        self._fig.update_traces(marker=dict(size=5, line=dict(color='black', width=0.5)),
                            selector=dict(mode='markers'))
        self._fig.update_layout(legend= {'itemsizing': 'constant'})

        if self.fix_axis:
            self._fix_axis_2d()

        return self._fig


    def _update_3d_lookandfeel(self):
        if self._fig is None:
            raise ValueError("Figure needs to be created beforehand.")

        # update look and feel...
        self._fig.update_layout(width=self.width, height=self.height)
        self._fig.update_layout(legend= {'itemsizing': 'constant'})
        self._fig.update_layout(title_text='Embedding',
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
        self._fig.update_traces(marker=dict(size=3, line=dict(color='black', width=0.1)))

        if self.fix_axis:
            self._fix_axis_3d()

        return self._fig


    def _fix_axis_2d(self):
        if self._fig is None:
            raise ValueError("Figure needs to be created before axis can be fixed.")

        self._fig.update_xaxes(range=[self._min_vals[0]*self.axis_margin, self._max_vals[0]*self.axis_margin])
        self._fig.update_yaxes(range=[self._min_vals[1]*self.axis_margin, self._max_vals[1]*self.axis_margin])


    def _fix_axis_3d(self):
        self._fix_axis_2d()
        if self.data.shape[1] < 3:
            raise ValueError("Data dimension needs to be 3 for fixing z-axis .")

        self._fig.update_scenes(zaxis=dict(range=[self._min_vals[2]*self.axis_margin, self._max_vals[2]*self.axis_margin]))


    def _plot_2d(self):
        self._fig = px.scatter(x=self.data[:, 0],
                         y=self.data[:, 1],
                         color=self.color,
                         hover_name=self.hover_name)
        self._update_2d_lookandfeel()

        return self._fig


    def _plot_3d(self):
        self._fig = px.scatter_3d(x=self.data[:, 0],
                            y=self.data[:, 1],
                            z=self.data[:, 2],
                            color=self.color,
                            hover_name=self.hover_name)
        self._update_3d_lookandfeel()

        return self._fig


    def _plot_thumb(self, f):
        def update(trace, points, state):
            if not points.point_inds:
                return
            
            ind = points.point_inds[0]
            fname = trace['hovertext'][ind]
            img.value = self.load_image(Path(fname))

        # create basic plot
        fig = f()

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