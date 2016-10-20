import numpy as np
import pyqtgraph as pg
from Visualiser.VisualiserComponent import VisualiserComponent


class RasterVisualiser(VisualiserComponent):

    def __init__(self, data, depth=50, size=1, y_ticks=False, title=None, parent=None):
        VisualiserComponent.__init__(self)
        self.index = 0
        self.history_depth = depth
        self.time_of_spike, self.spike_y, self.n_steps, self.n = self.convert_to_raster(data)

        self.window = pg.GraphicsWindow()
        self.window.setBackground((255, 255, 255, 255))

        self.plot = self.window.addPlot()

        self.plot.setDownsampling(mode='peak')

        self.plot.setTitle(title)

        self.raster_plot = self.plot.plot(pen=None,
                                          symbol='o',
                                          symbolPen=pg.mkPen((0, 0, 0, 255)),
                                          symbolSize=size,
                                          symbolBrush=pg.mkBrush(0, 0, 0, 255))

        self.plot.getAxis("bottom").setTickSpacing(major=int(self.history_depth / 5), minor=1)

        if y_ticks:
            # self.plot.getAxis("left").setTickSpacing(major = 1)
            self.plot.getAxis("left").showLabel(False)
            self.plot.getAxis("left").setTicks([[(j, '') for j in range(self.n + 1)], []]
                                               )
            self.plot.showGrid(True, True, 0.2)
        else:
            self.plot.hideAxis('left')
            self.plot.showGrid(True, False, 0.2)

        self.plot.setLabel("bottom", "Time /steps")

        to_get = self.to_get()

        # self.raster_plot.setData(x=self.time_of_spike[to_get],
        #                          y=self.spike_y[to_get])

        self.raster_plot.setPos(-self.index, 0)
        self.plot.setXRange(-self.history_depth, 0, padding=0)
        self.plot.setYRange(0, self.n, padding=0.1)
        self.plot.setLimits(xMax=0, yMin=0.9, yMax=self.n + 0.1, xMin=-self.history_depth)
        # self.plot.setMouseEnabled(False, False)

        if parent is not None:
            existing = parent.info.text()
            parent.info.setText(existing + "[" + title + ": " + str(self.n) + " spike traces] ")

        self.plot.addItem(self.raster_plot)

    def reset_zoom(self, i):
        self.history_depth = i
        self.plot.setXRange(-self.history_depth, 0, padding=0)
        self.plot.setYRange(0, self.n, padding=0.1)

    def to_get(self):
        return np.logical_and(self.time_of_spike <= self.index, self.time_of_spike > (self.index - self.history_depth))
        # self.time_of_spike <= self.index

    def update(self):
        self.index += 1
        if self.index > self.n_steps:
            return False

        to_get = self.to_get()
        # if not np.any(to_get):
        #     self.plot.setClipToView(False)
        # else:
        #     self.plot.setClipToView(True)
        x = self.time_of_spike[to_get]
        y = self.spike_y[to_get]
        self.raster_plot.setData(x=x, y=y)
        self.raster_plot.setPos(-self.index, 0)
        self.plot.setLimits(xMin=min(-self.history_depth, -(self.index + 1)))  # ,xMax=0, yMin = 0, yMax = self.n + 0.1)


        return True

    @staticmethod
    def convert_to_raster(data):
        n = data.shape[1]
        steps = data.shape[0]
        time_steps = np.array(range(0, data.shape[0]))
        y_offsets = np.arange(1, data.shape[1] + 1)
        data_y_offset = data * y_offsets

        time_steps = np.array([time_steps, ] * n).transpose()[data_y_offset > 0]
        data_y_offset = data_y_offset[data_y_offset > 0]

        return time_steps, data_y_offset, steps, n

    def reset_to_start(self):
        self.index = 0

    @property
    def steps(self):
        return self.n_steps