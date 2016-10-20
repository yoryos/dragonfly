from pyqtgraph.Qt import QtCore, QtGui
from Visualiser.RasterVisualiser import RasterVisualiser
from Visualiser.VisualiserComponent import VisualiserComponent


class StdpVisualiser(VisualiserComponent):
    def __init__(self, afferent_data=None, output_data=None, default_history=50):
        VisualiserComponent.__init__(self)

        self.window = QtGui.QGroupBox()
        self.window.setContentsMargins(0, 0, 0, 0)
        self.layout = QtGui.QVBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.window.setLayout(self.layout)
        self.title = QtGui.QLabel("STDP")
        self.title.setAlignment(QtCore.Qt.AlignCenter)
        self.layout.addWidget(self.title)

        self.data_layout = QtGui.QHBoxLayout()
        self.data_layout.setContentsMargins(0, 0, 0, 0)
        self.layout.addLayout(self.data_layout)

        self.control_layout = QtGui.QHBoxLayout()
        self.layout.addLayout(self.control_layout)

        self.info = QtGui.QLabel("")
        self.widgets = []

        if afferent_data is not None:
            self.widgets.append(RasterVisualiser(afferent_data, title="Afferents", parent=self))
        if output_data is not None:
            self.widgets.append(RasterVisualiser(output_data, size=5, y_ticks=True, title="Output", parent=self))

        self.reset_zoom_button = QtGui.QPushButton("Reset Zooms")
        self.reset_zoom_button.clicked.connect(self.reset_zoom)
        self.control_layout.addWidget(self.reset_zoom_button)

        self.history_length_input = QtGui.QSpinBox()
        self.history_length_input.setRange(0, self.steps)
        self.history_length_input.setValue(default_history)
        self.history_length_input.valueChanged.connect(self.reset_zoom)
        self.history_length_input_label = QtGui.QLabel("History Length")

        self.control_layout.addWidget(self.history_length_input_label)
        self.control_layout.addWidget(self.history_length_input)
        self.control_layout.addWidget(self.info)
        self.control_layout.addStretch()

        for w in self.widgets:
            self.data_layout.addWidget(w.window)

    def reset_zoom(self):

        i = self.history_length_input.value()

        for plot in self.widgets:
            plot.reset_zoom(i)

        return i

    @property
    def steps(self):
        if len(self.widgets) > 0:
            if not all(self.widgets[0].steps == w.steps for w in self.widgets):
                print "The two stdp plots have different number of steps"
            return min(self.widgets, key=lambda x: x.steps).steps
        return 0

    def reset_to_start(self):

        for w in self.widgets:
            w.reset_to_start()

    def update(self):

        for plot in self.widgets:
            if not plot.update():
                return False

        return True

    def save(self, name=None, dir=".", index=None, fmt=".png"):
        i = 0
        for w in self.widgets:
            i += w.save(name=None, dir=".", index=None, fmt=".png")
        return i
