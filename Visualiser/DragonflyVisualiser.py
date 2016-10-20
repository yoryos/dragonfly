import os
from progressbar import Percentage, Bar, ETA, ProgressBar
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
class DragonflyVisualiser(QtGui.QWidget):
    """ Visualiser for neural module data"""

    def __init__(self, dt, modules, size =(100,600), num_cols = 2):

        QtGui.QWidget.__init__(self)
        self.modules = []
        self.dt = dt
        self.stop = False
        self.modules = modules

        self.resize(size[0],size[1])
        self.setWindowTitle("Dragonfly Visualiser")
        self.layout_manager = QtGui.QGridLayout()
        self.setLayout(self.layout_manager)

        self.layout_manager.setContentsMargins(0, 0, 0, 0)
        num_mods = len(modules)
        self.num_cols = num_cols
        self.num_rows = np.ceil(float(num_mods) / num_cols)

        rows = np.repeat(np.arange(self.num_rows), self.num_cols)
        cols = np.tile(np.arange(num_cols), self.num_rows)
        print "Creating a visualiser with " + str(self.num_rows) + " rows and " + str(num_cols) + " cols "

        for module,i,j in zip(*(modules,rows,cols)):
            self.layout_manager.addWidget(module.window,i,j)

        for row in xrange(self.layout_manager.rowCount()):
            self.layout_manager.setRowStretch(row, 1)
        for col in xrange(self.layout_manager.columnCount()):
            self.layout_manager.setColumnStretch(col, 1)

        self.__make_buttons()

    def update(self):

        if not self.running:
            self.status.setText("Status: Running")

        for module in self.modules:
            if not module.update():
                if self.index == self.steps:
                    self.status.setText("Finished")
                # self.pbar.finish()
                self.timer.stop()
                self.stop = True
                self.running = False
                self.status.setText("STOP: Error updating")
                return -1

        self.index = (self.index + 1) % self.steps
        self.pbar.update(self.index)
        self.vis_pbar.setValue(self.index)
        self.time_stamp.setText("Time " + str(self.index * self.dt) + "ms")

        if self.index == 0:
            self.reset()

        return self.index

    def __make_buttons(self):

        self.control_buttons = QtGui.QHBoxLayout()
        self.pause_button = QtGui.QPushButton("Pause", self)
        self.pause_button.clicked.connect(self.pause)
        self.control_buttons.addWidget(self.pause_button)
        self.start_button = QtGui.QPushButton("Start", self)
        self.start_button.clicked.connect(self.start)
        self.control_buttons.addWidget(self.start_button)
        self.save_button = QtGui.QPushButton("Save", self)
        self.save_button.clicked.connect(self.save)
        self.control_buttons.addWidget(self.save_button)
        self.reset_button = QtGui.QPushButton("Reset", self)
        self.reset_button.clicked.connect(self.reset)
        self.control_buttons.addWidget(self.reset_button)
        self.layout_manager.addLayout(self.control_buttons, 2, 0)
        self.maximise_button = QtGui.QPushButton("Full Screen", self)
        self.maximise_button.clicked.connect(self.toggle_screen)
        self.control_buttons.addWidget(self.maximise_button)

    def run(self, time_step=5):

        self.index = 0
        self.running = False
        self.time_step = time_step

        module_steps = [m.steps for m in self.modules]
        self.steps = int(min(module_steps)) - 1
        print "Module Steps ", module_steps, "setting to minimum ", self.steps

        self.vis_pbar = QtGui.QProgressBar()
        self.vis_pbar.setMinimum(0)
        self.vis_pbar.setMaximum(self.steps - 1)
        self.vis_pbar.setTextVisible(True)

        self.text_outputs = QtGui.QHBoxLayout()
        self.time_stamp = QtGui.QLabel("Time _ms")
        self.status = QtGui.QLabel("Status:")
        self.text_outputs.addWidget(self.time_stamp)
        self.text_outputs.addWidget(self.status)

        self.layout_manager.addWidget(self.vis_pbar)
        self.layout_manager.addLayout(self.text_outputs, 3, 0, 1, 2)

        widgets = ['Dragonfly Visualiser: ', Percentage(), ' ',
                   Bar(marker='#', left='[', right=']'),
                   ' ', ETA()]

        self.pbar = ProgressBar(widgets=widgets, maxval=(self.steps - 1), redirect_stdout=True).start()
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)

    def toggle_screen(self):

        if self.isFullScreen():
            self.maximise_button.setText("Full Screen")
            self.showNormal()
        else:
            self.maximise_button.setText("Exit")
            self.showFullScreen()

    def save(self, dir = None):

        if self.running:
            self.pause()

        directory = "VisualiserDump"
        if dir is not None and isinstance(dir,str):
            directory = dir

        if not os.path.isdir(directory):
            print "Making directory ", directory
            os.mkdir(directory)

        i = 0
        for module in self.modules:
            i += module.save(index = self.index, dir = directory)

        screenshot = QtGui.QPixmap.grabWindow(self.winId())
        screenshot.save(os.path.join(directory, "screenshot"), 'jpg')
        i += 1
        print "Saved to " + os.path.join(directory, "screenshot") + ".jpg"

        self.status.setText("Status: Saved " + str(i) + " screenshots at index " + str(self.index))

        if self.running:
            self.restart()

        return i

    def keyPressEvent(self, QKeyEvent):

        QtGui.QWidget.keyPressEvent(self, QKeyEvent)

        if QKeyEvent.key() == QtCore.Qt.Key_Escape:
            self.close()

        if QKeyEvent.key() == QtCore.Qt.Key_F:

            if self.isFullScreen():
                self.showNormal()
            else:
                self.showFullScreen()

    def pause(self):

        if self.running:
            self.status.setText("Status: Paused at " + str(self.index))
            self.timer.stop()
            self.running = False
            return True
        return False

    def reset(self):

        self.pause()

        self.index = 0
        self.stop = False
        for module in self.modules:
            module.reset_to_start()

        self.pbar.update(0)
        self.vis_pbar.setValue(0)
        self.status.setText("Status: Reset to index " + str(self.index))

    def start(self):

        if not self.running and not self.stop:
            self.timer.start(self.time_step)
            self.running = True
            self.status.setText("Status: Started at " + str(self.index))
            return True
        return False
