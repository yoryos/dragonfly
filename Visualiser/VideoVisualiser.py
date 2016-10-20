from pyqtgraph.Qt import QtCore, QtGui
from Helper.Video_Tools import VideoToFrameConverter
import pyqtgraph as pg
import os
from Visualiser.VisualiserComponent import VisualiserComponent


class VideoVisualiser(VisualiserComponent):
    def __init__(self, name, vid_path, color=False, preload=10):
        VisualiserComponent.__init__(self)
        self.window = QtGui.QGroupBox()
        self.window.setContentsMargins(0, 0, 0, 0)
        self.layout = QtGui.QVBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.window.setLayout(self.layout)
        self.data_window = pg.GraphicsLayoutWidget()
        self.color = color
        self.view = self.data_window.addViewBox(invertY=True)
        self.img = pg.ImageItem(border='w')
        self.view.addItem(self.img)

        self.title = QtGui.QLabel(name)
        self.title.setAlignment(QtCore.Qt.AlignCenter)
        self.layout.addWidget(self.title)
        self.layout.addWidget(self.data_window)

        self.vp = VideoToFrameConverter()
        self.vp.open_video(vid_path)
        self.n_steps = self.vp.get_frame_count()
        self.preload = preload
        s, self.frames = self.vp.get_frames(color=color, total_frames=preload)
        frame = self.frames.pop(0)
        if color:
            frame = frame.transpose((1, 0, 2))
        else:
            frame = frame.transpose()

        self.view.setRange(QtCore.QRectF(0, 0, frame.shape[0], frame.shape[1]))
        self.view.setAspectLocked(True, ratio=float(frame.shape[0]) / frame.shape[1])
        self.view.setMouseEnabled(False, False)

        self.img.setImage(frame)
        print "Set image successful for " + name + " visualiser"

    def update(self):

        if len(self.frames) == 0:
            print "Buffer empty for " + self.title.text() + " rebuilding to " + str(self.preload)
            status, self.frames = self.vp.get_frames(color=self.color, total_frames=self.preload)

            if not status:
                return False

        frame = self.frames.pop(0)

        if self.color:
            frame = frame.transpose((1, 0, 2))
        else:
            frame = frame.transpose()

        self.img.setImage(frame)

        return True

    def save(self, name=None, dir=".", index=None, fmt=".png"):

        if name is None:
            name = str(self.title.text())
        if index is not None:
            name += ("_" + str(index))

        name += fmt
        path = os.path.join(dir, name)

        self.img.save(path)

        print "Saved to " + path
        return 1

    def reset_to_start(self):

        self.vp.reset_to_start()

    @property
    def steps(self):
        return self.n_steps