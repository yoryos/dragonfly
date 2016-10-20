from CSTMD1.Morphology.NeuronCollection import NeuronCollection
from CSTMD1.Morphology.CSTMD1Visualiser import CSTMD1Visualiser
from Visualiser.DragonflyVisualiser import DragonflyVisualiser
import cPickle as pickle
import sys
from pyqtgraph.Qt import QtCore, QtGui

preload_path = ""
synapses = ""
spikes = ""
nc = pickle.load(open(preload_path))
nc.load_spikes_from_file()
nc.import_synapses_from_file()


app = QtGui.QApplication([])
cstmd1 = CSTMD1Visualiser(nc, True, False, 100, True, True, False)
visualiser = DragonflyVisualiser(1.0,[cstmd1],num_cols=1)

visualiser.show()
visualiser.run()

if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
    app.exec_()