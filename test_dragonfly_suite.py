import ESTMD.tests.test_suite as ESTMD
import CSTMD1.tests.test_suite as CSTMD1
import Environment.tests.test_suite as Environment
import Visualiser.tests.test_suite as Visualisers
import Helper.tests.test_suite as Helper
import Integration.tests.test_suite as Dragonfly
import STDP.tests.test_suite as STDP
from pyqtgraph.Qt import QtGui
import sys
import unittest

def test_suite():

    test_suites = [ESTMD.test_suite(),
                   Environment.test_suite(),
                   Helper.test_suite(),
                   CSTMD1.test_suite(),
                   Visualisers.test_suite(),
                   Dragonfly.test_suite(),
                   STDP.test_suite()]

    return  unittest.TestSuite(test_suites)

if __name__ == "__main__":

    global app
    app = QtGui.QApplication.instance()
    if app is None:
        app = QtGui.QApplication(sys.argv)

    dragonfly_tests = unittest.TextTestRunner(verbosity = 2, buffer = True)
    dragonfly_tests.run(test_suite())
    
    app.quit()
