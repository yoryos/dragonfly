import sys
import unittest

from pyqtgraph.Qt import QtGui

from test_CSTMD1Visualiser import TestCSTMD1Visualiser
from test_Compartment import TestCompartment
from test_Cstmd1 import TestCstmd1
from test_MCN import TestMCN
from test_NeuronCollection import TestNeuronCollection
from test_NeuronGenerator import TestNeuronGenerator


def test_suite():

    testList = [TestCompartment,
                TestMCN,
                TestNeuronCollection,
                TestNeuronGenerator,
                TestCstmd1,
                TestCSTMD1Visualiser]

    testLoad = unittest.TestLoader()

    caseList = []
    for testCase in testList:
        testSuite = testLoad.loadTestsFromTestCase(testCase)
        caseList.append(testSuite)

    return unittest.TestSuite(caseList)

if __name__ == "__main__":

    global app
    app = QtGui.QApplication.instance()
    if app is None:
        app = QtGui.QApplication(sys.argv)

    runner = unittest.TextTestRunner(verbosity = 2)
    runner.run(test_suite())

    app.quit()