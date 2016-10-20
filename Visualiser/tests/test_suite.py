import unittest
from pyqtgraph.Qt import QtGui
from test_DragonflyVisualiser import TestDragonflyVisualiser
from test_StdpVisualiser import TestStdpVisualisers
from test_VideoVisualiser import TestVideoVisualiser
from test_RasterVisualiser import TestRasterVisualiser
import sys

def test_suite():

    testList = [TestStdpVisualisers,
                TestDragonflyVisualiser,
                TestVideoVisualiser,
                TestRasterVisualiser]

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