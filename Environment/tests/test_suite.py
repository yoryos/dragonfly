import unittest
from test_animationwindow import TestAnimationWindow
from test_dragonfly import TestDragonfly
from test_target import TestTarget
from test_background import TestBackground
from test_environment import TestEnvironment

def test_suite():

    testList = [TestAnimationWindow,
                TestDragonfly,
                TestTarget,
                TestBackground,
                TestEnvironment]

    testLoad = unittest.TestLoader()

    caseList = []
    for testCase in testList:
        testSuite = testLoad.loadTestsFromTestCase(testCase)
        caseList.append(testSuite)

    return unittest.TestSuite(caseList)

if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity = 2)
    runner.run(test_suite())
