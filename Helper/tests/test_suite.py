import unittest

from test_Configurer import TestConfigurer, TestEnvironmentConfigurer
from test_BrainModule import TestBrainModule
from test_Vectors import Test2DVector, Test3DVector
from test_VideoTools import TestVG, TestVTFC

def test_suite():

    testList = [TestConfigurer,
                TestEnvironmentConfigurer,
                TestBrainModule,
                Test2DVector,
                Test3DVector,
                TestVG,
                TestVTFC]

    testLoad = unittest.TestLoader()

    caseList = []
    for testCase in testList:
        testSuite = testLoad.loadTestsFromTestCase(testCase)
        caseList.append(testSuite)

    return unittest.TestSuite(caseList)

if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity = 2)
    runner.run(test_suite())
