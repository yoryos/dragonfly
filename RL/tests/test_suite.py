import unittest
from test_RL import TestRL

def test_suite():

    testList = [TestRL]

    testLoad = unittest.TestLoader()

    caseList = []
    for testCase in testList:
        testSuite = testLoad.loadTestsFromTestCase(testCase)
        caseList.append(testSuite)

    return unittest.TestSuite(caseList)

if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity = 2)
    runner.run(test_suite())
