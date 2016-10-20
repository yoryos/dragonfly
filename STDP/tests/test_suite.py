import unittest
from test_neuron import NeuronTests
from test_pyboParameterSearch import pyboParameterSearchTests
from test_stdp import StdpTests
from test_sample import SampleGeneratorTests

def test_suite():

    testList  = [NeuronTests,
                 StdpTests,
                 SampleGeneratorTests
                 ]

    testLoad = unittest.TestLoader()

    caseList = []
    for testCase in testList:
        testSuite = testLoad.loadTestsFromTestCase(testCase)
        caseList.append(testSuite)

    return unittest.TestSuite(caseList)

if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity = 2)
    runner.run(test_suite())
