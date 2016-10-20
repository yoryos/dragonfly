import os
import unittest

from CSTMD1.Morphology.NeuronGenerator import NeuronGenerator

TESTING_DIR = os.path.join(os.path.dirname(__file__), "testingDirectory/")
TREES = os.path.join(os.path.dirname(__file__), "../Morphology/trees")


class TestNeuronGenerator(unittest.TestCase):
    def setUp(self):
        self.ng = NeuronGenerator()
        self.ng.debug = True
        self.outputDir = TESTING_DIR
        self.filePrefix = "testNeuron"
        self.clean()

    def test_Points(self):
        self.assertEqual(len(self.ng.generate_compartment_points([0, 0, 0], 1000)),
                         1000, "Incorrect number of compartment points generated")

    def test_GenerateNeurons(self):

        self.ng.generate_neuron_morphologies(2, 10, self.outputDir, self.filePrefix, trees_path=TREES)
        assert os.path.exists(self.outputDir + self.filePrefix + str(0) + ".swc")
        assert os.path.exists(self.outputDir + self.filePrefix + str(1) + ".swc")

        with self.assertRaises(OSError):
            self.ng.generate_neuron(0, 10, 0, self.outputDir, self.filePrefix, trees_path=TREES)

    def tearDown(self):
        self.clean()

    def clean(self):
        try:
            os.remove(self.outputDir + self.filePrefix + str(0) + ".swc")
        except(OSError):
            pass
        try:
            os.remove(self.outputDir + self.filePrefix + str(1) + ".swc")
        except(OSError):
            pass

        try:
            os.rmdir(self.outputDir)
        except(OSError):
            pass
