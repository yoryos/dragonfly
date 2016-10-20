import STDP.pyboParameterSearch as pps
import numpy as np
import unittest
import os
import pickle

class pyboParameterSearchTests(unittest.TestCase):

    def test_example(self):
        pps.main(['dummy','example'])
        pass
