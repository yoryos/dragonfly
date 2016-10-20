from Helper.BrainModule import BrainModule
import unittest
import os
import numpy as np

TEST_DIR = "temp_test"
SUB_DIR = "sub_dir"
TEST_ALT = "alt_temp_test"
F_NAME = "out.dat"
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__),  TEST_DIR)
TEST_ALT_DATA_DIR = os.path.join(os.path.dirname(__file__), TEST_ALT)


class TestBrainModule(unittest.TestCase):
    def setUp(self):
        self.bm = BrainModule(TEST_DATA_DIR)
        self.bm.prefix = ""
        self.bm.reset_default_output_directory()

    def test_get_full_output_name(self):

        with_run_id = self.bm.get_full_output_name("test_file.dat", run_id_prefix=True)
        expected = os.path.join(self.bm.prefix + TEST_DATA_DIR, TEST_DATA_DIR + "_test_file.dat")
        self.assertEqual(with_run_id, expected,
                         "Wrong output name found when prefixing with run_id")
        without_run_id = self.bm.get_full_output_name("test_file.dat", run_id_prefix=False)
        expected = os.path.join(self.bm.prefix + TEST_DATA_DIR, "test_file.dat")
        self.assertEqual(without_run_id, expected,
                         "Wrong output name found when not prefixing with run_id")

    def test_set_reset_output_dir(self):

        self.bm.set_output_directory(TEST_ALT_DATA_DIR)
        found = self.bm.get_output_directory(SUB_DIR)
        self.assertEqual(found, os.path.join(TEST_ALT_DATA_DIR, SUB_DIR),
                         "Wrong output directory found when using non default")

        self.bm.reset_default_output_directory()
        found = self.bm.get_output_directory(SUB_DIR)
        self.assertEqual(found, os.path.join(self.bm.prefix + TEST_DATA_DIR, SUB_DIR),
                         "Wrong output directory after resetting to default")

    def test_get_output_dir(self):

        f = self.bm.get_output_directory()
        e = self.bm.prefix + TEST_DATA_DIR
        self.assertEqual(f, e, "Wrong output dir when no sub specified")
        f = self.bm.get_output_directory(SUB_DIR)
        e = os.path.join(self.bm.prefix + TEST_DATA_DIR, SUB_DIR)
        self.assertEqual(f, e, "Wrong output dir when sub specified")

    def test_save_text_numpy(self):

        boolean_array = np.array([True, False])
        self.bm.save_numpy_array(boolean_array, name=F_NAME, fmt="%i")

        found = np.loadtxt(os.path.join(TEST_DATA_DIR, F_NAME))
        np.testing.assert_array_equal(found, boolean_array.astype(int))

    def test_save_npy_numpy(self):

        float_array = np.array([1.1, 2.2])
        self.bm.save_numpy_array(float_array, name=F_NAME, npz=True)

        found = np.load(os.path.join(TEST_DATA_DIR, F_NAME + ".npy"))
        np.testing.assert_array_equal(found, float_array)

        os.remove(os.path.join(TEST_DATA_DIR, F_NAME + ".npy"))

    def test_save_dictionary(self):

        d = {"key1": True, "key2": 1, "key3": "string"}

        self.bm.save_dictionary(d, name=F_NAME)

        f = []
        e = ["key1:True\n", "key2:1\n", "key3:string\n"]
        with open(os.path.join(TEST_DATA_DIR, F_NAME)) as file:
            for line in file:
                f.append(line)
        f.sort()
        e.sort()
        self.assertListEqual(f, e)

    def test_none_numpy_array(self):

        self.assertFalse(self.bm.save_numpy_array(None))

    def tearDown(self):
        try:
            os.remove(os.path.join(TEST_DATA_DIR, F_NAME))
        except:
            pass
        try:
            os.rmdir(os.path.join(TEST_DATA_DIR, SUB_DIR))
        except:
            pass
        try:
            os.rmdir(TEST_DATA_DIR)
        except:
            pass
        try:
            os.rmdir(os.path.join(TEST_ALT_DATA_DIR, SUB_DIR))
        except:
            pass
        try:
            os.rmdir(TEST_ALT_DATA_DIR)
        except:
            pass