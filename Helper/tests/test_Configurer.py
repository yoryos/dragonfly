from Helper.Configurer import Configurer, EnvironmentConfigurer
import unittest
import os
import numpy as np

TEST_CONFIG = os.path.join(os.path.dirname(__file__), "testConfig.ini")


class TestConfigurer(unittest.TestCase):
    def setUp(self):
        self.config = Configurer(TEST_CONFIG)

    def test_get_section(self):
        expected = {"an_int": 1, "a_bool": True, "a_float": 1.1, "a_string": "a_string"}
        found = self.config.config_section_map("General")

        self.assertDictEqual(expected, found, "Did not load test General section correctly")

    def test_override_variable_in_get(self):
        override = {"a_bool": False, "a_float": 0.1, "a_string": "another_string"}
        found = self.config.config_section_map("General", override)
        expected = {"an_int": 1, "a_bool": False, "a_float": 0.1, "a_string": "another_string"}
        self.assertDictEqual(expected, found, "Did override section when getting dictionary")

    def test_override_variable_in_constructor(self):
        override = {"a_bool": False, "a_float": 0.1, "a_string": "another_string"}
        c = Configurer(TEST_CONFIG, override)
        found = c.config_section_map("General", override)
        expected = {"an_int": 1, "a_bool": False, "a_float": 0.1, "a_string": "another_string"}
        self.assertDictEqual(expected, found, "Did not override section correctly in constructor")

    def test_set_to_none_or_false(self):
        found = self.config.config_section_map("NullSection")
        self.assertDictEqual(found, {"none_attribute": None, "none_bool_attribute": False},
                             "Did not set null attributes to None and False")

    def test_invalid_config_path(self):
        with self.assertRaises(IOError):
            config = Configurer("wrong_path.ini")

    def tearDown(self):
        pass


class TestEnvironmentConfigurer(unittest.TestCase):
    def setUp(self):
        self.config = EnvironmentConfigurer(TEST_CONFIG)

    def test_get_targets(self):
        status, targets = self.config.get_targets("Target_")
        self.assertTrue(status, "Could not load targets correctly")
        print targets

        t1 = {'size': 10.0, 'velocity': np.array([0., 10.]),
              'position': np.array([320., 0.]), 'wobble': 1.0,
              'color':[0,0,0]}
        t1_found = targets[0]
        self.assertEqual(t1_found['size'], t1['size'])
        self.assertEqual(t1_found['wobble'], t1['wobble'])
        self.assertListEqual(t1_found['color'], t1['color'])
        np.testing.assert_array_equal(t1_found['velocity'], t1['velocity'])
        np.testing.assert_array_equal(t1_found['position'], t1['position'])

        t2 = {'size': 10.0, 'wobble': 1.0}
        self.assertDictEqual(targets[1], t2, "Could not get target 1 data correctly")

    def test_incorrect_target_format(self):

        status, targets = self.config.get_targets("Fail_1")
        self.assertFalse(status, "Wrong return value when incorrect target data given")

    def tearDown(self):
        pass
