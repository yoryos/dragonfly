import unittest

import numpy as np

from Helper.Vectors import Vector_3D, Vector_2D


class Test3DVector(unittest.TestCase):
    def setUp(self):
        self.a = Vector_3D(1, 1, 0)
        self.b = Vector_3D(1, 1, 10)
        self.c = Vector_3D(1, 1, 0)

    def tearDown(self):
        self.a = None
        self.b = None

    def test_normalise(self):

        np.testing.assert_almost_equal(self.a.unit_vector(), np.array([np.sqrt(2)/2.0, np.sqrt(2)/2.0, 0.0]))

    def test_access(self):
        self.assertEquals(self.a.x, 1, "Incorrect x field")
        self.assertEquals(self.a.y, 1, "Incorrect y field")
        self.assertEquals(self.a.z, 0, "Incorrect z field")

    def test_map_to_2d(self):
        self.assertEquals(self.a.project("x"), Vector_2D(1, 0), "Incorrect x projection")
        self.assertEquals(self.a.project("y"), Vector_2D(1, 0), "Incorrect y projection")
        self.assertEquals(self.a.project("z"), Vector_2D(1, 1), "Incorrect z projection")

    def test_distance(self):
        self.assertEqual(self.a.distance(self.b), 10, "Incorrect distance")

    def test_list(self):
        self.assertEquals(self.a.to_list(), [1, 1, 0], "Incorrect list representation")

    def test_array(self):
        np.testing.assert_array_equal(self.a.to_array(), np.array([1, 1, 0]), "Incorrect array representation")

    def test_midpoint(self):
        self.assertEquals(self.a.midpoint(self.b), Vector_3D(1, 1, 5), "Incorrect midpoint")

    def test_mid(self):
        self.assertEquals(self.a.mid(self.b, 0.2), Vector_3D(1, 1, 2), "Incorrect 0.2 mid")

    def test_add(self):
        self.assertEquals(self.a + self.b, Vector_3D(2, 2, 10), "Incorrect Addition")

    def test_iadd(self):
        self.a += Vector_3D(1, 2, 3)
        self.assertEquals(self.a, Vector_3D(2, 3, 3))

    def test_isub(self):
        self.a -= Vector_3D(1, 2, 3)
        self.assertEquals(self.a, Vector_3D(0, -1, -3))

    def test_mult(self):
        self.assertEquals(self.b * 2, Vector_3D(2, 2, 20), "Incorrect multiplication")

    def test_sub(self):
        self.assertEquals(self.b - self.a, Vector_3D(0, 0, 10), "Incorrect subtraction")

    def test_div(self):
        self.assertEquals(self.b / 2.0, Vector_3D(0.5, 0.5, 5), "Incorrect division")

    def test_string(self):
        print str(self.a)
        self.assertEqual(str(self.a), "(1.0,1.0,0.0)", "Incorrect string")

    def test_equals(self):
        self.assertEqual(self.a, self.c, "Incorrect equality")


class Test2DVector(unittest.TestCase):
    def setUp(self):
        self.a = Vector_2D(1.0, 1.0)
        self.b = Vector_2D(1, 10)
        self.c = Vector_2D(1, 1.0)

    def test_access(self):
        self.assertEquals(self.a.x, 1, "Incorrect x field")
        self.assertEquals(self.a.y, 1, "Incorrect y field")

    def test_normalise(self):

        np.testing.assert_almost_equal(self.a.unit_vector(), np.array([np.sqrt(2)/2.0, np.sqrt(2)/2.0]))

    def test_angle(self):

        self.assertAlmostEqual(Vector_2D(1.0,1.0).angle(Vector_2D(1.0,0)), np.pi/4, 5)

    def test_distance(self):

        self.assertEqual(self.a.distance(self.b), 9, "Incorrect distance")

    def test_list(self):
        self.assertEquals(self.a.to_list(), [1, 1], "Incorrect list representation")

    def test_array(self):
        np.testing.assert_array_equal(self.a.to_array(), np.array([1, 1]), "Incorrect array representation")

    def test_midpoint(self):
        self.assertEquals(self.a.midpoint(self.b), Vector_2D(1, 5.5), "Incorrect midpoint")

    def test_mid(self):
        print self.a.mid(self.b, 0.2)
        self.assertEquals(self.a.mid(self.b, 0.2), Vector_2D(1, 1 + (10 - 1) * 0.2), "Incorrect 0.2 mid")

    def test_add(self):
        res = self.a + self.b
        self.assertEquals(res, Vector_2D(2, 11), "Incorrect Addition")
        self.assertEquals(res.x, 2, "Could no access x element after addition")
        self.assertEquals(res.y, 11, "Could no access y element after addition")

    def test_iadd(self):
        self.a += Vector_2D(1, 2)
        self.assertEquals(self.a, Vector_2D(2, 3))

    def test_isub(self):
        self.a -= Vector_2D(1, 2)
        self.assertEquals(self.a, Vector_2D(0, -1))

    def test_mult(self):
        self.assertEquals(self.b * 2, Vector_2D(2, 20), "Incorrect multiplication")

    def test_sub(self):
        self.assertEquals(self.b - self.a, Vector_2D(0, 9), "Incorrect subtraction")

    def test_div(self):
        self.assertEquals(self.b / 2.0, Vector_2D(0.5, 5), "Incorrect division")

    def test_string(self):

        self.assertEqual(str(self.a), "(1.0,1.0)", "Incorrect string")

    def test_equals(self):
        self.assertEqual(self.a, self.c, "Incorrect equality")
