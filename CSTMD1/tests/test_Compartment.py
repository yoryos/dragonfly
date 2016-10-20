from CSTMD1.Morphology.Compartment import Compartment
from Helper.Vectors import Vector_3D
import unittest

class TestCompartment(unittest.TestCase):


    def setUp(self):

        self.a = Vector_3D(1, 1, 0)
        self.b = Vector_3D(1, 1, 5)
        self.c = Vector_3D(1, 1, 10)
        self.d = Vector_3D(1, 1, 20)
        self.e = Vector_3D(1, 1, 30)
        self.f = Vector_3D(1, 1, 40)
        self.g = Vector_3D(1, 1, 50)

        self.c1 = Compartment(0,self.a, self.c)
        self.c2 = Compartment(1,self.c, self.d)
        self.c3 = Compartment(2,self.d, self.e)
        self.c4 = Compartment(3,self.a, self.b)
        self.c5 = Compartment(4,self.c, self.f)
        self.c6 = Compartment(5,self.f, self.g)


        self.c1.stepsToSoma = 0
        self.c2.add_parent_relationship(self.c1)
        self.c3.add_parent_relationship(self.c2)
        self.c5.add_parent_relationship(self.c1)
        self.c6.add_parent_relationship(self.c3)

    def tearDown(self):

        self.a = None
        self.b = None
        self.c = None
        self.c1 = None
        self.c2 = None

    def test_steps_to_soma(self):

        self.assertEqual(self.c6.steps_to_root(),3)
        self.assertEqual(self.c6.steps_to_root(True),3)
        self.assertEqual(self.c3.steps_to_root(True),2)

    def test_midpoint(self):

        self.assertEqual(self.c1.midpoint(), self.b, "Incorrect midpoint")

    def test_adding_parent(self):

        self.assertEqual(self.c1, self.c2.parent, "Parent not found")
        self.assertIn(self.c2, self.c1.children, "Child not found")
        self.assertIn(self.c5, self.c2.siblings, "Sibling not found")
        self.assertIn(self.c2, self.c5.siblings, "Sibling not found")
        self.assertEquals(self.c2.steps_to_root(), 1, "Incorrect path to soma")

    def test_removing_parent(self):

        self.c2.remove_parent_relationship()
        self.assertIsNone(self.c2.parent,"Parent not removed")

    def test_removing_non_existent_parent(self):

        with self.assertRaises(AssertionError):
            self.c1.remove_parent_relationship()

    def test_length(self):

        self.assertEqual(self.c1.length(), 10, "Incorrect length")

    def test_split(self):

        newCompartment = self.c1.split(3)

        self.assertEquals(newCompartment.idx, 3, "Incorrect child idx")
        self.assertEquals(newCompartment.start, self.b, "Incorrect child start")
        self.assertEquals(newCompartment.end, self.c, "Incorrect child end")
        self.assertEquals(newCompartment.children, [self.c2, self.c5], "Incorrect child children")
        self.assertEquals(newCompartment.parent, self.c1, "Incorrect child parent")
        self.assertEquals(self.c1.end, self.b, "Incorrect parent end")
        self.assertEquals(self.c1.children, [newCompartment], "Incorrect parent child")

    def test_can_join(self):

        self.assertTrue(self.c2.can_join(), "Should be able to join")
        self.assertFalse(self.c1.can_join(), "Should not be able to join")

    def test_join(self):

        toRemove = self.c2.join_with_child()

        self.assertIsNotNone(toRemove)

        self.assertEquals(self.c2.start, self.c, "Incorrect starting coordinate for joined")
        self.assertEquals(self.c2.end, self.e, "Incorrect end coordinate for joined")
        self.assertEquals(self.c1.children, [self.c2, self.c5], "Child is not removed from join")

    def test_string(self):

        self.assertIsNotNone(str(self.c1))
        self.assertIsNotNone(str(self.c2))
        self.assertIsNotNone(str(self.c4))
