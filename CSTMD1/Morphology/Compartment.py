'''
Compartment

Class for compartment, basic building block for multi-compartmental neuron

__author__: Dragonfly Project 2016 - Imperial College London ({anc15, cps15, dk2015, gk513, lm1015,
zl4215}@imperial.ac.uk)

'''

from Helper.Vectors import Vector_3D

class Compartment(object):
    """ A model of a compartment of a multi-compartmental neuron

    Attributes:
            idx (int): (unique) identifier for the compartment
            start (Vector_3D): start coordinate for the compartment
            end (Vector_3D): end coordinate for the compartment
            radius (Optional[float]): radius of the compartment

            parent(Optional[Compartment]): parent compartment
            children(List[Compartment]): a list of all children of compartment
            siblings(List[Compartment]): a list of all siblings of the compartment. A sibling is defined any
                compartment which shares the same parent
    """
    idx = None

    start = None
    end = None
    radius = None

    parent = None
    children = None
    siblings = None
    axon_comp = False

    def __init__(self, idx, start, end, radius = None):
        """
        Coordinate constructor

        Args:
            idx (int): defaults to None, an identifier for the compartment
            start (Vector_3D): defaults to None, the start coordinate for the compartment
            end (Vector_3D): defaults to None, the end coordinate for the compartment
            radius(Optional[float)]): radiufs of the compartment
        """
        self.idx = int(idx)
        self.start = start
        self.end = end
        self.radius = radius
        self.connectedCompartments = []
        self.children = []
        self.siblings = []

    def midpoint(self):
        """
        Calculate the midpoint of a compartment

        Returns:
            Vector_3D: the midpoint of the compartment

        """

        return self.start.midpoint(self.end)

    def add_parent_relationship(self, parent):
        """
        Add a parent child relationship where self is the child.

        Example:
            >>> a = Compartment()
            >>> b = Compartment()
            >>> a.add_parent_relationship(b)
            # Results in tree c->[b]
            >>> c.add_parent_relationship(b)
            # Results in tree c->[a,b] and a and b are listed a siblings



        Raises:
            AssertionError: if: 1) the compartment already has a parent, 2) if is already listed as a child for the
                parent 3) it is already listed as a sibling for one of the parents children

        Args:
            parent (Compartment): parent to add

        """

        self.__add_parent(parent)

        self.siblings += parent.children

        for sibling in parent.children:
            sibling.__add_sibling(self)

        parent.__add_child(self)

    def steps_to_root(self, save = False):
        """
        Gives the number of steps from the compartment to the root of the tree. If the compartment has no parent,
        it is taken as the root and 0 is returned

        Returns:
            int: the number of steps to the root of the tree
        """
        if self.parent is None:
            return 0
        else:
            if save and hasattr(self, "steps") and self.steps is not None:
                return self.steps
            else:
                if save:
                    self.steps = self.parent.steps_to_root(save) + 1
                    return self.steps
                else:
                    return self.parent.steps_to_root(save) + 1

    def remove_parent_relationship(self):
        """
        Remove the parent relationship between this compartment and its parent

        Example:
            Given the tree c->[a,b]

            >>> a.remove_parent_relationship()
            #c->[b] a where a is removed from the child list of a and the sibling list of c

        Raises:
            AssertionError: if
                - the compartment does not have a parent
                - it is not listed as a child of its parent
                - there is inconsistency in the sibling record
        """
        parent = self.parent
        self.__remove_parent()

        for sibling in list(self.siblings):
            sibling.__remove_sibling(self)
            self.__remove_sibling(sibling)

        parent.__remove_child(self)

    def __add_sibling(self, sibling):
        """
        Add a compartment as a sibling

        Raises:
            AssertionError: if the compartment is already a sibling

        Args:
            sibling (Compartment): compartment to add as a sibling
        """
        assert sibling not in self.siblings
        self.siblings.append(sibling)

    def __remove_sibling(self, sibling):
        """
        Remove a compartment as a sibling

        Raises:
            AssertionError: if the compartment is not a sibling

        Args:
            sibling (Compartment): compartment to remove as a sibling
        """
        assert sibling in self.siblings
        self.siblings.remove(sibling)

    def __add_child(self, child):
        """
        Add a compartment as a child

        Raises:
            AssertionError: if the compartment is already a child

        Args:
            child (Compartment): compartment to add as a child
        """
        assert child not in self.children
        self.children.append(child)

    def __remove_child(self, child):
        """
        Remove a compartment as a child

        Raises:
            AssertionError: if the compartment is no a child

        Args:
            child (Compartment): compartment to remove as a child
        """
        assert child in self.children
        self.children.remove(child)

    def __add_parent(self, parent):
        """
        Add parent to compartment

        Raises:
            AssertionError: if there is already a parent of the compartment. It must be removed first

        Args:
            parent (Compartment): parent to add
        """
        assert self.parent is None
        self.parent = parent

    def __remove_parent(self):

        """
        Remove the compartment's parent

        Raises:
            AssertionError: if the compartment does not have a parent
        """
        assert not self.parent is None
        self.parent = None

    def length(self):
        """
        Check the length of the compartment

        Returns:
            float: length of the compartment
        """

        return self.start.distance(self.end)

    def split(self, new_idx):
        """
        Split a compartment in two creating a new compartment starting at the midpoint and ending at the endpoint of
        the original compartment.

        Args:
            new_idx (int): identifier for the new compartment

        Raises:
            TypeError: if the midpoint of the compartment cannot be calculated
            AssertionError: if there is an inconsistency in the tree

        Returns:
            Compartment: the new compartment

        """
        mid = self.midpoint()

        new_compartment = Compartment(new_idx, mid, self.end)

        self.end = mid

        for child in list(self.children):
            child.remove_parent_relationship()
            child.add_parent_relationship(new_compartment)

        new_compartment.add_parent_relationship(self)

        if self.axon_comp:
            new_compartment.axon_comp = True

        return new_compartment

    def can_join(self):
        """
        Checks whether a compartment can join with its child

        Example:
               c ->[(b),(a -> d)]

            >>> a.can_join()
            True
            >>> c.can_join()
            False
            >>> b.can_join()
            True

        Returns:
            bool: if the compartment can join with its child. i.e. does the compartment have exactly one child

        """
        return len(self.children) == 1

    def join_with_child(self):
        """
        Join a compartment with its child. The compartment the method is called to is updated to end at the child's
        endpoint.

        Raises:
            AssertionError: if the compartment cannot joint with it's child or there is an inconsistency in the tree

        Returns:
            Compartment: the child compartment which can now be removed

        """
        assert self.can_join()

        child = self.children[0]
        self.children = []
        self.end = child.end

        for new_child in list(child.children):
            new_child.remove_parent_relationship()
            new_child.add_parent_relationship(self)

        return child

    def __str__(self):
        """
        Get a string representation of the compartment

        Returns:
            str: string representation
        """
        status = "[idx:" + str(self.idx) + \
                 ", Diameter: " + str(self.radius) + \
                 ", SomaDistance: " + str(self.steps_to_root()) + \
                 ", Start: " + str(self.start) + \
                 ", End: " + str(self.end) + \
                 ", Parent: "

        if self.parent is not None:
            status += str(self.parent.idx)

        if self.children == [] and self.siblings == []:
            status += "]"
        else:
            status += ", Children ("
            for comp in self.children:
                status += (str(comp.idx) + ", ")
            status +="]"

        if self.siblings == []:
            status += "]"
        else:
            status += ", Siblings ("
            for comp in self.siblings:
                status += (str(comp.idx) + ", ")
            status +=")]"

        return status

    def __eq__(self, compartment):
        """
        Checks whether two compartments are equivalent, even if they are not the same object

        Args:
            compartment (Compartment): compartment to compare with

        Returns:
            bool: True if equivalent, False it not

        """
        return self.start == compartment.start and \
               self.end == compartment.end and \
               self.idx == compartment.idx and \
               self.parent == compartment.parent and \
               self.children == compartment.children
