"""
NeuronGenerator

Creates random targets inside a bounded area, generates a dentritic tree outputting the results as a SWC file

__author__: Dragonfly Project 2016 - Imperial College London ({anc15, cps15, dk2015, gk513, lm1015,
zl4215}@imperial.ac.uk)

Originally by Zafeirios Fountas - Imperial College London (zfountas@imperial.ac.uk)
"""

import random as rnd
from oct2py import octave
from mock import patch
import os


class NeuronGenerator(object):
    """Class to generate neuron SWC files
    """

    def __init__(self):
        self.debug = False
        pass


    def generate_neuron_morphologies(self, number_of_neurons, number_of_compartments,
                                     directory_path="./generatedMorphologies/",
                                     file_prefix="neuron", z_offset=25.0, safe=True, trees_path = "./trees"):
        """
        Generate the morphologies for a given number of neurons, each with a given number of compartments an a
        z_offset between each

        Args:
            number_of_neurons (int): Number of neurons to create morphologies for
            number_of_compartments (int): Number of compartments in each neuron
            directory_path (Optional[str]): Path to director to save swc file in, default "./generatedMorphologies/"
            file_prefix (Optional[str]): Prefix for output file, default "neuron"
            z_offset (Optional[float]): Offset between each neuron, default = 25.0
            safe (Optional[bool]): ensures that no files are overwritten
            trees_path(Optional[str]): path to Octave trees package

        Raises:
            OSError: if safe is true, and a file already exists at a write path
        """

        for n in xrange(number_of_neurons):
            self.generate_neuron(n, number_of_compartments, z_offset, directory_path, file_prefix, safe, trees_path)

    def generate_neuron(self, idx, number_of_compartments, z_offset=25.0, directory_path="./generatedMorphologies/",
                        file_prefix="neuron", safe=True, trees_path = "./trees"):
        """
        Generate the morphology for one neuron with index and a given number of compartments. The z-offset of the
        neuron is the idx * z_offset

        Args:
            idx (int): Index of the neuron
            number_of_compartments (int): Number of compartments in the neuron
            z_offset (Optional[float]): Offset of soma from origin, default = 25.0
            directory_path (): Path to director to save swc file in, default "./generatedMorphologies/"
            file_prefix (Optional[str]): Prefix for output file, default "neuron"
            safe (Optional[bool]): ensures that no files are overwritten
            trees_path(Optional[str]): path to Octave trees package

        Raises:
            OSError: if safe is true, and a file already exists at a write path
        """

        somaCoordinates = [0.0, 0.0, idx * z_offset]

        max_length = 1000.0
        scale = 100.5
        # Used by trees library for 'quaddiameter_tree' function from help: scale of diameter of root {DEFAULT: 0.5}
        # from help: added base diameter {DEFAULT: 0.5}
        offset = 10.0

        file = file_prefix + str(idx) + ".swc"
        path = directory_path + file

        if safe:
            if os.path.exists(path):
                print "File already exists in write location: " + path
                raise OSError()
            elif not os.access(os.path.dirname(directory_path), os.W_OK):
                os.makedirs(directory_path)

        if self.debug:
            print("Generating Compartments Locations")
        coordinates = self.generate_compartment_points(somaCoordinates, number_of_compartments)
        if self.debug:
            print("Generated Compartment Locations")

        if self.debug:
            print "Attempting to use trees package"
        octave.addpath(trees_path)
        check = octave.generateTree(trees_path, coordinates, scale, offset, max_length, path)
        if self.debug:
            print "Finished using trees package"
            print "Saved to" + check

    def generate_compartment_points(self, soma_coordinates, n):
        """
        Generates a list of n-9 random points within the CSTMD1 dentrite boundary and 9 specified coordinates from the
        soma to the dentrite boundary.

        Args:
            soma_coordinates (List[float,float,float): The neuron's soma coordinates
            n (int): number of compartment points

        Returns:
           List[(float,float,float)]: A list of coordinates used as the start and end of the compartments (soma_x,
           soma_y,soma_z) ... (xN,yN,zN)]
        """
        soma_x, soma_y, soma_z = soma_coordinates

        # INITIALIZATION
        # Set the boarders of area I
        border_xL = []
        border_xL.append(-700.0)
        border_xL.append(-625.0)
        border_xL.append(-675.0)
        border_xL.append(-750.0)
        border_xL.append(-750.0)
        border_xL.append(-700.0)
        border_xL.append(-700.0)
        border_xL.append(-837.5)
        border_xL.append(-900.0)
        border_xL.append(-1100.0)
        border_xL.append(-1200.0)
        border_xL.append(-1050.0)
        border_xL.append(-700.0)

        border_yL = []
        border_yL.append(150.0)
        border_yL.append(125.0)
        border_yL.append(62.5)
        border_yL.append(-125.0)
        border_yL.append(-175.0)
        border_yL.append(-225.0)
        border_yL.append(-412.5)
        border_yL.append(-675.0)
        border_yL.append(-725.0)
        border_yL.append(-625.0)
        border_yL.append(-300.0)
        border_yL.append(-25.0)
        border_yL.append(150.0)

        for a in range(len(border_xL)):
            border_xL[a] = border_xL[a] - 530

        for a in range(len(border_yL)):
            border_yL[a] = border_yL[a] + 300

        targets = []
        targets.append((soma_x, soma_y, soma_z))
        targets.append((-200.0, 260.0, soma_z))
        targets.append((-400.0, 300.0, soma_z))
        targets.append((-450.0, 290.0, soma_z))
        targets.append((-680.0, 260.0, soma_z))
        targets.append((-890.0, 200.0, soma_z))
        targets.append((-980.0, 130.0, soma_z))
        targets.append((-1030.0, 140.0, soma_z))
        targets.append((-1090.0, 20.0, soma_z))

        while len(targets) < n:
            x = rnd.uniform(-1750.0, -1000.0)
            y = rnd.uniform(-450.0, 460.0)
            z = rnd.uniform(-20.0 + soma_z, 20.0 + soma_z)
            if self.__point_inside_polygon(x, y, zip(border_xL, border_yL)):
                targets.append((x, y, z))

        return targets

    @staticmethod
    def __point_inside_polygon(x, y, poly):
        """
        Check if a randomly generated 2D point with coordinates (x,y) is inside a polygon poly

        Args:
            x (float): x-coordinate
            y (float): y-coordinate
            poly (List[(float,float)]): Polygon border coordinates as list of tuples of [(xb1, yb1),..(xbn,ybn)]

        Returns:
            bool: True if x, y are inside the polygon, alse false
        """

        n = len(poly)
        inside = False

        p1x, p1y = poly[0]
        for i in range(n + 1):
            p2x, p2y = poly[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside
