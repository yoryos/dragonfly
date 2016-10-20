"""
Environment

Class for environment controller

@author: Dragonfly Project 2016 - Imperial College London
        ({anc15, cps15, dk2015, gk513, lm1015, zl4215}@imperial.ac.uk)
"""

import numpy as np

from AnimationWindow import AnimationWindow
from Dragonfly import Dragonfly
from Target import Target
from Background import Background
from GeneralEnvironment import GeneralEnvironment
from Helper.Vectors import Vector_2D


class Environment(GeneralEnvironment):
    """
    Module representing dragonfly's  physical environment with targets
    """
    def __init__(self, run_id, dt=1.0, ppm=100.0, width=640.0, height=480.0, background_path=None,
                background_pos=(0.0, 0.0), target_config=[],
                 reward_version=0, rewards=(1.0, -1.0/3.0), max_angle=0.3*np.pi, gauss_constants=(6.0, 0.0, 0.25*np.pi, -3.0),velocity_gain=10.0):
        """
        Constructor

        Args:
            dt (Optional[int]): time step ms
            ppm (Optional[float]): pixels per meter
            width (Optional[int]): image width px
            height (Optional[int]): image height px
            background_path (Optional[str]): path to background image
            target_config (Optional[List[Dictionary]]): list of dictionaries,
                each defines non-default parameters for each of the targets
            reward_version (Optional[int]): 0 to not learn (always return 0 reward),
                                            1 to binary learn by distance,
                                            2 to binary learn by angle range (cone),
                                            3 to learn by gaussian of angle difference
            rewards (Optional[list: floats]): reward values for binary learning, positive then negative
            max_angle (Optional[float]): Max angle (cone range) for angle_learning=1 in radians
            gauss_constants, (Optional[list: floats]): Parameters for angle_learning=2 gaussian function

        Notes:
            target_config:
        """
        GeneralEnvironment.__init__(self, dt, run_id)

        self.time = 0
        self.ppm = ppm
        self.frame_dimensions = (width, height)
        self.background = None


        self.reward_version = reward_version
        self.reward_values = rewards
        self.max_angle = max_angle
        self.gauss_constants = gauss_constants
        self.dragonfly = Dragonfly(Vector_2D(width / 2.0, height / 2.0))
        self.velocity_gain = velocity_gain

        if background_path is not None:
            self.background = Background(background_path, background_pos)
        self.target_list = self.__generate_target_list(target_config)

        self.animator = AnimationWindow(self.run_id, self.target_list, int(width), int(height), self.background, self.dragonfly)

        self.distance_to_closest_target = self.get_distance_to_closest_target()

    def __generate_target_list(self, target_config):
        """
        Create a target list from a config list

        Args:
            target_config (Optional[List[Dictionary]]): list of dictionaries,
                each defines non-default parameters for each of the targets

        Returns:
            List[Target]: list of targets
        """
        target_list = []
        for target_config_item in target_config:
            position = Vector_2D(0.0, 0.0)
            velocity = Vector_2D(1.0,1.0)
            wobble = 1
            size = 1
            color = [0, 0, 0]
            if 'position' in target_config_item:
                pos_array = target_config_item['position']
                position = Vector_2D(pos_array[0], pos_array[1])
            if 'velocity' in target_config_item:
                velocity_array = target_config_item['velocity']
                velocity = Vector_2D(velocity_array[0], velocity_array[1])
            if 'wobble' in target_config_item:
                wobble = target_config_item['wobble']
            if 'size' in target_config_item:
                size = target_config_item['size']
            if 'color' in target_config_item:
                color = target_config_item['color']

            target_list.append(Target(position, velocity * (self.ppm / self.fps),
                                      wobble, size, color, len(target_list)))

        return target_list

    def step(self, rates=(0, 0, 0, 0), savePNG = False):
        """
        Move dragonfly and targets and return the next frame

        Args:
            rates (R1,R2,R3,R4): R1 -> R R2 -> L R3 = Down R4 = Left

        Return:
            ndarray: next frame
        """
        velocity = Vector_2D(rates[0] - rates[1], rates[2] - rates[3]) * (self.ppm / self.fps) * self.velocity_gain

        reward = 0.0

        if self.reward_version > 1:
            if velocity.to_array().any():
                smallest_angle = np.abs(self.get_smallest_angle(velocity))
                if self.reward_version == 2:
                    if np.abs(smallest_angle) <= self.max_angle:
                        reward = self.reward_values[0]
                    else:
                        reward = self.reward_values[1]
                else:
                    reward = self.get_gauss_reward(smallest_angle)

        if self.background is not None:
            self.background.update(velocity)

        for target in self.target_list:
            target.update(velocity)

        self.dragonfly.update(velocity)

        step_result = self.animator.draw(savePNG)

        if self.reward_version == 1:
            reward = self.get_distance_reward()

        self.frames.append(step_result)
        return (self.green_filter(step_result), reward)

    def get_distance_reward(self):
        """
        Return reward based upon distance to closest target
        """

        weight = self.reward_values[0]

        new_distance = self.get_distance_to_closest_target()

        if new_distance >= self.distance_to_closest_target:
            weight = self.reward_values[1]

        self.distance_to_closest_target = new_distance

        return weight

    def get_smallest_angle(self, velocity):
        """ Gets smallest angle between velocity angle and angle to targets """
        target_angles = self.get_target_angles(velocity)
        smallest_angle = 2 * np.pi

        if len(target_angles) > 0:
            return min(target_angles)

        return smallest_angle

    def get_gauss_reward(self, angle):
        """ Gaussian curve using smallest angle """
        a, b, c, d = self.gauss_constants
        return (d + a * np.power(np.e, -((angle - b)**2) / (2 * (c**2))))

    def get_distance_to_closest_target(self):
        """ Return distance to the closest target to dragonfly """
        closest_target = self.get_closest_target_to_dragonfly()
        if closest_target is None:
            return 0
        return self.dragonfly.perspective_position.distance(closest_target.position)

    def get_closest_target_to_dragonfly(self):

        d = None
        try:
            d = min(self.target_list, key=lambda x: self.dragonfly.perspective_position.distance(x.position))
        except ValueError:
            print "No targets returning"

        return d

    def get_target_angles(self,dragonfly_velocity):
        """ Return list of angles from dragonfly to targets """
        angles = []
        for t in self.target_list:
            #angles.append(self.dragonfly.perspective_position.angle_to(t.position))
            rel_target_position =  t.position - self.dragonfly.perspective_position
            angles.append((rel_target_position).angle(dragonfly_velocity))
        return angles

    def __str__(self):

        s = "Environment:\n"

        for t in self.target_list:
            s += "\t" + str(t) + "\n"

        s += "\t" + str(self.dragonfly) + " min distance: " + "{0:.2f}".format(self.get_distance_to_closest_target()) \
             + "px"

        return s
