"""
Dragonfly Brain

Integration of seperate neurons into pipeline

__author__: Dragonfly Project 2016 - Imperial College London
            ({anc15, cps15, dk2015, gk513, lm1015, zl4215}@imperial.ac.uk)

"""
import datetime
import getpass
import os
import pprint
import time

import numpy as np
import matplotlib.pyplot as plt

from RL.RL4 import RL
from Environment.Environment import Environment
from Helper.Configurer import Configurer, EnvironmentConfigurer



class RL_Test_Loop(object):
    """Class for testing RL with Environment"""
    stage_Setup, stage_Environment, stage_ESTMD, stage_CSTMD1, stage_STDP, stage_RL, stage_ALL = range(7)
    Names = ['Setup', 'Environment', 'ESTMD', 'CSTMD1', 'STDP', 'RL', 'ALL']

    def __init__(self,
                 dt,
                 cstmd_buffer_size,
                 environment_config, estmd_config, dragonfly_config, cstmd_config, stdp_config,
                 run_until=None,
                 run_id=None,
                 overwrite=None,
                 pyplot=False,
                 cuda_device=0,
                 webcam=False):

        """

        Args:
            dt ():
            cstmd_buffer_size ():
            environment_config ():
            estmd_config ():
            dragonfly_config ():
            cstmd_config ():
            stdp_config ():
            run_until ():
            run_id ():
            overwrite ():
            pyplot ():
            cuda_device ():
            webcam ():

        """

        print "====================BUILDING DRAGONFLY BRAIN===================="

        self.start_time = time.time()
        self.end_time = self.start_time
        self.env_time = self.estmd_time = self.cstmd_time = self.stdp_time = 0
        self.webcam = webcam
        self.global_time = 0
        self.dt = float(dt)
        self.debug = False
        self.pyplot = pyplot
        self.configurer = Configurer(dragonfly_config)
        self.config = self.configurer.config_section_map('Simulation_Parameters')
        self.run_until = run_until

        if run_id is None:
            st = datetime.datetime.fromtimestamp(self.start_time).strftime('%H-%M')
            run_id = getpass.getuser() + "_" + st
            print "DEFAULTED run_id generated: " + run_id
        self.run_id = run_id


        print "Dragonfly simulation parameters: "
        pprint.pprint(self.config)
        print

        print "=====Attempting to build Environment Module====="
        env_config = EnvironmentConfigurer(environment_config)
        env_general = env_config.config_section_map("General")
        print "Environment General parameters: "
        pprint.pprint(env_general)
        print
        success, targets = env_config.get_targets()
        if not success:
            raise Exception
        gauss_constants = [env_general["ga"],env_general["gb"],np.deg2rad(env_general["gc_deg"]),env_general["gd"]]
        self.environment = Environment(background_path=env_general["background"],
                                        background_pos=(env_general["background_x"],env_general["background_y"]),
                                       dt=env_general["dt"],
                                       ppm=env_general["ppm"],
                                       width=env_general["width"],
                                       height=env_general["height"],
                                       target_config=targets,
                                       reward_version=env_general["reward_version"],
                                       rewards=(env_general["reward_pos"],env_general["reward_neg"]),
                                       max_angle=np.deg2rad(env_general["max_angle_deg"]),
                                       gauss_constants=gauss_constants,
                                       run_id=self.run_id)
        print "=====Constructed Environment Module===== "

        self.environment.dragonfly.visible = True

        self.distances = []
        self.distances.append(self.environment.get_distance_to_closest_target())
        self.times = []
        self.times.append(self.global_time)

        self.rl_output = [0,0,0,0]

        print "=====Constructed RL Module====="
        self.rl = RL(run_id =self.run_id, topology=[4,4])

        print "====================FINISHED BUILDING DRAGONFLY BRAIN===================="
        self.end_time = time.time()
        print "Finished construction in %f seconds" % (self.end_time - self.start_time)

    def check_if_need_to_return(self, stage, status=True, data=None, error=False):

        if not status:
            print "Error at step " + str(self.global_time) + " in module " + self.Names[stage]
            return True, True

        if data is not None and self.debug:
            print self.Names[stage] + " output: ", data

        if self.run_until <= stage:
            self.global_time += self.dt
            print "TIME UPDATED TO " + str(self.global_time) + "ms"
            return True, False

        return False, False,

    def step(self):
        self.end_time = time.time()

        frame, reward = self.environment.step(self.rl_output)
        now = time.time()
        self.env_time += (now - self.end_time)
        self.end_time = now

        self.distances.append(self.environment.get_distance_to_closest_target())
        self.times.append(self.global_time)

        ret, error = self.check_if_need_to_return(self.stage_Environment, data=reward)
        if ret or error:
            return False if error else True

        #rl_input = [True, False, False, False]
        if self.global_time % 2.0:
            rl_input = [False,True,False,False]
        else:
            rl_input = [False,False,False,True]

        self.rl_output = self.rl.step(rl_input,reward)
        ret, error = self.check_if_need_to_return(self.stage_RL, data=self.rl_output)
        if ret or error:
            return False if error else True

        self.global_time += self.dt
        print "TIME UPDATED TO " + str(self.global_time) + "ms"
        return True

    def data_dump(self, Environment=False, RL=False, directory=None, run_id_prefix=False):

        if not directory:
            if not os.path.isdir("DATA"):
                os.mkdir("DATA")

            day_folder = os.path.join("DATA", str(datetime.datetime.now().date()))

            if not os.path.isdir(day_folder):
                os.mkdir(day_folder)

            directory = os.path.join(os.path.abspath(os.getcwd()), (os.path.join(day_folder, str(self.run_id))))

        if not os.path.isdir(directory):
            os.mkdir(directory)

        if Environment and self.run_until >= self.stage_Environment:
            self.environment.get_video(directory=directory, run_id_prefix=run_id_prefix)

        if RL and self.run_until >= self.stage_RL:
            self.rl.save_weights(directory=directory, run_id_prefix=run_id_prefix)
            self.plot(directory)

    def time_analysis(self):

        print "Full loop took %f seconds" % (self.end_time - self.start_time)
        if self.run_until >= self.stage_Environment:
            print "Time in environment: ", self.env_time

    def plot(self, directory=None):
        plt.plot(self.times, self.distances)
        plt.xlabel('Time (ms)')
        plt.ylabel('Distance (pixels)')
        plt.title('Distance of Dragonfly from Target with RL')
        if not directory:
            plt.show()
        else:
            path = os.path.join(directory, "rl_distance.png")
            plt.savefig(path)
