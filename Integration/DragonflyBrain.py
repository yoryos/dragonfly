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

import matplotlib.pyplot as plt
import numpy as np

from RL.RL4 import RL
from CSTMD1.Cstmd1 import Cstmd1
from ESTMD.Estmd import Estmd
from Environment.Environment import Environment
from Environment.WebCamHandler import WebCamHandler
from Helper.Configurer import Configurer, EnvironmentConfigurer
from STDP.stdp import Stdp


class DragonflyBrain(object):
    """
    Integration framework that creates, owns and operates all modules as one continous flow cycle
    """
    stage_Setup, stage_Environment, stage_ESTMD, stage_CSTMD1, stage_STDP, stage_RL, stage_ALL = range(7)
    Names = ['Setup', 'Environment', 'ESTMD', 'CSTMD1', 'STDP', 'RL', 'ALL']

    def __init__(self,
                 dt,
                 cstmd_buffer_size,
                 environment_config, estmd_config, dragonfly_config, cstmd_config, stdp_config,
                 rl_weights=None,
                 run_until=stage_ALL,
                 run_id=None,
                 overwrite=None,
                 pyplot=False,
                 cuda_device=0,
                 webcam=False):


        """

        Args:
            dt (): Global time step in milliseconds
            cstmd_buffer_size (): Size of buffer memory for CSTMD1
            environment_config (): Config file path for Environment
            estmd_config (): Config file path for ESTMD
            dragonfly_config (): Config file path for DragonflyBrain
            cstmd_config (): Config file path for CSTMND
            stdp_config (): Config file path for STDP
            run_until (): Set which stage to run the simulation up to
            run_id (): Run ID for this run
            overwrite (): Overwrite parameters for CSTMND config
            pyplot (): Plot results
            cuda_device (): Select which device to stage CuDA on
            webcam (): Set whether to use WebCam in place of Environment

        """

        print "====================BUILDING DRAGONFLY BRAIN===================="

        self.start_time = time.time()
        self.end_time = self.start_time
        self.env_time = self.estmd_time = self.cstmd_time = self.stdp_time = self.rl_time = 0
        self.webcam = webcam
        self.global_time = 0
        self.dt = float(dt)
        self.debug = False
        self.pyplot = pyplot
        self.configurer = Configurer(dragonfly_config)
        self.config = self.configurer.config_section_map('Simulation_Parameters')
        self.run_until = run_until
        self.rl_output = (0, 0, 0, 0)

        if pyplot:
            plt.ion()
            plt.show()

        if run_id is None:
            st = datetime.datetime.fromtimestamp(self.start_time).strftime('%H-%M')
            run_id = getpass.getuser() + "_" + st
            print "DEFAULTED run_id generated: " + run_id
        self.run_id = run_id

        print "Dragonfly simulation parameters: "
        pprint.pprint(self.config)
        print

        if not self.webcam:
            print "=====Attempting to build Environment Module====="
            env_config = EnvironmentConfigurer(environment_config)
            env_general = env_config.config_section_map("General")
            print "Environment General parameters: "
            pprint.pprint(env_general)
            print
            success, targets = env_config.get_targets()
            if not success:
                raise Exception
            self.environment = Environment(background_path=env_general["background"],
                                           dt=env_general["dt"],
                                           ppm=env_general["ppm"],
                                           width=env_general["width"],
                                           height=env_general["height"],
                                           target_config=targets,
                                           reward_version=env_general["reward_version"],
                                           rewards=(env_general["reward_pos"],env_general["reward_neg"]),
                                           run_id=self.run_id)
            print "=====Constructed Environment Module===== "
        else:
            print "=====Using WebCam as Environment====="
            self.environment = WebCamHandler(dt=self.dt, run_id=self.run_id)
            print "WebCam Dimensions: ", self.environment.frame_dimensions

        self.rl_output = [0,0,0,0]

        if self.run_until < self.stage_ESTMD:
            self.end_time = time.time()
            print "Finished construction in %f seconds" % (self.end_time - self.start_time)
            return

        print "=====Attempting to build ESTMD Module====="
        estmd_config = Configurer(estmd_config).config_section_map("ESTMD_Parameters")
        print "ESTMD parameters: "
        pprint.pprint(estmd_config)
        print
        self.estmd = Estmd(input_dimensions=self.environment.frame_dimensions,
                           preprocess_resize=estmd_config["preprocess_resize"],
                           resize_factor=estmd_config["resize_factor"],
                           threshold=estmd_config["threshold"],
                           gain=estmd_config["gain"],
                           LMC_rec_depth=estmd_config["lmc_rec_depth"],
                           run_id=self.run_id)

        print "=====Constructed ESTMD Module===== "
        print "ESTMD output dimensions:", self.estmd.output_dimensions

        if self.run_until < self.stage_CSTMD1:
            self.end_time = time.time()
            print "Finished construction in %f seconds" % (self.end_time - self.start_time)
            return

        print "=====Attempting to build CSTMD Module====="
        cstmd_config = Configurer(cstmd_config)
        cstmd_config_pref = cstmd_config.config_section_map("Simulation_Parameters", overwrite)
        print "CSTMD1 top level parameters"
        print cstmd_config_pref
        try:
            preload  = cstmd_config_pref['preloaded_morphology_path']
        except KeyError:
            preload = None
        try:
            save_neurons = cstmd_config_pref['save_neurons']
        except KeyError:
            save_neurons = None

        self.cstmd = Cstmd1(global_dt=self.dt,
                            speed_up=cstmd_config_pref['cstmd_speed_up'],
                            estmd_dim=self.estmd.output_dimensions,
                            buffer_size=cstmd_buffer_size,
                            cuda_device=cuda_device,
                            debug_mode=cstmd_config_pref['debug_mode'],
                            sim_parameters=cstmd_config.config_section_map("CSTMD1_Simulator", overwrite),
                            morph_param=cstmd_config.config_section_map("Morphology_Homogenised_Top", overwrite),
                            run_id=self.run_id,
                            enable_spike_dump=cstmd_config_pref['enable_spike_dump'],
                            preload_path= preload,
                            save_bare_nc_path=save_neurons)

        print "=====Constructed CSTMD Module===== "

        if self.run_until < self.stage_STDP:
            self.end_time = time.time()
            print "Finished construction in %f seconds" % (self.end_time - self.start_time)
            return

        print "=====Attempting to build STDP Module====="
        self.stdp_sum = 0
        self.stdp_config = Configurer(stdp_config).config_section_map("STDP_Parameters")
        print "STDP parameters: "
        pprint.pprint(self.stdp_config)
        print
        self.stdp = Stdp(run_id=self.run_id,
                         num_afferents=self.stdp_config["num_afferents"],
                         num_neurons=self.stdp_config["num_neurons"],
                         training=self.stdp_config["training"],
                         load_weights=self.stdp_config["load_weights"],
                         weights_path =self.stdp_config["weights_path"],
                         a_plus=self.stdp_config["a_plus"],
                         a_ratio=self.stdp_config["a_ratio"],
                         t_plus=self.stdp_config["t_plus"],
                         t_minus=self.stdp_config["t_minus"],
                         alpha=self.stdp_config["alpha"],
                         theta=self.stdp_config["theta"],
                         historic_weights_path=self.stdp_config["historic_weights_path"])
        print "=====Constructed STDP Module===== "

        if self.run_until < self.stage_RL:
            self.end_time = time.time()
            print "Finished construction in %f seconds" % (self.end_time - self.start_time)
            return

        print "=====Attempting to build RL Module====="
        self.rl = RL(run_id = self.run_id,
                    topology=[len(self.stdp.neurons),4],
                    load_weights=rl_weights)
        print "RL topology: ", self.rl.topology
        print "=====Constructed RL Module====="

        print "====================FINISHED BUILDING DRAGONFLY BRAIN===================="
        self.end_time = time.time()
        print "Finished construction in %f seconds" % (self.end_time - self.start_time)
        self.start_time = None

    def check_if_need_to_return(self, stage, status=True, data=None, error=False):
        """
        Check if need to end step according to run_until and print outout data if debug
        """
        if not status:
            print "Error at step " + str(self.global_time) + " in module " + self.Names[stage]
            return True, True

        if data is not None and self.debug:
            print self.Names[stage] + " output: ", data

        if self.run_until <= stage:
            self.global_time += self.dt
            print "TIME UPDATED TO " + str(self.global_time) + "ms"
            return True, False

        return False, False

    def run(self, number_of_steps=1000, save_data=True):
        """
        Run several steps in a cycle
        """
        for i in xrange(number_of_steps):
            if not self.step():
                return False

        if save_data:
            self.data_dump(Environment=True, ESTMD=True, CSTMD=True, STDP=True, RL=True)



    def step(self):
        """
        Run a step of simulation. Runs through the step functions of all modules enabled before run_until module.
        """
        if not self.start_time:
            self.start_time = time.time()

        self.end_time = time.time()

        frame, reward = self.environment.step(self.rl_output)
        now = time.time()
        self.env_time += (now - self.end_time)
        self.end_time = now

        if self.pyplot:
            plt.imshow(frame, cmap='gray')
            plt.draw()

        ret, error = self.check_if_need_to_return(self.stage_Environment, data=reward)
        if ret or error:
            return False if error else True

        estmd_output = self.estmd.step(frame)
        now = time.time()
        self.estmd_time += (now - self.end_time)
        self.end_time = now
        ret, error = self.check_if_need_to_return(self.stage_ESTMD, data=('|' * len(estmd_output)))
        if ret or error:
            return False if error else True

        cstmd_status, cstmd_spikes = self.cstmd.step(self.dt, estmd_output)
        now = time.time()
        self.cstmd_time += (now - self.end_time)
        self.end_time = now
        ret, error = self.check_if_need_to_return(self.stage_CSTMD1, cstmd_status, cstmd_spikes)
        if ret or error:
            return False if error else True

        pattern_r = self.stdp.step(cstmd_spikes)
        now = time.time()
        self.stdp_time += (now - self.end_time)
        self.end_time = now
        ret, error = self.check_if_need_to_return(self.stage_STDP, data=pattern_r)
        if ret or error:
            return False if error else True

        self.rl_output = self.rl.step(pattern_r,reward)
        now = time.time()
        self.rl_time += (now - self.end_time)
        self.end_time = now
        ret, error = self.check_if_need_to_return(self.stage_RL, data=self.rl_output)
        if ret or error:
            return False if error else True

        self.global_time += self.dt
        print "TIME UPDATED TO " + str(self.global_time) + "ms"
        return True

    def data_dump(self, Environment=False, ESTMD=False, CSTMD=False, STDP=False, RL=False, directory=None, run_id_prefix=False):
        """
        Save output data from run from modules and time analysis
        """
        if not directory:
            if not os.path.isdir("DATA"):
                os.mkdir("DATA")

            day_folder = os.path.join("DATA", str(datetime.datetime.now().date()))

            if not os.path.isdir(day_folder):
                os.mkdir(day_folder)

            directory = os.path.join(os.path.abspath(os.getcwd()), (os.path.join(day_folder, str(self.run_id))))

        if not os.path.isdir(directory):
            os.mkdir(directory)

        np.savetxt(os.path.join(directory, 'time_analysis.dat'), self.time_analysis())

        if Environment and self.run_until >= self.stage_Environment:
            self.environment.get_video(directory=directory, run_id_prefix=run_id_prefix)

        if ESTMD and self.run_until >= self.stage_ESTMD:
            self.estmd.get_video(fps=self.environment.fps * 0.1, directory=directory, run_id_prefix=run_id_prefix)

        if CSTMD and self.run_until >= self.stage_CSTMD1:
            self.cstmd.save_spikes(directory=directory, run_id_prefix=run_id_prefix)
            # self.cstmd.save_graphs(directory=directory, run_id_prefix=run_id_prefix)
            # self.cstmd.save_morphology(directory=directory, run_id_prefix=run_id_prefix)
            self.cstmd.save_parameters(directory=directory, run_id_prefix=run_id_prefix)

        if STDP and self.run_until >= self.stage_STDP:
            self.stdp.save_output_spikes(directory=directory, run_id_prefix=run_id_prefix)
            self.stdp.save_current_weights(directory)

        if RL and self.run_until >= self.stage_RL:
            self.rl.save_weights(directory=directory, run_id_prefix=run_id_prefix)


    def time_analysis(self):
        """
        Output time analysis so far
        """
        loop_time = (self.end_time - self.start_time)
        return (loop_time, self.env_time, self.estmd_time, self.cstmd_time, self.stdp_time, self.rl_time)
