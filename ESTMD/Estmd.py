'''
ESTMD

Tool to isolate targets from video. You can generate appropriate videos
using module Target_animation.

__author__: Dragonfly Project 2016 - Imperial College London
            ({anc15, cps15, dk2015, gk513, lm1015,zl4215}@imperial.ac.uk)

CITE
'''

import os
from copy import deepcopy

import cv2
import numpy as np
from scipy import signal
from Helper.BrainModule import BrainModule

class Estmd(BrainModule):
    """
    With this class we set parameters and extract targets from movie.

    Main engine of the class is the process_frame method, to store a frame and
    then detect movement in the frame compared to store frames in frame_history.
    This is called in its step method which then returns all nonzero values.
    """

    @staticmethod
    def rtc_exp(t_s, x):
        x[x > 0] = 1 / x[x > 0]
        x[x > 0] = np.exp(-t_s * x[x > 0])
        return x

    def __init__(self,
                 run_id,
                 input_dimensions=(640, 480),
                 preprocess_resize=True,
                 resize_factor=0.1,
                 threshold=0.1,
                 time_step=0.001,
                 LMC_rec_depth=12,
                 H_filter=None,
                 b=None,
                 a=None,
                 CSKernel=None,
                 b1=None,
                 a1=None,
                 gain=50

                 ):

        BrainModule.__init__(self, run_id)

        # Set H_filter.
        if H_filter is None:
            self.H_filter = np.array([[-1, -1, -1, -1, -1],
                                      [-1, 0, 0, 0, -1],
                                      [-1, 0, 2, 0, -1],
                                      [-1, 0, 0, 0, -1],
                                      [-1, -1, -1, -1, -1]])
        else:
            self.H_filter = H_filter

        # Set b.
        if b is None:
            self.b = [0.0, 0.00006, -0.00076, 0.0044,
                      -0.016, 0.043, -0.057, 0.1789, -0.1524]
        else:
            self.b = b

        # Set a.
        if a is None:
            self.a = [1.0, -4.333, 8.685, -10.71, 9.0, -5.306,
                      2.145, -0.5418, 0.0651]
        else:
            self.a = a

        # Set CSKernel.
        if CSKernel is None:
            self.CSKernel = np.array([[-1.0 / 9.0, -1.0 / 9.0, -1.0 / 9.0],
                                      [-1.0 / 9.0, 8.0 / 9.0, -1.0 / 9.0],
                                      [-1.0 / 9.0, -1.0 / 9.0, -1.0 / 9.0]])
        else:
            self.CSKernel = CSKernel

        # Set b1.
        if b1 is None:
            self.b1 = [1.0, 1.0]
        else:
            self.b1 = b1

        # Set a1.
        if a1 is None:
            self.a1 = [51.0, -49.0]
        else:
            self.a1 = a1

        self.pre_resize = preprocess_resize

        self.resize_factor = resize_factor

        self.input_dimensions = input_dimensions

        self.output_dimensions = (int(self.input_dimensions[0] * self.resize_factor),
                                  int(self.input_dimensions[1] * self.resize_factor))

        self.frame_history = []

        self.LMC_rec_depth = LMC_rec_depth

        self.dt = self.t = self.T0 = time_step

        self.threshold = threshold

        self.gain = gain

        self.result_values = []

    def get_video(self, fps, directory=None, name="estmd_output.avi", run_id_prefix=True, cod="MJPG"):
        """
        Returns a video of processed frames after processing through step

        Args:
            fps (): Frame rate
            cod (): Codec of output
            run_id_prefix (): Prefix with run_id?
            name (): Output file name
            directory (): Output file directory
        """
        path = self.get_full_output_name(name, directory, run_id_prefix)

        codec = cv2.cv.CV_FOURCC(cod[0], cod[1], cod[2], cod[3])
        video = cv2.VideoWriter(path, codec, fps, self.output_dimensions, isColor=0)

        print "ESTMD outputting at: ", self.output_dimensions

        for values in self.result_values:
            frame = np.zeros(self.output_dimensions[::-1])
            for v in values:
                ycord, xcord, pixel = v
                frame[ycord, xcord] = pixel
            frame = (frame * 255.0).astype('u1')
            video.write(frame)

        video.release()
        cv2.destroyAllWindows()
        print "Saved ESTMD output video to " + path

        return

    def process_frame(self, downsize):
        """
        The engine of the class.

        Applies concepts from paper:
        'Discrete Implementation of Biologically Inspired Image Processing for
         Target Detection' by K. H., S. W., B. C. and D. C. from
        The University of Adelaide, Australia.
        """
        # if (not hasattr(downsize,'shape')) and (not hasattr(downsize,'len')):
        #     downsize = np.array(downsize)

        if type(downsize) != np.ndarray:
            raise TypeError

        if not downsize.any():
            raise ValueError

        if self.pre_resize:
            downsize = cv2.resize(downsize, (0, 0), fx=self.resize_factor, fy=self.resize_factor)

        self.frame_history.append(downsize)

        # Remove no longer needed frames from memory
        self.frame_history = self.frame_history[-(self.LMC_rec_depth):]
        downsize = signal.lfilter(self.b, self.a, self.frame_history, axis=0)[-1]

        # Center surround antagonism kernel applied.

        downsize = cv2.filter2D(downsize, -1, self.CSKernel)

        # RTC filter.
        u_pos = deepcopy(downsize)
        u_neg = deepcopy(downsize)
        u_pos[u_pos < 0] = 0
        u_neg[u_neg > 0] = 0
        u_neg = -u_neg

        # On first step, instead of computing just save the images.
        if self.t == self.T0:
            self.v_pos_prev = deepcopy(u_pos)
            self.v_neg_prev = deepcopy(u_neg)
            self.u_pos_prev = deepcopy(u_pos)
            self.u_neg_prev = deepcopy(u_neg)

        # Do everything for pos == ON.
        tau_pos = u_pos - self.u_pos_prev
        tau_pos[tau_pos >= 0] = 0.001
        tau_pos[tau_pos < 0] = 0.1
        mult_pos = self.rtc_exp(self.dt, tau_pos)
        v_pos = -(mult_pos - 1) * u_pos + mult_pos * self.v_pos_prev
        self.v_pos_prev = deepcopy(v_pos)

        # Do everything for neg == OFF.
        tau_neg = u_neg - self.u_neg_prev
        tau_neg[tau_neg >= 0] = 0.001
        tau_neg[tau_neg < 0] = 0.1
        mult_neg = self.rtc_exp(self.dt, tau_neg)
        v_neg = -(mult_neg - 1) * u_neg + mult_neg * self.v_neg_prev
        self.v_neg_prev = deepcopy(v_neg)

        # keep track of previous u.
        self.u_pos_prev = deepcopy(u_pos)
        self.u_neg_prev = deepcopy(u_neg)

        # Subtract v from u to give the output of each channel.
        out_pos = u_pos - v_pos
        out_neg = u_neg - v_neg

        # Now apply yet another filter to both parts.
        out_pos = cv2.filter2D(out_pos, -1, self.H_filter)
        out_neg = cv2.filter2D(out_neg, -1, self.H_filter)
        out_pos[out_pos < 0] = 0
        out_neg[out_neg < 0] = 0

        if self.t == self.T0:
            self.out_neg_prev = deepcopy(out_neg)

        # Delay off channel.
        out_neg = signal.lfilter(self.b1, self.a1, [self.out_neg_prev, out_neg], axis=0)[-1]
        self.out_neg_prev = out_neg
        downsize = out_neg * out_pos

        # Show image.
        downsize *= self.gain
        downsize = np.tanh(downsize)

        # Threshold.
        downsize[downsize < self.threshold] = 0

        if not self.pre_resize:
            downsize = cv2.resize(downsize, (0, 0), fx=self.resize_factor, fy=self.resize_factor)

        self.t += self.dt

        return downsize

    def step(self, frame):
        """
        Process a given frame and return all nonzero values
        """
        result = []
        frame = self.process_frame(frame)
        ycords, xcords = frame.nonzero()
        for i in xrange(len(ycords)):
            result.append((ycords[i], xcords[i], frame[ycords[i], xcords[i]]))
        self.result_values.append(result)
        return result
