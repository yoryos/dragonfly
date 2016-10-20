from abc import abstractmethod
from Helper.BrainModule import BrainModule

from Helper.Video_Tools import VideoGenerator


class GeneralEnvironment(BrainModule):
    """ Generic class for handling either Environment or WebCamHandler"""

    def __init__(self, dt, run_id):

        BrainModule.__init__(self, run_id)

        self.frames = []
        self.dt = dt
        self.fps = 1000.0 / dt

    @abstractmethod
    def step(self, rates=(0,0,0,0)):
        pass

    @staticmethod
    def green_filter(frame):
        green = frame[:, :, 1]
        green = 1.0 * green / 256.0
        return green

    def get_video(self, slow_down=0.1, directory=None, name="environment_output.avi", run_id_prefix=True):

        path = self.get_full_output_name(name, directory, run_id_prefix)
        vg = VideoGenerator()
        vg.generate_video(self.frames, fps=self.fps * slow_down, out_dir=path)
        print "Saved Environment video to " + path
