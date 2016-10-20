import os
import numpy as np
import cv2


class VideoGenerator(object):
    """Class for creating and loading videos"""
    def __init__(self):
        pass

    @staticmethod
    def generate_video(frames, fps, out_dir="out.avi", cod="MJPG"):
        img = frames[0]
        height, width, depth = img.shape
        codec = cv2.cv.CV_FOURCC(cod[0], cod[1], cod[2], cod[3])
        video = cv2.VideoWriter(out_dir, codec, fps, (width, height), isColor=True)

        for frame in frames:

            video.write(frame.astype('u1'))

        video.release()
        cv2.destroyAllWindows()

        return


class VideoToFrameConverter(object):
    video = False

    def __init__(self):
        self.video = False

    def open_video(self, movie_dir):
        """
        This method sets movie that we'll try to modify.
        Args:
            movie_dir: Directory of input we want to modify.
        """

        if not os.path.exists(movie_dir):
            raise NameError

        self.video = cv2.VideoCapture(movie_dir)

    def get_frame_count(self):
        if not self.video:
            raise NameError

        return int(self.video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

    def reset_to_start(self):
        if not self.video:
            raise NameError

        self.video.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, 0)

    def get_frame(self, color=False, stop_begin_stupid=True):
        if not self.video:
            raise NameError

        try:
            ret, frame = self.video.read()
            # Split into basic colours for green component
            # blue, green, red = cv2.split(frame)
        except:
            # No more frames
            return False, []
        if not ret:
            return False, []

        if not color:
            frame = frame[:, :, 1]

        if color and stop_begin_stupid:
            temp = np.copy(frame[:, :, 0])
            frame[:, :, 0] = frame[:, :, 2]
            frame[:, :, 2] = temp

        return True, frame.astype(float)

    def get_fps(self):
        if not self.video:
            raise NameError

        return self.video.get(cv2.cv.CV_CAP_PROP_FPS)

    def get_frames(self, total_frames=10, color=False, stop_begin_stupid=True):
        frames = []

        if not self.video:
            raise NameError

        for i in xrange(total_frames):
            s,frame = self.get_frame(color=color, stop_begin_stupid=stop_begin_stupid)
            if not s:
                if len(frames) == 0:
                    print "Failed to get frame"
                    return False, []
                else:
                    break
            frames.append(frame)

        return True, frames

    def get_all_frames(self):
        frames = []

        if not self.video:
            raise NameError

        while (True):
            s,frame = self.get_frame()
            if not s:
                break
            frames.append(frame)

        return frames
