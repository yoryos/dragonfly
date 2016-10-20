import numpy as np
import os


class BrainModule(object):
    """
    Generic class which the core modules of DragonflyBrain can inherit from
    """

    prefix = "DATA_"

    def __init__(self, run_id):

        self.run_id = run_id
        self.output_dir = None
        self.reset_default_output_directory()

    def save_dictionary(self, dict, directory=None, name="dict.dat", run_id_prefix=False):

        path = self.get_full_output_name(name, directory, run_id_prefix)
        with open(path, 'w') as file:
            file.writelines('{}:{}\n'.format(k,v) for k, v in dict.items())

        print "Saved " + name + " to " + path

    def save_numpy_array(self, ndarray, directory=None, name="out.dat", npz=False, fmt='%.18e', run_id_prefix=False, transpose=False):

        if ndarray is None:
            print "Cannot data to " + name + " data array is None"
            return False

        path = self.get_full_output_name(name, directory, run_id_prefix)

        if transpose:
            saved_ndarray = ndarray.transpose()
        else:
            saved_ndarray = ndarray

        if npz:
            path += ".npy"
            np.save(path, saved_ndarray)
        else:
            np.savetxt(path, saved_ndarray, fmt)

        print "Saved " + name + " to " + path
        return True

    def load_numpy_array(self, file):
        return(np.loadtxt(file))

    def set_output_directory(self, directory):

        self.output_dir = directory

    def reset_default_output_directory(self):

        self.output_dir = self.prefix + str(self.run_id)

    def get_output_directory(self, dir = None):

        directory = self.output_dir
        if dir is not None:
            directory = os.path.join(directory, dir)

        if not os.path.isdir(directory):
            os.makedirs(directory)

        return directory

    def get_full_output_name(self, name, directory = None, run_id_prefix = False):

        if directory is None:
            directory = self.get_output_directory()

        if run_id_prefix:
            path = os.path.join(directory, self.run_id + "_" + name)
        else:
            path = os.path.join(directory, name)

        return path
