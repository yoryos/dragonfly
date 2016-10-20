import abc


class VisualiserComponent(object):
    __metaclass__ = abc.ABCMeta


    @abc.abstractmethod
    def update(self):
        return True


    def save(self, name=None, dir=".", index=None, fmt=None):
        return 0

    @abc.abstractmethod
    def reset_to_start(self):
        pass

    @abc.abstractproperty
    def steps(self):
        return 0