from abc import *

__all__ = ["Dictable"]

class Dictable(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def _as_dict(self):
        return {}

    @classmethod
    @abstractmethod
    def _from_dict(self):
        raise NotImplementedError

    # TODO: validation method?

