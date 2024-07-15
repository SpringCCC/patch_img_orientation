from abc import ABC, abstractmethod

class LossAbstractClass(ABC):

    @abstractmethod
    def calc_loss(self):
        pass