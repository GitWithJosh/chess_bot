from abc import ABC, abstractmethod

class EngineOpponent(ABC):

    @abstractmethod
    def choose_move(self):
        pass