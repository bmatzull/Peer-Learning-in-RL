import numpy as np
from predefined_agents import RoomExpert, RoomAdversarial


class RoomIntermittentAdversary(RoomAdversarial):
    def __init__(self,attack_frequency=0.15, **kwargs):
        super().__init__(**kwargs)
        self.attack_frequency = attack_frequency

    def _predict(self, obs, **kwargs):
        if np.random.random() < self.attack_frequency:
            return super()._predict(obs, **kwargs)  # bad advice
        else:
            return RoomExpert._predict(self, obs, **kwargs)  # good advice
