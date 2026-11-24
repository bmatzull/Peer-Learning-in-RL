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

class RoomIntermittentAdversary5(RoomIntermittentAdversary):
    def __init__(self, **kwargs):
        super().__init__(attack_frequency=0.05, **kwargs)


class RoomIntermittentAdversary25(RoomIntermittentAdversary):
    def __init__(self, **kwargs):
        super().__init__(attack_frequency=0.25, **kwargs)


class RoomIntermittentAdversary50(RoomIntermittentAdversary):
    def __init__(self, **kwargs):
        super().__init__(attack_frequency=0.5, **kwargs)


class RoomIntermittentAdversary75(RoomIntermittentAdversary):
    def __init__(self, **kwargs):
        super().__init__(attack_frequency=0.75, **kwargs)
