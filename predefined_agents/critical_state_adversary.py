import numpy as np
from predefined_agents import RoomExpert, RoomAdversarial


class RoomCriticalStateAdversary(RoomAdversarial):
    def __init__(self,critical_distance=0.3, **kwargs):
        super().__init__(**kwargs)
        self.critical_distance = critical_distance


    def _predict(self, obs, **kwargs):
        pos = obs[:len(obs) // 2]
        goal = obs[len(obs) // 2:]
        is_critical = np.sum(np.abs(pos - goal)) <= self.critical_distance
        if is_critical:
            return super()._predict(obs, **kwargs)  # bad advice
        else:
            return RoomExpert._predict(self, obs, **kwargs)  # good advice
