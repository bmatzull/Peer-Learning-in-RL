import numpy as np
from predefined_agents.mujoco.mujoco_peer import HalfCheetahExpert, HalfCheetahAdversarial

class HalfCheetahIntermittentAdversary(HalfCheetahAdversarial):
    def __init__(self, bad_advice_prob=0.5, **kwargs):
        super().__init__(**kwargs)
        self.bad_advice_prob = bad_advice_prob

    def _predict(self, obs, **kwargs):
        if np.random.random() < self.bad_advice_prob:
            return super()._predict(obs, **kwargs)
        else:
            return HalfCheetahExpert._predict(self, obs, **kwargs)
