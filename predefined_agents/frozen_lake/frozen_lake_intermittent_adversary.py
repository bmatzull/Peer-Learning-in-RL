import numpy as np
from predefined_agents.frozen_lake.frozen_lake_peer import FrozenLakeAdversarial, FrozenLakeExpert

class FrozenLakeIntermittentAdversary(FrozenLakeAdversarial):
    def __init__(self, bad_advice_prob=0.5, **kwargs):
        super().__init__(**kwargs)
        self.bad_advice_prob = bad_advice_prob

    def _predict(self, obs, **kwargs):
        if np.random.random() < self.bad_advice_prob:
            return super()._predict(obs, **kwargs)                  # bad advice
        else:
            return FrozenLakeExpert._predict(self, obs, **kwargs)   # good advice
