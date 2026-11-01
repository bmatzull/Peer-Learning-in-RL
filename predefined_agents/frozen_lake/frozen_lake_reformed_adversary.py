import numpy as np
from predefined_agents.frozen_lake.frozen_lake_peer import FrozenLakeAdversarial, FrozenLakeExpert

class FrozenLakeReformedAdversary(FrozenLakeAdversarial):
    def __init__(self, reform_steps=500000, **kwargs):
        super().__init__(**kwargs)
        self.reform_steps = reform_steps
        self.step_count = 0

    def _predict(self, obs, **kwargs):
        self.step_count += 1
        if self.step_count < self.reform_steps:
            return super()._predict(obs, **kwargs)                  # bad advice
        else:
            return FrozenLakeExpert._predict(self, obs, **kwargs)   # good advice