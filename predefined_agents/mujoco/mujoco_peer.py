import numpy as np
from stable_baselines3 import SAC
from manual_peer import ManualPeer

class HalfCheetahExpert(ManualPeer):
    ModelPath = "predefined_agents/mujoco/half_cheetah_expert"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.expert_model = SAC.load(self.MODEL_PATH)

    def _predict(self, obs, **kwargs):
        action, _ = self.expert_model.prediction(obs, deterministic=True)
        return action

class HalfCheetahAdversarial(HalfCheetahExpert):
    def _predict(self, obs, **kwargs):
        action = HalfCheetahExpert._predict(self, obs, **kwargs)
        return -action