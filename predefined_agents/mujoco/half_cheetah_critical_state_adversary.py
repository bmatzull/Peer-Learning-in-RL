from predefined_agents.mujoco.mujoco_peer import HalfCheetahExpert, HalfCheetahAdversarial

class HalfCheetahCriticalStateAdversary(HalfCheetahAdversarial):
    def __init__(self, angle_threshold=0.2, angvel_threshold=1.0,**kwargs):
        super().__init__(**kwargs)
        self.angle_threshold = angle_threshold
        self.angvel_threshold = angvel_threshold

    def _predict(self, obs, **kwargs):
        torso_angle = obs[2]
        torso_angvel = obs[11]
        is_critical = (
            abs(torso_angle) >= self.angle_threshold or
            abs(torso_angvel) >= self.angvel_threshold
        )
        if is_critical:
            return super()._predict(obs, **kwargs)                  # bad advice
        else:
            return HalfCheetahExpert._predict(self, obs, **kwargs)  # good advice