from predefined_agents import RoomExpert, RoomAdversarial


class RoomReformedAdversary(RoomAdversarial):
    def __init__(self, total_steps=1000000, **kwargs):
        super().__init__(**kwargs)
        self.total_steps = total_steps
        self.current_step = 0

    def _predict(self, obs, **kwargs):
        self.current_step += 1
        if self.current_step < self.total_steps / 2:
            return super()._predict(obs, **kwargs)          # bad advice
        else:
            return RoomExpert._predict(self, obs, **kwargs) # good advice
