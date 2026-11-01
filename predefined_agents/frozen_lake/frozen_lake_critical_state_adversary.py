from predefined_agents.frozen_lake.frozen_lake_peer import FrozenLakeAdversarial, FrozenLakeExpert

class FrozenLakeCriticalStateAdversary(FrozenLakeAdversarial):
    def __init__(self, critical_distance=2, **kwargs):
        super().__init__(**kwargs)
        self.critical_distance = critical_distance

    def _predict(self, obs, **kwargs):
        bfs_dist = self._get_bfs_distance(obs)
        if bfs_dist <= self.critical_distance:
            return super()._predict(obs, **kwargs)                  # bad advice
        else:
            return FrozenLakeExpert._predict(self, obs, **kwargs)   # good advice

    def _get_bfs_distance(self, tile):
        return self.bfs_distance.get(tile, float('inf'))