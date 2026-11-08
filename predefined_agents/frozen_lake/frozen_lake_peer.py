import numpy as np
from collections import deque
from manual_peer import ManualPeer

class FrozenLakeExpert(ManualPeer):

    GRID_SIZE = 4
    HOLES = {5, 7, 11, 12}
    GOAL = 15

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.optimal_policy = self._compute_policy()

    def _compute_policy(self):
        policy = {}
        self.bfs_distance = {}
        visited = set()
        queue = deque()

        queue.append((self.GOAL, 0))
        visited.add(self.GOAL)
        self.bfs_distance[self.GOAL] = 0

        while queue:
            tile, dist = queue.popleft()
            row, col = tile // self.GRID_SIZE, tile % self.GRID_SIZE

            neighbors = [
                (row, col - 1, 2), # neighbor to the left
                (row - 1, col, 1), # neighbor above
                (row, col + 1, 0), # neighbor to the right
                (row + 1, col, 3) # neighbor below me
            ]

            for n_row, n_col, action in neighbors:
                n_tile = n_row * self.GRID_SIZE + n_col
                if 0 <= n_row < self.GRID_SIZE and 0 <= n_col < self.GRID_SIZE and n_tile not in self.HOLES and n_tile not in visited:
                    policy[n_tile] = action
                    self.bfs_distance[n_tile] = dist + 1
                    visited.add(n_tile)
                    queue.append((n_tile, dist + 1))
        return policy


    def _predict(self, obs, **kwargs):
        return self.optimal_policy[int(obs)]

class FrozenLakeAdversarial(FrozenLakeExpert):
    def _predict(self, obs, **kwargs):
        best = self.optimal_policy[int(obs)]
        return (best + 2) % 4