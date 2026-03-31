import heapq
import numpy as np


class Vertex:
    def __init__(self, pos):
        self.pos = pos 
        self.g = np.inf
        self.rhs = np.inf
        self.key = (np.inf, np.inf)


class DStar:

    def __init__(self, start, goal, costmap):
        self.start = tuple(start) 
        self.goal = tuple(goal)   
        self.costmap = np.array(costmap, dtype=np.int8)

        self.U = []
        self.V = {}
        self.K_m = 0.0
        self.K_old = (np.inf, np.inf)

        self._init_vertices()

    def _init_vertices(self):
        rows, cols = self.costmap.shape
        for i in range(rows):
            for j in range(cols):
                if self.costmap[i, j] == 0:
                    self.V[(i, j)] = Vertex((i, j))

    @staticmethod
    def _heuristic(a_pos, b_pos):
        return abs(a_pos[0] - b_pos[0]) + abs(a_pos[1] - b_pos[1])

    def _calc_key(self, u):
        m = min(u.g, u.rhs)
        return (m + self._heuristic(self.start, u.pos) + self.K_m, m)

    @staticmethod
    def _trans_cost(pos1, pos2):
        return (pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2

    def initialize(self):
        self.U = []
        self.K_m = 0.0

        for vert in self.V.values():
            vert.g = np.inf
            vert.rhs = np.inf
            vert.key = (np.inf, np.inf)

        if self.goal not in self.V:
            raise ValueError("Goal is outside free space or on obstacle")
        if self.start not in self.V:
            raise ValueError("Start is outside free space or on obstacle")

        goal_vert = self.V[self.goal]
        goal_vert.rhs = 0.0
        goal_vert.key = self._calc_key(goal_vert)
        heapq.heappush(self.U, (goal_vert.key, id(goal_vert), goal_vert))

    def get_neighbors(self, pos):
        neighbors = []
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nxt = (pos[0] + di, pos[1] + dj)
            if nxt in self.V:
                neighbors.append(nxt)
        return neighbors

    def update_vertex(self, u):
        if u.pos != self.goal:
            vals = []
            for neighbor_pos in self.get_neighbors(u.pos):
                neighbor = self.V[neighbor_pos]
                vals.append(neighbor.g + self._trans_cost(u.pos, neighbor_pos))
            u.rhs = min(vals) if vals else np.inf

        self.U = [(k, iid, v) for (k, iid, v) in self.U if v.pos != u.pos]
        heapq.heapify(self.U)

        if u.g != u.rhs:
            u.key = self._calc_key(u)
            heapq.heappush(self.U, (u.key, id(u), u))

    def compute_shortest_path(self):
        start_vert = self.V[self.start]

        while self.U and ((self.U[0][0] < self._calc_key(start_vert)) or (start_vert.rhs != start_vert.g)):
            self.K_old = self.U[0][0]
            u = heapq.heappop(self.U)[2]

            if self.K_old < self._calc_key(u):
                heapq.heappush(self.U, (self._calc_key(u), id(u), u))
                continue

            if u.g > u.rhs:
                u.g = u.rhs
                for neighbor_pos in self.get_neighbors(u.pos):
                    if neighbor_pos != self.goal:
                        self.update_vertex(self.V[neighbor_pos])
            else:
                u.g = np.inf
                self.update_vertex(u)
                for neighbor_pos in self.get_neighbors(u.pos):
                    if neighbor_pos != self.goal:
                        self.update_vertex(self.V[neighbor_pos])

    def plan(self, new_costmap=None, max_steps=10000):
        self.initialize()
        self.compute_shortest_path()

        path = [self.start]
        start_vert = self.V[self.start]
        goal_vert = self.V[self.goal]

        steps = 0
        while start_vert.pos != goal_vert.pos and steps < max_steps:
            steps += 1

            if start_vert.g == np.inf:
                return None

            best_pos = None
            best_val = np.inf
            for nb in self.get_neighbors(start_vert.pos):
                val = self._trans_cost(start_vert.pos, nb) + self.V[nb].g
                if val < best_val:
                    best_val = val
                    best_pos = nb

            if best_pos is None:
                return None

            last = self.start
            self.start = best_pos
            path.append(self.start)
            start_vert = self.V[self.start]

            if new_costmap is not None and not np.array_equal(new_costmap, self.costmap):
                self.costmap = np.array(new_costmap, dtype=np.int8)
                self.K_m += self._heuristic(last, self.start)
                for nb in self.get_neighbors(start_vert.pos):
                    self.update_vertex(self.V[nb])
                self.compute_shortest_path()

        return path
