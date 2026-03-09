import numpy as np
import heapq

class Vertex:
    def __init__(self, pos):
        self.pos = pos
        self.g = np.inf
        self.rhs = np.inf
        self.key = [np.inf, np.inf]
    
    def h(self, goal):
        return (abs(self.pos[0] - goal.pos[0]) + abs(self.pos[1] - goal.pos[1]))


class DStar:
    def __init__(self, start, goal, costmap):
        self.start = start
        self.goal = goal
        self.costmap = costmap
        self.U = []
        self.V = {}
        self.K_m = None
        self.K_old = None
        self.init_vertices()

    def heuristic(self, a_pos, b_pos):
        return abs(a_pos[0] - b_pos[0]) + abs(a_pos[1] - b_pos[1])
    
    def calc_key(self, u):
        m = min(u.g, u.rhs)
        return (m + self.heuristic(self.start, u.pos) + self.K_m, m)
    
    def trans_cost(self, pos1, pos2):
        return ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def init_vertices(self):
        for i in range(self.costmap.shape[0]):
            for j in range(self.costmap.shape[1]):
                if self.costmap[i, j] == 0:
                    self.V[(i, j)] = Vertex((i, j))

    def initialize(self):
        self.U = []
        self.K_m = 0
        for pos, vert in self.V.items():
            vert.rhs = vert.g = np.inf
        try:
            goal_vert = self.V[self.goal]
            goal_vert.rhs = 0
            goal_vert.key = self.calc_key(goal_vert)
            heapq.heappush(self.U, (goal_vert.key, id(goal_vert), goal_vert))

        except Exception as e:
            print('The goal is an obstacle! Please change the goal')

        # self.U.append([self.goal, self.calc_key(self.goal)])

    def update_vertex(self, u):
        if u.pos != self.goal:
            neighbors = self.get_neighbors(u.pos)
            rhss = []
            for neighbor_pos in neighbors:
                neighbor = self.V[neighbor_pos]
                rhss.append(neighbor.g + self.trans_cost(u.pos, neighbor_pos))
            u.rhs = min(rhss) if rhss else np.inf

        self.U = [(k, id_, v,) for (k, id_, v,) in self.U if v.pos != u.pos]
        heapq.heapify(self.U)
        
        if u.g != u.rhs:
            u.key = self.calc_key(u)
            heapq.heappush(self.U, (u.key, id(u), u))

    def compute_shortest_path(self):
        start_vert = self.V[self.start]
        while self.U and ((self.U[0][0] < self.calc_key(start_vert)) or start_vert.rhs != start_vert.g):
            self.K_old = self.U[0][0]
            u = heapq.heappop(self.U)[2]
            if self.K_old < self.calc_key(u):
                heapq.heappush(self.U, (self.calc_key(u), id(u), u))
                continue
            if u.g > u.rhs:
                u.g = u.rhs
                for neighbor_pos in self.get_neighbors(u.pos):
                    if neighbor_pos != self.goal:
                        neighbor = self.V[neighbor_pos]
                        self.update_vertex(neighbor)
            else:
                u.g = np.inf
                self.update_vertex(u)
                for neighbor_pos in self.get_neighbors(u.pos):
                    if neighbor_pos != self.goal:
                        neighbor = self.V[neighbor_pos]
                        self.update_vertex(neighbor)

    def get_neighbors(self, pos):
        neighbors = []
        for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
            np_ = (pos[0] + di, pos[1] + dj)
            if np_ in self.V:
                neighbors.append(np_)
        return neighbors
    
    def plan(self, new_costmap=None, max_steps=10000):
        self.initialize()
        self.compute_shortest_path()

        path = [self.start]
        start_vert = self.V[self.start]
        goal_vert  = self.V[self.goal]

        steps = 0
        while start_vert.pos != goal_vert.pos and steps < max_steps:
            steps += 1
            if start_vert.g == np.inf:
                return None

            best_pos = None
            best_val = np.inf
            for nb in self.get_neighbors(start_vert.pos):
                val = self.trans_cost(start_vert.pos, nb) + self.V[nb].g
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
                self.costmap = new_costmap
                self.K_m += self.heuristic(last, self.start)
                for nb in self.get_neighbors(start_vert.pos):
                    self.update_vertex(self.V[nb])
                self.compute_shortest_path()

        return path