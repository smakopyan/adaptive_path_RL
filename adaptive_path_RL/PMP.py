import numpy as np

def clip(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)

class PMP:
    def __init__(self, x, y, yaw, path, alpha = 1.0):
        self.x = np.array([x, y, yaw, 0., 0., 0.])
        self.path = path
        self.alpha = alpha
        self.rv = 1.0
        self.rw = 0.5
        self.v_min, self.v_max = (0.2, 1.0)
        self.w_max = 2.


    def nearest_point(self, x, y):
        idx = np.argmin(np.linalg.norm(self.path - np.array([x, y]), axis = 1))
        return self.path[idx]


    def control(self, x, y, yaw, lx, ly, lyaw):
        v = -(lx * np.cos(yaw) + ly * np.sin(yaw)) / self.rv
        w = -lyaw / self.rw

        v = clip(v, self.v_min, self.v_max)
        w = clip(w, -self.w_max, self.w_max)
        return float(v), float(w)
    

    def dynamics(self, state):
        x, y, yaw, lx, ly, lyaw = state
        xr, yr = self.nearest_point(x, y)

        ex = x - xr
        ey = y - yr

        v, w =self.control(x, y, yaw, lx, ly, lyaw)
        
        dx = v * np.cos(yaw)
        dy = v * np.sin(yaw)
        dyaw = w

        dlx = -self.alpha * ex
        dly = -self.alpha * ey
        dlyaw = v * (lx * np.sin(yaw) - ly * np.cos(yaw))
        return np.array([dx, dy, dyaw, dlx, dly, dlyaw])
    


    def rk4_step(self, state, dt):
        k1 = self.dynamics(state)
        k2 = self.dynamics(state + 0.5 * dt * k1)
        k3 = self.dynamics(state + 0.5 * dt * k2)
        k4 = self.dynamics(state + dt * k3)
        return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


    def rollout(self, T=5.0, dt=0.02):
        
        n = int(np.ceil(T / dt)) + 1
        traj = np.zeros((n, 6), dtype=float)
        u = np.zeros((n, 2), dtype=float)
        t = np.linspace(0.0, T, n)

        state = self.x.copy()
        traj[0] = state

        for k in range(n - 1):
            x, y, yaw, lx, ly, lyaw = state
            v, w = self.control(x, y, yaw, lx, ly, lyaw)
            u[k] = [v, w]

            state = self.rk4_step(state, dt)
            traj[k + 1] = state

        # last control
        x, y, yaw, lx, ly, lyaw = traj[-1]
        u[-1] = self.control(x, y, yaw, lx, ly, lyaw)

        return t, traj, u