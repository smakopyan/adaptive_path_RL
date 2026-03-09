import numpy as np


def clip(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)


def wrap_angle(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


class DifferentialDriveModel:

    def dynamics(self, state, control):
        x, y, theta = state
        v, w = control

        dx = v * np.cos(theta)
        dy = v * np.sin(theta)
        dtheta = w

        return np.array([dx, dy, dtheta], dtype=float)


class Hamiltonian:
    def value(self, state, control, psi):
        x, y, theta = state
        v, w = control
        psi_x, psi_y, psi_theta = psi

        return (
            psi_x * v * np.cos(theta)
            + psi_y * v * np.sin(theta)
            + psi_theta * w
            + 1.0
        )

    def dH_dx(self, state, control, psi):

        x, y, theta = state
        v, w = control
        psi_x, psi_y, psi_theta = psi

        dHx = 0.0
        dHy = 0.0
        dHtheta = -psi_x * v * np.sin(theta) + psi_y * v * np.cos(theta)

        return np.array([dHx, dHy, dHtheta], dtype=float)


class PMP:

    def __init__(
        self,
        x,
        y,
        yaw,
        path,
        v_max=0.8,
        w_max=1.5,
        dt_internal=0.05,
        goal_tolerance=0.15,
        waypoint_tolerance=0.25,
        lookahead_dist=0.4,
        psi_theta_gain=1.0,
        theta_weight=1.0,
        use_reverse=False,
    ):
        self.model = DifferentialDriveModel()
        self.ham = Hamiltonian()

        self.x = np.array([x, y, yaw], dtype=float)

        self.path = np.array(path, dtype=float) if path is not None else np.zeros((0, 2), dtype=float)

        self.v_max = float(v_max)
        self.w_max = float(w_max)
        self.use_reverse = bool(use_reverse)

        self.dt_internal = float(dt_internal)
        self.goal_tolerance = float(goal_tolerance)
        self.waypoint_tolerance = float(waypoint_tolerance)
        self.lookahead_dist = float(lookahead_dist)

        self.psi_theta_gain = float(psi_theta_gain)
        self.theta_weight = float(theta_weight)

        self.psi = np.array([0.0, 0.0, 0.0], dtype=float)

        self.current_index = 0


    def set_path(self, path):
        self.path = np.array(path, dtype=float) if path is not None else np.zeros((0, 2), dtype=float)
        self.current_index = 0

    def nearest_index(self, x, y):
        if self.path is None or len(self.path) == 0:
            return None

        d = np.linalg.norm(self.path - np.array([x, y]), axis=1)
        return int(np.argmin(d))

    def nearest_point(self, x, y):
        idx = self.nearest_index(x, y)
        if idx is None:
            return np.array([x, y], dtype=float)
        return self.path[idx]

    def path_heading(self, idx):
        if self.path is None or len(self.path) < 2:
            return 0.0

        idx = int(np.clip(idx, 0, len(self.path) - 1))

        if idx >= len(self.path) - 1:
            p0 = self.path[-2]
            p1 = self.path[-1]
        else:
            p0 = self.path[idx]
            p1 = self.path[idx + 1]

        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        return np.arctan2(dy, dx)

    def lookahead_point(self, x, y, lookahead_dist=None):
        if self.path is None or len(self.path) == 0:
            return np.array([x, y], dtype=float), 0

        if lookahead_dist is None:
            lookahead_dist = self.lookahead_dist

        idx = self.nearest_index(x, y)
        if idx is None:
            return np.array([x, y], dtype=float), 0

        accum = 0.0
        prev = self.path[idx]

        for j in range(idx + 1, len(self.path)):
            cur = self.path[j]
            seg = np.linalg.norm(cur - prev)
            accum += seg
            if accum >= lookahead_dist:
                return cur.copy(), j
            prev = cur

        return self.path[-1].copy(), len(self.path) - 1

    def is_goal_reached(self, x, y):
        if self.path is None or len(self.path) == 0:
            return True
        goal = self.path[-1]
        return np.linalg.norm(np.array([x, y]) - goal) < self.goal_tolerance


    def reset_costate_from_target(self, state, target):

        x, y, theta = state
        tx, ty = target

        dx = tx - x
        dy = ty - y
        dist = np.hypot(dx, dy)

        if dist < 1e-8:
            dir_x, dir_y = np.cos(theta), np.sin(theta)
        else:
            dir_x, dir_y = dx / dist, dy / dist

        target_yaw = np.arctan2(dy, dx)
        yaw_err = wrap_angle(target_yaw - theta)

        self.psi[0] = dir_x
        self.psi[1] = dir_y
        self.psi[2] = self.psi_theta_gain * yaw_err

    def costate_dynamics(self, state, control, psi):
        dHdx = self.ham.dH_dx(state, control, psi)
        return -dHdx

    def rk4_costate_step(self, state, control, psi, dt):

        k1 = self.costate_dynamics(state, control, psi)
        k2 = self.costate_dynamics(state, control, psi + 0.5 * dt * k1)
        k3 = self.costate_dynamics(state, control, psi + 0.5 * dt * k2)
        k4 = self.costate_dynamics(state, control, psi + dt * k3)

        new_psi = psi + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return new_psi

    def argmax_hamiltonian(self, state, psi):
        x, y, theta = state
        psi_x, psi_y, psi_theta = psi

        s_v = psi_x * np.cos(theta) + psi_y * np.sin(theta)
        s_w = psi_theta

        if self.use_reverse:
            v = self.v_max * np.sign(s_v)
        else:
            v = self.v_max if s_v >= 0.0 else 0.0

        if s_w > 1e-6:
            w = self.w_max
        elif s_w < -1e-6:
            w = -self.w_max
        else:
            w = 0.0

        if self.path is not None and len(self.path) > 0:
            goal = self.path[-1]
            dist_goal = np.linalg.norm(np.array([x, y]) - goal)
            v_limit = min(self.v_max, 0.5 * dist_goal + 0.05)
            if self.use_reverse:
                v = clip(v, -v_limit, v_limit)
            else:
                v = clip(v, 0.0, v_limit)

        return np.array([float(v), float(w)], dtype=float)


    def rk4_state_step(self, state, control, dt):

        k1 = self.model.dynamics(state, control)
        k2 = self.model.dynamics(state + 0.5 * dt * k1, control)
        k3 = self.model.dynamics(state + 0.5 * dt * k2, control)
        k4 = self.model.dynamics(state + dt * k3, control)

        new_state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        new_state[2] = wrap_angle(new_state[2])
        return new_state

    def control(self, x, y, yaw):

        state = np.array([x, y, yaw], dtype=float)

        if self.path is None or len(self.path) < 2:
            self.x = state
            return 0.0, 0.0

        if self.is_goal_reached(x, y):
            self.x = state
            return 0.0, 0.0

        target, target_idx = self.lookahead_point(x, y)
        self.current_index = target_idx

        self.reset_costate_from_target(state, target)

        u = self.argmax_hamiltonian(state, self.psi)

        self.x = state
        return float(u[0]), float(u[1])

    def step(self, dt=None):
        if dt is None:
            dt = self.dt_internal

        x, y, yaw = self.x
        v, w = self.control(x, y, yaw)
        u = np.array([v, w], dtype=float)

        new_state = self.rk4_state_step(self.x, u, dt)

        new_psi = self.rk4_costate_step(self.x, u, self.psi, dt)

        new_state[2] = wrap_angle(new_state[2])

        self.x = new_state
        self.psi = new_psi

        return self.x.copy(), u.copy(), self.psi.copy()


    def rollout(self, T=5.0, dt=0.05):
        n = int(np.ceil(T / dt)) + 1

        t_arr = np.linspace(0.0, T, n)
        traj = np.zeros((n, 3), dtype=float)
        u_arr = np.zeros((n, 2), dtype=float)
        psi_arr = np.zeros((n, 3), dtype=float)

        state_backup = self.x.copy()
        psi_backup = self.psi.copy()
        idx_backup = self.current_index

        traj[0] = self.x
        psi_arr[0] = self.psi

        for k in range(n - 1):
            x, u, psi = self.step(dt)
            traj[k + 1] = x
            u_arr[k] = u
            psi_arr[k + 1] = psi

            if self.is_goal_reached(x[0], x[1]):
                for j in range(k + 1, n - 1):
                    u_arr[j] = np.array([0.0, 0.0], dtype=float)
                    traj[j + 1] = traj[k + 1]
                    psi_arr[j + 1] = psi_arr[k + 1]
                break

        u_arr[-1] = np.array([0.0, 0.0], dtype=float)

        self.x = state_backup
        self.psi = psi_backup
        self.current_index = idx_backup

        return t_arr, traj, u_arr, psi_arr