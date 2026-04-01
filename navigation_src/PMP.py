import numpy as np


def clip(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)


def wrap_angle(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


class DifferentialDriveModel:
    @staticmethod
    def dynamics(state, control):
        x, y, theta = state
        v, w = control
        dx = v * np.cos(theta)
        dy = v * np.sin(theta)
        dtheta = w
        return np.array([dx, dy, dtheta], dtype=float)


class Hamiltonian:
    @staticmethod
    def dH_dx(state, control, psi):
        _, _, theta = state
        v, _ = control
        psi_x, psi_y, _ = psi
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
        v_max=0.35,
        w_max=1.0,
        dt_internal=0.05,
        goal_tolerance=0.15,
        lookahead_dist=0.35,
        psi_theta_gain=1.3,
        use_reverse=False,
    ):
        self.model = DifferentialDriveModel()
        self.ham = Hamiltonian()

        self.x = np.array([x, y, yaw], dtype=float)
        self.psi = np.array([0.0, 0.0, 0.0], dtype=float)

        self.path = np.array(path, dtype=float) if path is not None else np.zeros((0, 2), dtype=float)

        self.v_max = float(v_max)
        self.w_max = float(w_max)
        self.dt_internal = float(dt_internal)
        self.goal_tolerance = float(goal_tolerance)
        self.lookahead_dist = float(lookahead_dist)
        self.psi_theta_gain = float(psi_theta_gain)
        self.use_reverse = bool(use_reverse)

        self.current_index = 0

    def set_path(self, path):
        self.path = np.array(path, dtype=float) if path is not None else np.zeros((0, 2), dtype=float)
        self.current_index = 0

    def nearest_index(self, x, y):
        if self.path is None or len(self.path) == 0:
            return None
        d = np.linalg.norm(self.path - np.array([x, y]), axis=1)
        return int(np.argmin(d))

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
            accum += np.linalg.norm(cur - prev)
            if accum >= lookahead_dist:
                return cur.copy(), j
            prev = cur

        return self.path[-1].copy(), len(self.path) - 1

    def is_goal_reached(self, x, y):
        if self.path is None or len(self.path) == 0:
            return True
        return np.linalg.norm(np.array([x, y]) - self.path[-1]) < self.goal_tolerance

    def reset_costate_from_target(self, state, target):
        x, y, theta = state
        tx, ty = target

        dx = tx - x
        dy = ty - y
        dist = np.hypot(dx, dy)

        if dist < 1e-8:
            # Если уже очень близко к цели, используем текущее направление
            dir_x, dir_y = np.cos(theta), np.sin(theta)
        else:
            dir_x, dir_y = dx / dist, dy / dist

        target_yaw = np.arctan2(dy, dx)
        yaw_err = wrap_angle(target_yaw - theta)


        self.psi[0] = dir_x
        self.psi[1] = dir_y

        self.psi[2] = self.psi_theta_gain * np.tanh(yaw_err)

    def costate_dynamics(self, state, control, psi):
        return -self.ham.dH_dx(state, control, psi)

    def rk4_costate_step(self, state, control, psi, dt):
        k1 = self.costate_dynamics(state, control, psi)
        k2 = self.costate_dynamics(state, control, psi + 0.5 * dt * k1)
        k3 = self.costate_dynamics(state, control, psi + 0.5 * dt * k2)
        k4 = self.costate_dynamics(state, control, psi + dt * k3)
        return psi + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def argmax_hamiltonian(self, state, psi):
        _, _, theta = state
        psi_x, psi_y, psi_theta = psi

        s_v = psi_x * np.cos(theta) + psi_y * np.sin(theta)
        s_w = psi_theta
        theta_weight = getattr(self, "theta_weight", self.psi_theta_gain)
        
        if self.use_reverse:
            v = self.v_max * np.sign(s_v)
        else:
            v = self.v_max if s_v >= 0.0 else 0.0

        w = clip(theta_weight * s_w, -self.w_max, self.w_max)

        if self.path is not None and len(self.path) > 0:
            goal = self.path[-1]
            dist_goal = np.linalg.norm(state[:2] - goal)
            v_limit = min(self.v_max, max(0.1, 0.5 * dist_goal + 0.05))
            v = clip(v, -v_limit if self.use_reverse else 0.0, v_limit)

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

        if self.path is None or len(self.path) < 2 or self.is_goal_reached(x, y):
            self.x = state
            return 0.0, 0.0

        target, target_idx = self.lookahead_point(x, y)
        self.current_index = target_idx
        self.reset_costate_from_target(state, target)
        u = self.argmax_hamiltonian(state, self.psi)
        target_yaw = np.arctan2(target[1] - y, target[0] - x)
        yaw_err = wrap_angle(target_yaw - yaw)

        if not self.use_reverse:
            heading_factor = max(0.1, float(np.cos(yaw_err)))  # min 0.1, max 1.0
            u[0] = u[0] * heading_factor
        
        self.x = state
        return float(u[0]), float(u[1])
