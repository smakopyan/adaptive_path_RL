import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover
    import gym
    from gym import spaces

import rclpy
from geometry_msgs.msg import Point, Twist
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty
from visualization_msgs.msg import Marker

from navigation_src.D_star import DStar
from navigation_src.PMP import PMP

EXPANSION_SIZE = 20


def costmap(data, width, height):
    grid = np.array(data, dtype=np.int8).reshape(height, width)
    obstacles_mask = np.where(grid == 100, 255, 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (EXPANSION_SIZE, EXPANSION_SIZE))
    dilated_obstacles = cv2.dilate(obstacles_mask, kernel)
    result = np.where(dilated_obstacles == 255, 100, grid)
    result[grid == -1] = -1
    return result


@dataclass
class EnvConfig:
    dt: float = 0.1
    max_steps: int = 500
    goal_tolerance: float = 0.20
    linear_limit: float = 0.26
    angular_limit: float = 1.2
    residual_v_limit: float = 0.10
    residual_w_limit: float = 0.5
    obstacle_stop_dist: float = 0.21
    replan_period: int = 10

    slip_lin_max: float = 0.25
    slip_ang_max: float = 0.20
    wind_v_max: float = 0.03
    wind_w_max: float = 0.15
    battery_min: float = 0.75
    battery_max: float = 1.00


@dataclass
class RosConfig:
    namespace: str = ""
    map_topic: str = "/map"
    odom_topic: str = "/odom"
    scan_topic: str = "/scan"
    cmd_vel_topic: str = "/cmd_vel"
    reset_service: str = "/reset_world"
    path_marker_topic: str = "/path_marker"
    map_offset_x: int = 45
    map_offset_y: int = 15


class Ros2Bridge(Node):
    def __init__(self, ros_cfg: RosConfig):
        super().__init__("td3_ros_env")
        self.ros_cfg = ros_cfg

        self.map_msg: Optional[OccupancyGrid] = None
        self.odom_msg: Optional[Odometry] = None
        self.scan_msg: Optional[LaserScan] = None

        self.odom_update_count = 0
        self.last_odom_wall_time = 0.0

        self.map_initialized = False

        self.cmd_pub = self.create_publisher(Twist, self._topic(ros_cfg.cmd_vel_topic), 10)
        self.path_marker_pub = self.create_publisher(Marker, self._topic(ros_cfg.path_marker_topic), 10)

        self.create_subscription(OccupancyGrid, self._topic(ros_cfg.map_topic), self._map_cb, 10)
        self.create_subscription(Odometry, self._topic(ros_cfg.odom_topic), self._odom_cb, 10)
        self.create_subscription(LaserScan, self._topic(ros_cfg.scan_topic), self._scan_cb, 10)

        self.reset_client = self.create_client(Empty, ros_cfg.reset_service)

    def _topic(self, topic: str):
        return f"/{topic.strip('/')}"

    def _map_cb(self, msg):
        self.map_msg = msg
        self.map_resolution = msg.info.resolution
        self.map_origin = (msg.info.origin.position.x, msg.info.origin.position.y)
        self.width = msg.info.width
        self.height = msg.info.height
        self.map_offset = (self.ros_cfg.map_offset_x, self.ros_cfg.map_offset_y)

        grid = costmap(msg.data, self.width, self.height)
        self.grid = np.where((grid == 100) | (grid == -1), 1, 0).astype(np.int8)

        if not self.map_initialized:
            self.map_initialized = True
            self.get_logger().info("Map initialized")

    def _odom_cb(self, msg):
        self.odom_msg = msg
        self.odom_update_count += 1
        self.last_odom_wall_time = time.time()

    def _scan_cb(self, msg):
        self.scan_msg = msg

    def publish_cmd(self, v: float, w: float):
        msg = Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(w)
        self.cmd_pub.publish(msg)

    def reset_world(self, timeout_sec: float = 2.0):
        if not self.reset_client.wait_for_service(timeout_sec=timeout_sec):
            return False
        req = Empty.Request()
        fut = self.reset_client.call_async(req)
        start = time.time()
        while rclpy.ok() and not fut.done() and time.time() - start < timeout_sec:
            rclpy.spin_once(self, timeout_sec=0.05)
        return fut.done()

    def wait_for_fresh_odom(self, prev_count: int, timeout_sec: float = 2.0) -> bool:
        start = time.time()
        while rclpy.ok() and (time.time() - start) < timeout_sec:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.odom_update_count > prev_count:
                return True
        return False

    def visualize_path(self, path: Sequence[Tuple[float, float]]):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "planned_path"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.05
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0

        for x, y in path:
            p = Point()
            p.x = float(x)
            p.y = float(y)
            p.z = 0.0
            marker.points.append(p)

        self.path_marker_pub.publish(marker)


class Environment(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: Optional[EnvConfig] = None, ros: Optional[RosConfig] = None):
        super().__init__()
        self.cfg = config or EnvConfig()
        self.ros_cfg = ros or RosConfig()

        self.action_space = spaces.Box(low=np.array([-1.0, -1.0], dtype=np.float32), high=np.array([1.0, 1.0], dtype=np.float32), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)

        self.node = Ros2Bridge(self.ros_cfg)
        self.map_origin = (-6.85, -7.83)
        self.map_resolution = 0.05
        self.map_offset = (self.ros_cfg.map_offset_x, self.ros_cfg.map_offset_y)
        self.grid: Optional[np.ndarray] = None

        self.goal = None
        self.path_world: List[Tuple[float, float]] = []
        self.pmp = None
        self.steps = 0
        self.last_dist = float("inf")
        self.fixed_goal_grid = None
        self.fixed_goal_world = None

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        prev_odom_count = self.node.odom_update_count
        self.node.reset_world()

        self._wait_topics()
        if not self.node.wait_for_fresh_odom(prev_odom_count, timeout_sec=2.0):
            raise RuntimeError("Odom did not update after reset_world; cannot safely plan from stale pose")

        start_xy, start_g = self._wait_for_valid_start_pose(timeout_sec=3.0)

        if self.fixed_goal_grid is None:
            self._init_fixed_goal(start_g)

        goal_g = self.fixed_goal_grid

        path_rc = None
        for attempt in range(10):
            if not (0 <= start_g[0] < self.grid.shape[1] and 0 <= start_g[1] < self.grid.shape[0]):
                raise RuntimeError(f"Start is out of map bounds after reset: {start_g}")
            if not (0 <= goal_g[0] < self.grid.shape[1] and 0 <= goal_g[1] < self.grid.shape[0]):
                raise RuntimeError(f"Goal is out of map bounds: {goal_g}")

            local_grid = self.grid.copy()
            # Robot cell may be occupied in inflated map; free start for planner bootstrap.
            local_grid[start_g[1], start_g[0]] = 0
            # planner = DStar((start_g[1], start_g[0]), (goal_g[1], goal_g[0]), local_grid)

            planner = DStar((start_g[1], start_g[0]), (goal_g[1], goal_g[0]), local_grid)
            path_rc = planner.plan()
            if path_rc is not None and len(path_rc) >= 2:
                break

            self.node.get_logger().warn(f"[reset] planning failed on attempt {attempt+1}; resetting again")
            prev_odom_count = self.node.odom_update_count
            self.node.reset_world()
            self._wait_topics()
            if not self.node.wait_for_fresh_odom(prev_odom_count, timeout_sec=2.0):
                continue
            start_xy, start_g = self._wait_for_valid_start_pose(timeout_sec=2.0)

        if path_rc is None or len(path_rc) < 2:
            raise RuntimeError(f"Failed to plan path to fixed goal {goal_g} after several attempts")

        self.path_world = [self._grid_to_world(col, row) for row, col in path_rc]
        path_viz = [self._grid_to_world_viz(col, row) for row, col in path_rc]
        self.node.visualize_path(path_viz)
        self.goal = self.fixed_goal_world

        yaw = self._robot_yaw()
        self.pmp = PMP(start_xy[0], start_xy[1], yaw, np.array(self.path_world, dtype=float))

        self.steps = 0
        self.last_dist = self._distance_to_goal()
        self.battery_scale = np.random.uniform(self.cfg.battery_min, self.cfg.battery_max)
        self.slip_lin = np.random.uniform(0.0, self.cfg.slip_lin_max)
        self.slip_ang = np.random.uniform(0.0, self.cfg.slip_ang_max)
        self.wind_bias_v = np.random.uniform(-self.cfg.wind_v_max, self.cfg.wind_v_max)
        self.wind_bias_w = np.random.uniform(-self.cfg.wind_w_max, self.cfg.wind_w_max)
        return self._get_obs(), {}

    def _apply_disturbances(self, v: float, w: float) -> Tuple[float, float]:
        slip_v = np.random.uniform(0.0, self.slip_lin)
        slip_w = np.random.uniform(0.0, self.slip_ang)
        v_real = v * self.battery_scale * (1.0 - slip_v) + self.wind_bias_v
        w_real = w * self.battery_scale * (1.0 - slip_w) + self.wind_bias_w
        v_real = float(np.clip(v_real, 0.0, self.cfg.linear_limit))
        w_real = float(np.clip(w_real, -self.cfg.angular_limit, self.cfg.angular_limit))
        return v_real, w_real

    def step(self, action):
        self.steps += 1
        self._refresh_grid()

        if self.steps % self.cfg.replan_period == 0:
            self._replan()

        x, y = self._robot_world_xy()
        yaw = self._robot_yaw()
        base_v, base_w = self.pmp.control(x, y, yaw)

        action = np.asarray(action, dtype=np.float32)
        dv = float(action[0]) * self.cfg.residual_v_limit
        dw = float(action[1]) * self.cfg.residual_w_limit

        v_cmd = np.clip(base_v + dv, 0.0, self.cfg.linear_limit)
        w_cmd = np.clip(base_w + dw, -self.cfg.angular_limit, self.cfg.angular_limit)

        v_real, w_real = self._apply_disturbances(v_cmd, w_cmd)
        if self._min_scan() < 0.25:
            v_real = min(v_real, 0.03)

        self.node.publish_cmd(v_real, w_real)

        elapsed = 0.0
        while rclpy.ok() and elapsed < self.cfg.dt:
            rclpy.spin_once(self.node, timeout_sec=0.02)
            elapsed += 0.02

        obs = self._get_obs()
        dist = self._distance_to_goal()
        progress = self.last_dist - dist
        self.last_dist = dist

        path_error, heading_error, _ = self._compute_path_errors()
        min_scan = self._min_scan()

        collision = min_scan < self.cfg.obstacle_stop_dist
        reached = dist < self.cfg.goal_tolerance
        timeout = self.steps >= self.cfg.max_steps

        reward = (
            8.0 * progress
            - 0.25 * path_error
            - 0.10 * abs(heading_error)
            - 0.02 * np.linalg.norm(action)
            - 0.01
        )

        safe_dist = 0.50
        danger_dist = 0.25

        if min_scan < safe_dist:
            reward -= 1.5 * ((safe_dist - min_scan) / safe_dist) ** 2

        if min_scan < danger_dist:
            reward -= 3.0 * ((danger_dist - min_scan) / danger_dist) ** 2

        if collision:
            reward -= 60.0

        if reached:
            reward += 100.0
        
        reward /= 100
        terminated = collision or reached
        truncated = timeout

        info: Dict[str, float] = {
            "dist": float(dist),
            "base_v": float(base_v),
            "base_w": float(base_w),
            "residual_v": float(dv),
            "residual_w": float(dw),
            "v_cmd": float(v_cmd),
            "w_cmd": float(w_cmd),
            "v_real": float(v_real),
            "w_real": float(w_real),
            "path_error": float(path_error),
            "heading_error": float(heading_error),
            "min_scan": float(min_scan),
            "battery_scale": float(self.battery_scale),
        }

        if terminated or truncated:
            self.node.publish_cmd(0.0, 0.0)

        return obs, float(reward), bool(terminated), bool(truncated), info

    def close(self):
        self.node.publish_cmd(0.0, 0.0)
        self.node.destroy_node()

    def _wait_topics(self, timeout=120.0):
        start = time.time()
        while rclpy.ok() and time.time() - start < timeout:
            rclpy.spin_once(self.node, timeout_sec=0.05)
            if self.node.map_msg is not None and self.node.odom_msg is not None and self.node.scan_msg is not None:
                return
        raise RuntimeError("ROS topics /map, /odom or /scan are not available")

    def _refresh_grid(self):
        msg = self.node.map_msg
        self.map_resolution = msg.info.resolution
        self.map_origin = (msg.info.origin.position.x, msg.info.origin.position.y)
        inflated = costmap(msg.data, msg.info.width, msg.info.height)
        self.grid = np.where((inflated == 100) | (inflated == -1), 1, 0).astype(np.int8)

    def _wait_for_valid_start_pose(self, timeout_sec: float = 3.0):
        start = time.time()
        while rclpy.ok() and (time.time() - start) < timeout_sec:
            rclpy.spin_once(self.node, timeout_sec=0.05)
            if self.node.map_msg is None or self.node.odom_msg is None:
                continue

            self._refresh_grid()
            start_xy = self._robot_world_xy()
            start_g = self._world_to_grid(*start_xy)

            in_bounds = 0 <= start_g[0] < self.grid.shape[1] and 0 <= start_g[1] < self.grid.shape[0]
            if not in_bounds:
                continue

            # Do not start planning from inflated obstacle / unknown space.
            if self.grid[start_g[1], start_g[0]] != 0:
                continue

            # Additional guard: if lidar still reports immediate contact after reset,
            # wait for simulator/physics to settle before planning.
            if self.node.scan_msg is not None and self._min_scan() < self.cfg.obstacle_stop_dist:
                continue

            return start_xy, start_g

        raise RuntimeError("Failed to obtain a valid start pose after reset (stale or colliding pose)")

    def _robot_world_xy(self):
        p = self.node.odom_msg.pose.pose.position
        return float(p.x), float(p.y)

    def _robot_yaw(self):
        q = self.node.odom_msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return float(math.atan2(siny, cosy))

    def _world_to_grid(self, x_world, y_world):
        x_grid = int((x_world - self.map_origin[0]) / self.map_resolution) + self.map_offset[0]
        y_grid = int((y_world - self.map_origin[1]) / self.map_resolution) + self.map_offset[1]
        return (x_grid, y_grid)

    def _grid_to_world(self, x_grid, y_grid):
        x_world = (x_grid - self.map_offset[0]) * self.map_resolution + self.map_origin[0]
        y_world = (y_grid - self.map_offset[1]) * self.map_resolution + self.map_origin[1]
        # x_world = (x_grid ) * self.map_resolution + self.map_origin[0]
        # y_world = (y_grid ) * self.map_resolution + self.map_origin[1]
        return (x_world, y_world)
    
    def _grid_to_world_viz(self, x_grid, y_grid):
        x_world = (x_grid ) * self.map_resolution + self.map_origin[0]
        y_world = (y_grid ) * self.map_resolution + self.map_origin[1]
        return (x_world, y_world)

    def _distance_to_goal(self):
        x, y = self._robot_world_xy()
        return math.hypot(self.goal[0] - x, self.goal[1] - y)

    def _min_scan(self):
        ranges = [r for r in self.node.scan_msg.ranges if r > 0.0 and math.isfinite(r)]
        if not ranges:
            return 10.0
        return float(min(ranges))

    def _replan(self):
        start_xy = self._robot_world_xy()
        start_g = self._world_to_grid(start_xy[0], start_xy[1])
        goal_g = self._world_to_grid(self.goal[0], self.goal[1])

        if not (0 <= start_g[0] < self.grid.shape[1] and 0 <= start_g[1] < self.grid.shape[0]):
            return
        if not (0 <= goal_g[0] < self.grid.shape[1] and 0 <= goal_g[1] < self.grid.shape[0]):
            return

        local_grid = self.grid.copy()
        local_grid[start_g[1], start_g[0]] = 0
        planner = DStar((start_g[1], start_g[0]), (goal_g[1], goal_g[0]), local_grid)
        path_rc = planner.plan()
        if path_rc is None or len(path_rc) < 2:
            return

        self.path_world = [self._grid_to_world(col, row) for row, col in path_rc]
        path_viz = [self._grid_to_world_viz(col, row) for row, col in path_rc]
        self.node.visualize_path(path_viz)
        self.pmp.set_path(np.array(self.path_world, dtype=float))

    def _init_fixed_goal(self, start_g, max_tries=200):
        free = np.argwhere(self.grid == 0)

        for _ in range(max_tries):
            gy, gx = free[np.random.randint(0, len(free))]
            goal_g = (int(gx), int(gy))

            if abs(goal_g[0] - start_g[0]) + abs(goal_g[1] - start_g[1]) <= 30:
                continue

            if not (0 <= start_g[0] < self.grid.shape[1] and 0 <= start_g[1] < self.grid.shape[0]):
                continue
            if not (0 <= goal_g[0] < self.grid.shape[1] and 0 <= goal_g[1] < self.grid.shape[0]):
                continue

            local_grid = self.grid.copy()
            # Keep start traversable for D* even with obstacle inflation.
            local_grid[start_g[1], start_g[0]] = 0

            planner = DStar((start_g[1], start_g[0]), (goal_g[1], goal_g[0]), local_grid)
            path_rc = planner.plan()
            if path_rc is not None and len(path_rc) >= 2:
                self.fixed_goal_grid = goal_g
                self.fixed_goal_world = self._grid_to_world(goal_g[0], goal_g[1])
                return

        raise RuntimeError("Failed to select a valid fixed goal")

    def _compute_path_errors(self):
        if not self.path_world:
            return 0.0, 0.0, 0
        path = np.asarray(self.path_world, dtype=np.float32)
        pos = np.array(self._robot_world_xy(), dtype=np.float32)
        dists = np.linalg.norm(path - pos, axis=1)
        idx = int(np.argmin(dists))
        nearest = path[idx]
        nxt = path[idx + 1] if idx < len(path) - 1 else path[idx]

        path_heading = math.atan2(nxt[1] - nearest[1], nxt[0] - nearest[0])
        yaw = self._robot_yaw()
        heading_error = (path_heading - yaw + math.pi) % (2 * math.pi) - math.pi
        return float(dists[idx]), float(heading_error), idx

    def _scan_features(self):
        if self.node.scan_msg is None:
            return 1.0, 1.0, 1.0, 1.0

        ranges = np.array(self.node.scan_msg.ranges, dtype=np.float32)
        ranges[~np.isfinite(ranges)] = 10.0
        ranges[ranges <= 0.0] = 10.0

        n = len(ranges)
        front = np.min(np.concatenate([ranges[: n // 12], ranges[-n // 12 :]]))
        all_min = np.min(ranges)
        norm = 3.5
        return float(np.clip(front / norm, 0.0, 1.0)), 1.0, 1.0, float(np.clip(all_min / norm, 0.0, 1.0))

    def _get_obs(self):
        x, y = self._robot_world_xy()
        yaw = self._robot_yaw()

        dx = self.goal[0] - x
        dy = self.goal[1] - y
        dist_to_goal = math.hypot(dx, dy)

        goal_heading = math.atan2(dy, dx)
        goal_angle_err = (goal_heading - yaw + math.pi) % (2 * math.pi) - math.pi

        base_v, base_w = self.pmp.control(x, y, yaw)
        path_error, path_heading_error, _ = self._compute_path_errors()
        front_scan, _, _, min_scan = self._scan_features()

        return np.array([
            np.clip(dx / 5.0, -1.0, 1.0),
            np.clip(dy / 5.0, -1.0, 1.0),
            np.clip(dist_to_goal / 5.0, 0.0, 1.0),
            np.clip(goal_angle_err / math.pi, -1.0, 1.0),
            np.clip(path_error / 2.0, 0.0, 1.0),
            np.clip(path_heading_error / math.pi, -1.0, 1.0),
            np.clip(base_v / max(1e-5, self.cfg.linear_limit), 0.0, 1.0),
            np.clip(base_w / max(1e-5, self.cfg.angular_limit), -1.0, 1.0),
            2.0 * front_scan - 1.0,
            2.0 * min_scan - 1.0,
        ], dtype=np.float32)
