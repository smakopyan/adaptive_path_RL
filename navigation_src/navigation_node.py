import math
import time
from collections import deque

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Point, PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
try: 
    from D_star import DStar
    from PMP import PMP
except ModuleNotFoundError:
    from navigation_src.D_star import DStar
    from navigation_src.PMP import PMP

EXPANSION_SIZE = 10


def euler_from_quaternion(x, y, z, w):
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    t2 = +2.0 * (w * y - z * x)
    t2 = max(min(t2, 1.0), -1.0)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    _ = math.atan2(t0, t1)
    _ = math.asin(t2)
    return math.atan2(t3, t4)


def costmap(data, width, height):
    grid = np.array(data, dtype=np.int8).reshape(height, width)
    obstacles_mask = np.where(grid == 100, 255, 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (EXPANSION_SIZE, EXPANSION_SIZE))
    dilated_obstacles = cv2.dilate(obstacles_mask, kernel)
    result = np.where(dilated_obstacles == 255, 100, grid)
    result[grid == -1] = -1
    return result


class Navigation(Node):
    def __init__(self):
        super().__init__("navigation")

        self.map_initialized = False
        self.map_init_time = 0.0

        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

        self.goal = None
        self.path_world = []
        self.last_replan_t = 0.0

        self.laser_data = None
        self.pmp = None

        self.map_offset = (45, 15)

        self.create_subscription(OccupancyGrid, "map", self.map_callback, 10)
        self.create_subscription(Odometry, "odom", self.odom_callback, 10)
        self.create_subscription(PoseStamped, "goal_pose", self.goal_callback, QoSProfile(depth=10))
        self.create_subscription(LaserScan, "scan", self.laser_callback, 10)

        self.cmd_vel_pub = self.create_publisher(Twist, "cmd_vel", 10)
        self.path_marker_pub = self.create_publisher(Marker, "/path_marker", 10)

        self.timer = self.create_timer(0.1, self.navigation_loop)

    def map_callback(self, msg):
        self.map_resolution = msg.info.resolution
        self.map_origin = (msg.info.origin.position.x, msg.info.origin.position.y)
        self.width = msg.info.width
        self.height = msg.info.height

        grid = costmap(msg.data, self.width, self.height)
        self.grid = np.where((grid == 100) | (grid == -1), 1, 0).astype(np.int8)

        if not self.map_initialized:
            self.map_initialized = True
            self.map_init_time = time.time()
            self.get_logger().info("Map initialized")

    def odom_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.yaw = euler_from_quaternion(
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w,
        )

    def goal_callback(self, msg):
        self.goal = (msg.pose.position.x, msg.pose.position.y)
        self.pmp = None
        self.path_world = []
        self.last_replan_t = 0.0
        self.get_logger().info(f"New goal received: {self.goal}")

    def laser_callback(self, msg):
        self.laser_data = msg.ranges

    def world_to_grid(self, x_world, y_world):
        x_grid = int((x_world - self.map_origin[0]) / self.map_resolution) + self.map_offset[0]
        y_grid = int((y_world - self.map_origin[1]) / self.map_resolution) + self.map_offset[1]
        return (x_grid, y_grid)

    def grid_to_world(self, x_grid, y_grid):
        # x_world = (x_grid - self.map_offset[0]) * self.map_resolution + self.map_origin[0]
        # y_world = (y_grid - self.map_offset[1]) * self.map_resolution + self.map_origin[1]
        x_world = x_grid * self.map_resolution + self.map_origin[0]
        y_world = y_grid * self.map_resolution + self.map_origin[1]
        return (x_world, y_world)

    def in_bounds(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height

    def replan_dstar(self):
        if not self.map_initialized or self.goal is None:
            return False

        start_xy = self.world_to_grid(self.x, self.y)
        goal_xy = self.world_to_grid(self.goal[0], self.goal[1])

        if not self.in_bounds(*start_xy) or not self.in_bounds(*goal_xy):
            self.get_logger().warn("Start/Goal out of map bounds")
            return False

        grid_for_plan = self.grid.copy()
        
        # Ensure start position is free (robot is currently there)
        grid_for_plan[start_xy[1], start_xy[0]] = 0
        
        # Check if goal is reachable (not on obstacle)
        if grid_for_plan[goal_xy[1], goal_xy[0]] != 0:
            self.get_logger().warn(f"Goal at ({goal_xy[0]}, {goal_xy[1]}) is on obstacle, retrying with nearby cell")
            # Try to find nearby free cell for goal
            found_free = False
            for di, dj in [(0, 1), (1, 0), (-1, 0), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]:
                new_goal = (goal_xy[0] + dj, goal_xy[1] + di)
                if self.in_bounds(*new_goal) and grid_for_plan[new_goal[1], new_goal[0]] == 0:
                    goal_xy = new_goal
                    found_free = True
                    break
            if not found_free:
                self.get_logger().warn("No free cells near goal")
                return False

        start_rc = (start_xy[1], start_xy[0])
        goal_rc = (goal_xy[1], goal_xy[0])

        try:
            planner = DStar(start_rc, goal_rc, grid_for_plan)
            path_rc = planner.plan()
        except ValueError as exc:
            self.get_logger().warn(str(exc))
            return False

        if path_rc is None or len(path_rc) < 2:
            self.get_logger().warn("D* path not found")
            return False

        self.path_world = [self.grid_to_world(col, row) for (row, col) in path_rc]

        path_xy = np.array(self.path_world, dtype=float)
        if self.pmp is None:
            self.pmp = PMP(self.x, self.y, self.yaw, path_xy, lookahead_dist=0.5)
        else:
            self.pmp.set_path(path_xy)
            self.pmp.x = np.array([self.x, self.y, self.yaw], dtype=float)

        return True

    def navigation_loop(self):
        if not self.map_initialized or time.time() - self.map_init_time < 30.0:
            return

        if self.goal is None:
            self.goal = self.generate_new_goal(min_goal_dist_m=1.0)
            if self.goal is None:
                return
            self.get_logger().info(f"Generated goal: ({self.goal[0]:.2f}, {self.goal[1]:.2f})")

        if self.path_world:
            dx = self.x - self.path_world[-1][0]
            dy = self.y - self.path_world[-1][1]
            if math.hypot(dx, dy) < 0.15:
                self.stop_robot()
                self.get_logger().info("Goal reached!")
                self.goal = None
                self.path_world = []
                self.pmp = None
                return

        now = self.get_clock().now().nanoseconds * 1e-9
        if (now - self.last_replan_t) > 1.0 or not self.path_world:
            ok = self.replan_dstar()
            self.last_replan_t = now
            if not ok:
                self.stop_robot()
                return

        if self.pmp is None or len(self.pmp.path) < 2:
            self.stop_robot()
            return

        self.pmp.x = np.array([self.x, self.y, self.yaw], dtype=float)
        v, w = self.pmp.control(self.x, self.y, self.yaw)

        if self.laser_data:
            n = len(self.laser_data)
            left_range = (int(0.125 * n), int(0.375 * n))   # левые 45 градусов
            right_range = (int(0.625 * n), int(0.875 * n))  # правые 45 градусов
            
            obstacle_threshold = 0.4
            left_block = any(self.laser_data[i] < obstacle_threshold for i in range(*left_range) if self.laser_data[i] > 0.0)
            right_block = any(self.laser_data[i] < obstacle_threshold for i in range(*right_range) if self.laser_data[i] > 0.0)
            
            if left_block and not right_block:
                v, w = 0.1, -math.pi / 6
            elif right_block and not left_block:
                v, w = 0.1, math.pi / 6
            elif left_block and right_block:
                v, w = -0.05, math.pi / 4

        twist = Twist()
        twist.linear.x = float(v)
        twist.angular.z = float(w)
        self.cmd_vel_pub.publish(twist)
        self.visualize_path()

    def stop_robot(self):
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_vel_pub.publish(twist)

    def generate_new_goal(self, min_goal_dist_m=0.8):
        if not self.map_initialized:
            return None

        robot_xy = self.world_to_grid(self.x, self.y)
        if not self.in_bounds(*robot_xy):
            self.get_logger().warn("Robot is out of map bounds; cannot sample goal")
            return None

        grid = self.grid.copy()
        grid[robot_xy[1], robot_xy[0]] = 0

        reachable_rc = self._collect_reachable_free_cells(grid, (robot_xy[1], robot_xy[0]))
        if not reachable_rc:
            self.get_logger().warn("No reachable free cells to generate goal")
            return None

        min_goal_dist_cells = int(max(0.0, min_goal_dist_m / self.map_resolution))
        candidates = []
        for row, col in reachable_rc:
            if abs(col - robot_xy[0]) + abs(row - robot_xy[1]) >= min_goal_dist_cells:
                candidates.append((row, col))

        if not candidates:
            candidates = reachable_rc

        row, col = candidates[np.random.randint(0, len(candidates))]
        self.last_replan_t = 0.0
        return self.grid_to_world(col, row)

    def _collect_reachable_free_cells(self, grid, start_rc):
        sr, sc = start_rc
        if grid[sr, sc] != 0:
            return []

        rows, cols = grid.shape
        q = deque([(sr, sc)])
        visited = np.zeros((rows, cols), dtype=np.uint8)
        visited[sr, sc] = 1

        reachable = []
        while q:
            r, c = q.popleft()
            reachable.append((r, c))

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                    continue
                if visited[nr, nc] or grid[nr, nc] != 0:
                    continue
                visited[nr, nc] = 1
                q.append((nr, nc))

        return reachable

    def visualize_path(self):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "path"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.05
        marker.color.a = 1.0
        marker.color.g = 1.0

        for (x, y) in self.path_world:
            p = Point()
            p.x = float(x)
            p.y = float(y)
            p.z = 0.0
            marker.points.append(p)

        self.path_marker_pub.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    nav = Navigation()
    rclpy.spin(nav)
    nav.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
