import rclpy
import time
import cv2
import numpy as np
import math

from rclpy.node import Node
from rclpy.qos import QoSProfile
from nav_msgs.msg import OccupancyGrid, Odometry 
from geometry_msgs.msg import PoseStamped, Twist
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from queue import PriorityQueue

from D_star import DStar
# from PMP import PMP
from pmp_v2 import PMP

expansion_size = 4

def euler_from_quaternion(x, y, z, w):
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
     
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
     
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
     
    return yaw_z

def costmap(data, width, height, resolution):
    grid = np.array(data, dtype=np.int8).reshape(height, width)
    
    obstacles_mask = np.where(grid == 100, 255, 0).astype(np.uint8)
    
    kernel_size = 2 * expansion_size + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated_obstacles = cv2.dilate(obstacles_mask, kernel)
    result = np.where(dilated_obstacles == 255, 100, grid)
    result[grid == -1] = -1
    return result.flatten().tolist()

class Navigation(Node):
    def __init__(self):
        super().__init__('navigation')
        self.map_initialized = False
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        
        self.goal = [6.92748, -4.25212]
        self.path = []
        self.path_world = []
        self.last_replan_t = 0.0

        self.laser_data = None
        self.obstacle_detected = False

        self.pmp = None

        self.subscription_map = self.create_subscription(
            OccupancyGrid, 'map', self.map_callback, 10
        )
        self.subscription_odom = self.create_subscription(
            Odometry, 'odom', self.odom_callback, 10
        )
        self.subscription_goal = self.create_subscription(
            PoseStamped, 'goal_pose', self.goal_callback, QoSProfile(depth=10)
        )
        self.subscription_laser = self.create_subscription(
            LaserScan, 'scan', self.laser_callback, 10
        )


        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.path_marker_pub = self.create_publisher(Marker, '/path_marker', 10)
        self.lookahead_marker_pub = self.create_publisher(Marker, '/lookahead_marker', 10)

        timer_period = 0.1
        self.timer = self.create_timer(timer_period, self.navigation_loop)

    def map_callback(self, msg):
        self.map_resolution = msg.info.resolution
        self.map_origin = (msg.info.origin.position.x, msg.info.origin.position.y)
        self.width = msg.info.width
        self.height = msg.info.height

        self.grid = costmap(msg.data, self.width, self.height, self.map_resolution)
        self.grid = np.array(self.grid).reshape(self.height, self.width)
        self.grid = np.where((self.grid == 100) | (self.grid == -1), 1, 0).astype(np.int8)

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
            msg.pose.pose.orientation.w
        )

    def goal_callback(self, msg):
        self.goal = (msg.pose.position.x, msg.pose.position.y)
        self.pmp = None
        self.path_world = []
        self.last_replan_t = 0.0
        self.get_logger().info(f"New goal received: {self.goal}")

    def laser_callback(self, msg):
        self.laser_data = msg.ranges
        self.obstacle_detected = any(d < 0.3 for d in msg.ranges if d > 0.0)

    def replan_dstar(self):
        if not self.map_initialized or self.goal is None:
            return False

        start_g = self.world_to_grid(self.x, self.y)
        goal_g = self.world_to_grid(self.goal[0], self.goal[1])

        if not self.in_bounds(*start_g) or not self.in_bounds(*goal_g):
            self.get_logger().warn("Start/Goal out of map bounds")
            return False

        costmap = self.grid.copy()
        costmap[start_g[1], start_g[0]] = 0

        start_ij = (start_g[1], start_g[0])
        goal_ij = (goal_g[1], goal_g[0])

        planner = DStar(start_ij, goal_ij, costmap)
        path_ij = planner.plan()

        if path_ij is None or len(path_ij) < 2:
            self.get_logger().warn("D* path not found")
            return False

        self.path_world = [self.grid_to_world(j, i) for (i, j) in path_ij]

        path_xy = np.array(self.path_world, dtype=float)
        if self.pmp is None:
            self.pmp = PMP(
                self.x,
                self.y,
                self.yaw,
                path_xy,
                v_max=0.6,
                w_max=1.2,
                lookahead_dist=0.35,
                goal_tolerance=0.15
            )
        else:
            self.pmp.set_path(path_xy)
            self.pmp.x = np.array([self.x, self.y, self.yaw], dtype=float)

        return True

    def navigation_loop(self):
        if not self.map_initialized or time.time() - self.map_init_time < 90:
            return

        if self.goal is None:
            self.goal = self.generate_new_goal()
            if self.goal is None:
                return

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
            left_start, left_end = int(0.1 * n), int(0.2 * n)
            right_start, right_end = int(0.8 * n), int(0.9 * n)

            left_block = any(
                self.laser_data[i] < 0.3
                for i in range(left_start, left_end)
                if self.laser_data[i] > 0.0
            )
            right_block = any(
                self.laser_data[i] < 0.3
                for i in range(right_start, right_end)
                if self.laser_data[i] > 0.0
            )

            if left_block and not right_block:
                v = 0.08
                w = -math.pi / 4
            elif right_block and not left_block:
                v = 0.08
                w = math.pi / 4
            elif left_block and right_block:
                v = 0.0
                w = math.pi / 4

        twist = Twist()
        twist.linear.x = float(v)
        twist.angular.z = float(w)
        self.cmd_vel_pub.publish(twist)

        self.visualize_path()

    def generate_new_goal(self):
        if not self.map_initialized:
            return None

        self.get_logger().info("Generating new goal...")

        for _ in range(1000):
            x = np.random.randint(0, self.width)
            y = np.random.randint(0, self.height)

            if self.grid[y, x] == 0:
                self.last_replan_t = 0.0
                return self.grid_to_world(x, y)

        self.get_logger().warn("Failed to generate free goal")
        return None
    
    def distance_to_goal(self):
        if not self.path:
            return float('inf')
        return math.hypot(self.x - self.goal[0], self.y - self.goal[1])

    def world_to_grid(self, x_world, y_world, map_offset = (0, 0)):
        x_grid = int((x_world - self.map_origin[0]) / self.map_resolution) + map_offset[0]
        y_grid = int((y_world - self.map_origin[1]) / self.map_resolution) + map_offset[1]
        return (x_grid, y_grid)

    def grid_to_world(self, x_grid, y_grid, map_offset = (0,0)):
        x_world = (x_grid - map_offset[0])* self.map_resolution + self.map_origin[0]
        y_world = (y_grid - map_offset[1]) * self.map_resolution + self.map_origin[1]
        return (x_world, y_world)
    
    def in_bounds(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height

    def is_free(self, x, y):
        return self.in_bounds(x, y) and self.grid[y, x] == 0
    
    def stop_robot(self):
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_vel_pub.publish(twist)

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
        marker.color.r = 0.0
        marker.color.b = 0.0

        for (x, y) in self.path_world:
            p = Point()
            p.x = float(x)
            p.y = float(y)
            p.z = 0.0
            marker.points.append(p)

        self.path_marker_pub.publish(marker)

def main(args=None):
    rclpy.init(args=args)
    navigation = Navigation()
    rclpy.spin(navigation)
    navigation.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()