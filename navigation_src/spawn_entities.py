#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import random
from dataclasses import dataclass

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from geometry_msgs.msg import Pose, Twist
from gazebo_msgs.srv import SpawnEntity, DeleteEntity, SetEntityState
from gazebo_msgs.msg import EntityState


ARENA_X_MIN = -3.0
ARENA_X_MAX =  3.0
ARENA_Y_MIN = -3.0
ARENA_Y_MAX =  3.0

WALL_MARGIN = 0.5
MIN_DIST_BETWEEN_OBSTACLES = 0.9
MIN_DIST_FROM_ROBOT_START = 1.0

OBSTACLE_COUNT = 4
OBSTACLE_TTL_SEC = 20.0
UPDATE_PERIOD_SEC = 0.1

LINEAR_SPEED_MIN = 0.08
LINEAR_SPEED_MAX = 0.25

BOX_SIZE_X = 0.35
BOX_SIZE_Y = 0.35
BOX_SIZE_Z = 0.7


@dataclass
class DynamicObstacle:
    name: str
    x: float
    y: float
    vx: float
    vy: float
    born_time_sec: float


class TemporaryDynamicObstacles(Node):
    def __init__(self):
        super().__init__('temporary_dynamic_obstacles')

        self.spawn_client = self.create_client(SpawnEntity, '/spawn_entity')
        self.delete_client = self.create_client(DeleteEntity, '/delete_entity')
        self.set_state_client = self.create_client(SetEntityState, '/gazebo/set_entity_state')

        self.obstacles = {}

        self._wait_for_services()

        for i in range(OBSTACLE_COUNT):
            self._spawn_new_obstacle(i)

        self.timer = self.create_timer(UPDATE_PERIOD_SEC, self.update_callback)
        self.get_logger().info('Temporary dynamic obstacles node started.')

    def _wait_for_services(self):
        for client, name in [
            (self.spawn_client, '/spawn_entity'),
            (self.delete_client, '/delete_entity'),
            (self.set_state_client, '/gazebo/set_entity_state'),
        ]:
            while not client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info(f'Waiting for service {name} ...')

    def _make_box_sdf(self, size_x: float, size_y: float, size_z: float) -> str:
        return f"""
<sdf version="1.6">
  <model name="dynamic_box">
    <static>false</static>

    <link name="link">
      <inertial>
        <mass>2.0</mass>
        <inertia>
          <ixx>0.05</ixx>
          <iyy>0.05</iyy>
          <izz>0.05</izz>
          <ixy>0.0</ixy>
          <ixz>0.0</ixz>
          <iyz>0.0</iyz>
        </inertia>
      </inertial>

      <collision name="collision">
        <geometry>
          <box>
            <size>{size_x} {size_y} {size_z}</size>
          </box>
        </geometry>
      </collision>

      <visual name="visual">
        <geometry>
          <box>
            <size>{size_x} {size_y} {size_z}</size>
          </box>
        </geometry>
        <material>
          <ambient>0.8 0.1 0.1 1</ambient>
          <diffuse>0.8 0.1 0.1 1</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>
""".strip()

    def _distance(self, x1, y1, x2, y2) -> float:
        return math.hypot(x1 - x2, y1 - y2)

    def _random_velocity(self):
        speed = random.uniform(LINEAR_SPEED_MIN, LINEAR_SPEED_MAX)
        angle = random.uniform(0.0, 2.0 * math.pi)
        return speed * math.cos(angle), speed * math.sin(angle)

    def _is_valid_position(self, x: float, y: float, ignore_name: str = None) -> bool:
        if not (ARENA_X_MIN + WALL_MARGIN <= x <= ARENA_X_MAX - WALL_MARGIN):
            return False
        if not (ARENA_Y_MIN + WALL_MARGIN <= y <= ARENA_Y_MAX - WALL_MARGIN):
            return False

        if self._distance(x, y, 0.0, 0.0) < MIN_DIST_FROM_ROBOT_START:
            return False

        for name, obs in self.obstacles.items():
            if ignore_name is not None and name == ignore_name:
                continue
            if self._distance(x, y, obs.x, obs.y) < MIN_DIST_BETWEEN_OBSTACLES:
                return False

        return True

    def _random_free_position(self):
        for _ in range(200):
            x = random.uniform(ARENA_X_MIN + WALL_MARGIN, ARENA_X_MAX - WALL_MARGIN)
            y = random.uniform(ARENA_Y_MIN + WALL_MARGIN, ARENA_Y_MAX - WALL_MARGIN)
            if self._is_valid_position(x, y):
                return x, y
        raise RuntimeError('Could not find free position for obstacle.')

    def _spawn_new_obstacle(self, idx: int):
        name = f'temp_dynamic_obstacle_{idx}'
        x, y = self._random_free_position()
        vx, vy = self._random_velocity()

        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = BOX_SIZE_Z / 2.0

        req = SpawnEntity.Request()
        req.name = name
        req.xml = self._make_box_sdf(BOX_SIZE_X, BOX_SIZE_Y, BOX_SIZE_Z)
        req.robot_namespace = ''
        req.initial_pose = pose
        req.reference_frame = 'world'

        future = self.spawn_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is None:
            self.get_logger().error(f'Failed to spawn obstacle {name}')
            return

        now_sec = self.get_clock().now().nanoseconds / 1e9
        self.obstacles[name] = DynamicObstacle(
            name=name,
            x=x,
            y=y,
            vx=vx,
            vy=vy,
            born_time_sec=now_sec
        )

        self.get_logger().info(
            f'Spawned {name} at ({x:.2f}, {y:.2f}), velocity ({vx:.2f}, {vy:.2f})'
        )

    def _delete_obstacle(self, name: str):
        if name not in self.obstacles:
            return

        req = DeleteEntity.Request()
        req.name = name

        future = self.delete_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is None:
            self.get_logger().warning(f'Failed to delete obstacle {name}')
        else:
            self.get_logger().info(f'Deleted {name}')

        self.obstacles.pop(name, None)

    def _respawn_obstacle(self, name: str):
        idx = int(name.split('_')[-1])
        self._delete_obstacle(name)
        self._spawn_new_obstacle(idx)

    def _set_entity_state(self, obs: DynamicObstacle):
        state = EntityState()
        state.name = obs.name
        state.pose.position.x = obs.x
        state.pose.position.y = obs.y
        state.pose.position.z = BOX_SIZE_Z / 2.0
        state.twist = Twist()
        state.twist.linear.x = obs.vx
        state.twist.linear.y = obs.vy
        state.reference_frame = 'world'

        req = SetEntityState.Request()
        req.state = state

        future = self.set_state_client.call_async(req)
        return future

    def update_callback(self):
        now_sec = self.get_clock().now().nanoseconds / 1e9
        dt = UPDATE_PERIOD_SEC

        obstacle_names = list(self.obstacles.keys())

        for name in obstacle_names:
            if name not in self.obstacles:
                continue

            obs = self.obstacles[name]

            if now_sec - obs.born_time_sec >= OBSTACLE_TTL_SEC:
                self.get_logger().info(f'{name} expired, respawning...')
                self._respawn_obstacle(name)
                continue

            new_x = obs.x + obs.vx * dt
            new_y = obs.y + obs.vy * dt

            bounced = False

            if new_x <= ARENA_X_MIN + WALL_MARGIN or new_x >= ARENA_X_MAX - WALL_MARGIN:
                obs.vx *= -1.0
                bounced = True

            if new_y <= ARENA_Y_MIN + WALL_MARGIN or new_y >= ARENA_Y_MAX - WALL_MARGIN:
                obs.vy *= -1.0
                bounced = True

            if bounced:
                new_x = obs.x + obs.vx * dt
                new_y = obs.y + obs.vy * dt

            collision_predicted = False
            for other_name, other_obs in self.obstacles.items():
                if other_name == name:
                    continue
                if self._distance(new_x, new_y, other_obs.x, other_obs.y) < MIN_DIST_BETWEEN_OBSTACLES:
                    collision_predicted = True
                    break

            if collision_predicted:
                obs.vx, obs.vy = self._random_velocity()
                new_x = obs.x + obs.vx * dt
                new_y = obs.y + obs.vy * dt

            if self._is_valid_position(new_x, new_y, ignore_name=name):
                obs.x = new_x
                obs.y = new_y

            self._set_entity_state(obs)

    def destroy_node(self):
        for name in list(self.obstacles.keys()):
            self._delete_obstacle(name)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TemporaryDynamicObstacles()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()