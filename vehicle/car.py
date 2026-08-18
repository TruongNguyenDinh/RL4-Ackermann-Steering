import math
import pygame

from .wheel import Wheel
from .lidar import Lidar
from physics.bicycle_model import BicycleModel
from physics.ackermann import AckermannSteering
from physics.differential import Differential
from process.lidar_processor import LidarProcessor


class Car:
    def __init__(
        self,
        x,
        y,
        heading=0.0,
        length=60,
        width=30,
        wheel_base=40,
        track_width=30,
    ):
        # ==========================
        # Pose
        # ==========================
        self.x = x
        self.y = y
        self.heading = heading  # radian
        self.track_width = track_width

        # ==========================
        # Geometry
        # ==========================
        self.length = length
        self.width = width
        self.wheel_base = wheel_base

        # ==========================
        # Motion
        # ==========================
        self.velocity = 0.0
        self.acceleration = 0.0
        self.steering = 0.0

        self.max_speed = 300.0
        self.max_acceleration = 300.0
        self.max_steering = math.radians(30)

        # ==========================
        # Bicycle Model
        # ==========================
        self.model = BicycleModel(self.wheel_base)

        self.ackermann = AckermannSteering(
            wheelbase=self.wheel_base,
            track_width=self.track_width
        )

        self.differential = Differential(
            wheelbase=self.wheel_base,
            track_width=self.track_width
        )

        # ==========================
        # Wheels
        # ==========================
        half_wb = wheel_base / 2
        half_track = track_width / 2

        self.front_left = Wheel(
            half_wb,
            half_track,
            steerable=True
        )

        self.front_right = Wheel(
            half_wb,
            -half_track,
            steerable=True
        )

        self.rear_left = Wheel(
            -half_wb,
            half_track,
            driven=True
        )

        self.rear_right = Wheel(
            -half_wb,
            -half_track,
            driven=True
        )

        self.wheels = [
            self.front_left,
            self.front_right,
            self.rear_left,
            self.rear_right
        ]

        # ==========================
        # Sensor
        # ==========================
        self.lidar = Lidar(
            num_rays=1060,
            max_distance=550.0,
        )

        self.lidar_processor = LidarProcessor(
            num_features=108,
            front_fov_deg=180.0,
            max_distance=250.0,
        )

        # ==================================================
        # LiDAR scan indices
        # ==================================================
        # Tính một lần vì:
        # - LiDAR luôn có 360 hướng
        # - FOV luôn 180°
        # - RL luôn dùng 72 features
        #
        # Không cần tính lại mỗi frame.
        self.scan_indices = (
            self.lidar_processor.get_scan_indices(
                self.lidar.relative_angles
            )
        )

        # ==========================
        # LiDAR features
        # ==========================
        self.lidar_features = [
            1.0
        ] * self.lidar_processor.num_features

        # ==========================
        # Steering
        # ==========================
        self.max_steering_rate = math.radians(120)

    # ==================================================
    # Physics
    # ==================================================
    def update(self, action, dt):
        """
        action = (steering, throttle)

        steering : [-1, 1]
        throttle : [-1, 1]
        """

        steering_cmd, throttle = action

        # --------------------------
        # Steering
        # --------------------------
        steering_cmd = max(
            -1.0,
            min(1.0, steering_cmd)
        )

        target = (
            steering_cmd
            * self.max_steering
        )

        delta = target - self.steering

        max_change = (
            self.max_steering_rate
            * dt
        )

        delta = max(
            -max_change,
            min(max_change, delta)
        )

        self.steering += delta

        # --------------------------
        # Throttle -> Acceleration
        # --------------------------
        throttle = max(
            -1.0,
            min(1.0, throttle)
        )

        self.acceleration = (
            throttle
            * self.max_acceleration
        )

        # --------------------------
        # Update velocity
        # --------------------------
        self.velocity += (
            self.acceleration * dt
        )

        self.velocity = max(
            -self.max_speed,
            min(
                self.velocity,
                self.max_speed
            )
        )

        # --------------------------
        # Bicycle Model
        # --------------------------
        self.x, self.y, self.heading = (
            self.model.update(
                self.x,
                self.y,
                self.heading,
                self.velocity,
                self.steering,
                dt
            )
        )

        # Normalize heading
        self.heading = math.atan2(
            math.sin(self.heading),
            math.cos(self.heading)
        )

        # ------------------------------------
        # Ackermann steering
        # ------------------------------------
        left_angle, right_angle = (
            self.ackermann.compute(
                self.steering
            )
        )

        self.front_left.steer_angle = (
            left_angle
        )

        self.front_right.steer_angle = (
            right_angle
        )

        # ------------------------------------
        # Wheel speed
        # ------------------------------------
        wheel_speed = (
            self.differential.compute(
                self.velocity,
                self.steering
            )
        )

        self.front_left.speed = (
            wheel_speed["front_left"]
        )

        self.front_right.speed = (
            wheel_speed["front_right"]
        )

        self.rear_left.speed = (
            wheel_speed["rear_left"]
        )

        self.rear_right.speed = (
            wheel_speed["rear_right"]
        )

    # ==================================================
    # Car corners
    # ==================================================
    def get_corners(self):

        hl = self.length / 2
        hw = self.width / 2

        corners = [
            (-hl, -hw),
            (hl, -hw),
            (hl, hw),
            (-hl, hw)
        ]

        cos_h = math.cos(self.heading)
        sin_h = math.sin(self.heading)

        world = []

        for x, y in corners:

            wx = (
                self.x
                + x * cos_h
                - y * sin_h
            )

            wy = (
                self.y
                + x * sin_h
                + y * cos_h
            )

            world.append((wx, wy))

        return world

    # ==================================================
    # Draw
    # ==================================================
    def draw(self, screen, camera):

        # --------------------------
        # Car body
        # --------------------------

        world_corners = self.get_corners()

        screen_corners = [
            camera.world_to_screen(
                pygame.Vector2(
                    p[0],
                    p[1]
                )
            )
            for p in world_corners
        ]

        pygame.draw.polygon(
            screen,
            (60, 120, 255),
            screen_corners
        )

        # --------------------------
        # Wheels
        # --------------------------

        for wheel in self.wheels:
            wheel.draw(
                screen,
                self,
                camera
            )

        # --------------------------
        # LiDAR
        # --------------------------

        self.lidar.draw(
            screen,
            self,
            camera,
            show_full_scan=True
        )

    # ==================================================
    # Sensor
    # ==================================================
    def scan(self, world):

        # ==================================================
        # 1. Chỉ raycast 72 tia phía trước
        # ==================================================

        self.lidar.scan(
            self,
            world,
            scan_indices=self.scan_indices,
        )

        # ==================================================
        # 2. Lấy khoảng cách của 72 tia
        # ==================================================

        raw_distances = [
            self.lidar.distances[i]
            for i in self.scan_indices
        ]

        # ==================================================
        # 3. Process → 72 features
        # ==================================================

        self.lidar_features = (
            self.lidar_processor.process(
                raw_distances
            )
        )

    # ==================================================
    # RL State
    # ==================================================
    def get_state(self):
        return {
            "position": (
                self.x,
                self.y
            ),

            "heading": self.heading,

            "velocity": self.velocity,

            "steering": self.steering,

            "lidar": self.lidar_features.copy(),
        }

    # ==================================================
    # Reset
    # ==================================================
    def reset(
        self,
        x,
        y,
        heading=0
    ):

        self.x = x
        self.y = y
        self.heading = heading

        self.velocity = 0.0
        self.acceleration = 0.0
        self.steering = 0.0

        for wheel in self.wheels:

            wheel.speed = 0.0
            wheel.steer_angle = 0.0

        # Reset LiDAR
        self.lidar.reset()

        # Reset processed LiDAR features
        self.lidar_features = [
            1.0
        ] * self.lidar_processor.num_features

    # ==================================================
    # For testing
    # ==================================================
    def steer_to_point(
        self,
        target_x,
        target_y
    ):
        """
        Trả về steering command [-1, 1]
        theo vị trí chuột.

        Chỉ dùng để test.
        """

        target_angle = math.atan2(
            target_y - self.y,
            target_x - self.x
        )

        steering = (
            target_angle
            - self.heading
        )

        steering = math.atan2(
            math.sin(steering),
            math.cos(steering)
        )

        steering = max(
            -self.max_steering,
            min(
                self.max_steering,
                steering
            )
        )

        return (
            steering
            / self.max_steering
        )