import math

import numpy as np
import pygame

from world.world import World
from world.road import Road
from vehicle.car import Car
from vehicle.camera import VehicleCamera

from vision.lane_detection import LaneDetector
from vision.path_extraction import PathExtractor

from engine_3D.path_3d import Path3DRenderer
from engine_3D.ground_projection import GroundProjection


class CarEnv:

    # ==================================================
    # INIT
    # ==================================================

    def __init__(
        self,
        width=1400,
        height=900,
    ):

        self.width = width
        self.height = height

        # ==================================================
        # PYGAME
        # ==================================================

        if not pygame.get_init():
            pygame.init()

        # Off-screen surface.
        # Dùng cho perception, không cần cửa sổ.
        self.render_surface = pygame.Surface(
            (
                width,
                height
            )
        )

        # ==================================================
        # WORLD
        # ==================================================

        self.road = Road(
            width=70,
            segment_length=180,
            samples_per_segment=20,
        )

        self.car = Car(
            x=width // 2,
            y=height // 2,
        )

        self.world = World(
            self.road,
            self.car,
        )

        # ==================================================
        # VEHICLE CAMERA
        # ==================================================

        self.vehicle_camera = VehicleCamera(
            width=480,
            height=270,
            fov=90,
            view_distance=500,
        )

        # ==================================================
        # 2D LANE DETECTOR
        # ==================================================

        self.lane_detector = LaneDetector(
            num_waypoints=20,
            y_start=245,
            y_end=80,
        )

        # ==================================================
        # GROUND PROJECTION
        # ==================================================

        self.ground_projection = GroundProjection(
            image_width=480,
            image_height=270,
            fov=90,

            camera_height=80,

            horizon_y=40,
        )

        # ==================================================
        # 3D PATH RENDERER
        # ==================================================

        self.path_renderer = Path3DRenderer(
            width=500,
            height=400,

            fov=90,

            camera_height=80,

            lane_width=70,

            horizon_ratio=0.28,
        )

        # ==================================================
        # OFFICIAL PATH EXTRACTOR
        # ==================================================

        self.path_extractor = PathExtractor(
            num_waypoints=20,

            y_start=390,
            y_end=100,
        )

        # ==================================================
        # 3D SURFACE
        # ==================================================

        self.path_surface = pygame.Surface(
            (
                500,
                400
            )
        )

        # ==================================================
        # RL CONFIG
        # ==================================================

        self.num_points = 10

        self.state_dim = (
            self.num_points * 2
            + 2
        )

        # 10 points × 2
        # velocity
        # steering
        #
        # = 22

        # ==================================================
        # SIMULATION
        # ==================================================

        self.dt = 1.0 / 60.0

        self.done = False

        # ==================================================
        # PERCEPTION DATA
        # ==================================================

        self.camera_image = None

        self.waypoints_2d = []

        self.path_3d = []

        self.official_path = []

        self.processed_path = []

        # ==================================================
        # REWARD
        # ==================================================

        self.previous_lateral_error = None

        self.previous_steering = 0.0

    # ==================================================
    # RESET
    # ==================================================

    def reset(self):

        self.world.reset()

        self.done = False

        self.previous_lateral_error = None

        self.camera_image = None

        self.waypoints_2d = []

        self.path_3d = []

        self.official_path = []

        self.processed_path = []

        self.previous_steering = 0.0

        # --------------------------------------------------
        # Tạo perception ban đầu
        # --------------------------------------------------

        self.update_perception()

        return self.get_state()

    # ==================================================
    # STEP
    # ==================================================

    def step(
        self,
        action,
    ):

        if self.done:

            return (
                self.get_state(),
                0.0,
                True,
            )

        # ==================================================
        # UPDATE PHYSICS
        # ==================================================

        self.world.update(
            self.dt,
            action,
        )

        # ==================================================
        # CHECK WORLD
        # ==================================================

        self.done = self.world.done

        # ==================================================
        # UPDATE PERCEPTION
        # ==================================================

        self.update_perception()

        # ==================================================
        # STATE
        # ==================================================

        next_state = self.get_state()

        # ==================================================
        # REWARD
        # ==================================================

        reward = self.compute_reward()

        return (
            next_state,
            reward,
            self.done,
        )

    # ==================================================
    # PERCEPTION PIPELINE
    #
    # Vehicle Camera
    #       ↓
    # Lane Detector
    #       ↓
    # 2D Waypoints
    #       ↓
    # Ground Projection
    #       ↓
    # 3D Path
    #       ↓
    # Path Renderer
    #       ↓
    # 3D Image
    #       ↓
    # Path Extractor
    #       ↓
    # Official Path
    # ==================================================

    def update_perception(self):

        # ==================================================
        # 1. DRAW WORLD
        # ==================================================

        self.render_surface.fill(
            (240, 240, 240)
        )

        self.world.draw(
            self.render_surface
        )

        # ==================================================
        # 2. VEHICLE CAMERA
        # ==================================================

        self.camera_image = (
            self.vehicle_camera.render(
                self.render_surface,
                self.car,
                self.world.camera,
            )
        )

        # ==================================================
        # 3. 2D LANE DETECTION
        # ==================================================

        self.waypoints_2d = (
            self.lane_detector.extract_path(
                self.camera_image
            )
        )

        # ==================================================
        # 4. 2D -> 3D
        # ==================================================

        self.path_3d = (
            self.ground_projection.project_path(
                self.waypoints_2d
            )
        )

        # ==================================================
        # 5. RENDER 3D PATH
        # ==================================================

        self.path_surface.fill(
            (30, 30, 30)
        )

        self.path_renderer.draw(
            self.path_surface,
            self.path_3d,
        )

        # ==================================================
        # 6. EXTRACT OFFICIAL PATH
        #
        # Đây mới là path mà AI nhìn thấy.
        # ==================================================

        self.official_path = (
            self.path_extractor.extract_path(
                self.path_surface
            )
        )

        # ==================================================
        # 7. OFFICIAL PATH -> RL PATH
        # ==================================================

        self.processed_path = (
            self.official_to_vehicle_path(
                self.official_path
            )
        )

    # ==================================================
    # OFFICIAL PATH -> VEHICLE COORDINATES
    #
    # Official Path:
    #
    #     pixel x
    #     pixel y
    #
    # 3D renderer:
    #
    #     X = lateral
    #     Y = forward
    #
    # RL:
    #
    #     x = forward
    #     y = lateral
    # ==================================================

    def official_to_vehicle_path(
        self,
        official_path,
    ):

        if not official_path:
            return []

        result = []

        for u, v in official_path:

            point = (
                self.pixel_to_ground(
                    u,
                    v
                )
            )

            if point is None:
                continue

            lateral = point[0]
            forward = point[1]

            # ----------------------------------------------
            # RL coordinate convention
            # ----------------------------------------------
            #
            # x = forward
            # y = lateral
            #

            result.append(
                pygame.Vector2(
                    forward,
                    lateral
                )
            )

        return result

    # ==================================================
    # PIXEL -> GROUND
    #
    # Inverse của perspective projection.
    # ==================================================

    def pixel_to_ground(
        self,
        u,
        v,
    ):

        image_width = 500
        image_height = 400

        fov = math.radians(
            90
        )

        camera_height = 80

        # --------------------------------------------------
        # Horizon
        #
        # Path3DRenderer:
        #
        # horizon_ratio = 0.28
        # --------------------------------------------------

        horizon_y = (
            image_height
            * 0.28
        )

        # --------------------------------------------------
        # Focal length
        # --------------------------------------------------

        focal_length = (
            image_width / 2.0
        ) / math.tan(
            fov / 2.0
        )

        cx = (
            image_width / 2.0
        )

        cy = horizon_y

        # --------------------------------------------------
        # Vertical distance from horizon
        # --------------------------------------------------

        dy = (
            float(v)
            - cy
        )

        if dy <= 1.0:
            return None

        # --------------------------------------------------
        # Forward distance
        #
        # Y = H*f / dy
        # --------------------------------------------------

        forward = (
            camera_height
            * focal_length
            / dy
        )

        # --------------------------------------------------
        # Lateral distance
        # --------------------------------------------------

        lateral = (
            (float(u) - cx)
            * forward
            / focal_length
        )

        return (
            lateral,
            forward
        )

    # ==================================================
    # STATE
    # ==================================================

    def get_state(self):

        state = []

        # ==================================================
        # RL PATH
        # ==================================================

        path = self.processed_path

        # ==================================================
        # TAKE MAX 10 POINTS
        # ==================================================

        path = path[
            :self.num_points
        ]

        # ==================================================
        # PATH -> STATE
        # ==================================================

        for point in path:

            # ----------------------------------------------
            # x = forward
            # y = lateral
            # ----------------------------------------------

            state.append(
                float(point.x)
            )

            state.append(
                float(point.y)
            )

        # ==================================================
        # PADDING
        #
        # LUÔN ĐỦ 10 POINTS
        # ==================================================

        while len(path) < self.num_points:

            state.append(
                0.0
            )

            state.append(
                0.0
            )

            path = path + [
                pygame.Vector2(
                    0.0,
                    0.0
                )
            ]

        # ==================================================
        # VEHICLE STATE
        # ==================================================

        state.append(
            float(
                self.car.velocity
            )
        )

        state.append(
            float(
                self.car.steering
            )
        )

        state = np.asarray(
            state,
            dtype=np.float32
        )

        # ==================================================
        # SAFETY CHECK
        # ==================================================

        if len(state) != self.state_dim:

            raise RuntimeError(
                "Invalid state dimension: "
                f"{len(state)} "
                f"(expected {self.state_dim})"
            )

        return state

    # ==================================================
    # REWARD
    # ==================================================

    def compute_reward(self):

        reward = 0.0

        path = self.processed_path

        # ==================================================
        # NO PATH
        # ==================================================

        if len(path) == 0:
            return -100.0

        # ==================================================
        # LATERAL ERROR
        # ==================================================

        lateral_error = abs(path[0].y)

        # Phạt lệch khỏi center path
        reward -= lateral_error * 0.1

        # ==================================================
        # ERROR IMPROVEMENT
        # ==================================================

        if self.previous_lateral_error is not None:

            error_improvement = (
                self.previous_lateral_error
                - lateral_error
            )

            reward += error_improvement * 0.2

        self.previous_lateral_error = lateral_error

        # ==================================================
        # STEERING MAGNITUDE
        # ==================================================

        steering_penalty = abs(
            self.car.steering
        )

        reward -= steering_penalty * 0.02

        # ==================================================
        # STEERING CHANGE
        # ==================================================

        steering_change = abs(
            self.car.steering
            - self.previous_steering
        )

        reward -= steering_change * 0.05

        self.previous_steering = (
            self.car.steering
        )

        # ==================================================
        # SPEED
        # ==================================================

        reward += self.car.velocity * 0.003

        # ==================================================
        # ALIVE
        # ==================================================

        reward += 0.01

        # ==================================================
        # OUT OF ROAD
        # ==================================================

        if self.done:
            reward -= 100.0

        return reward

    # ==================================================
    # GET PATH
    #
    # Dùng cho test_env/debug.
    # ==================================================

    def get_path(self):

        return self.processed_path

    # ==================================================
    # CLOSE
    # ==================================================
    # ==================================================
    # GET PATH
    #
    # Dùng cho test_env/debug.
    # ==================================================

    def get_path(self):

        return self.processed_path

    # ==================================================
    # GET OFFICIAL PATH
    #
    # Path được trích xuất trực tiếp từ ảnh 3D.
    # ==================================================

    def get_official_path(self):

        return self.official_path

    # ==================================================
    # CLOSE
    # ==================================================

    def close(self):

        pass
