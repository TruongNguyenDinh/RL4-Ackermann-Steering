import numpy as np

from world.world import World
from world.road import Road
from vehicle.car import Car
from process.path_processor import PathProcessor


class CarEnv:

    def __init__(
        self,
        width=1400,
        height=900,
    ):
        self.width = width
        self.height = height

        # ==========================
        # World
        # ==========================
        self.road = Road(
            width=220,
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

        # ==========================
        # Path processor
        # ==========================
        self.path_processor = PathProcessor(
            num_points=10,
        )

        # ==========================
        # Environment
        # ==========================
        self.done = False

        self.dt = 1.0 / 60.0

        # ==========================
        # Reward
        # ==========================
        self.previous_lateral_error = None

    # ==================================================
    # RESET
    # ==================================================

    def reset(self):

        self.world.reset()

        self.done = False

        self.previous_lateral_error = None

        return self.get_state()

    # ==================================================
    # STEP
    # ==================================================

    def step(self, action):

        if self.done:

            return (
                self.get_state(),
                0.0,
                True,
            )

        self.world.update(
            self.dt,
            action,
        )

        self.done = self.world.done

        next_state = self.get_state()

        reward = self.compute_reward()

        return (
            next_state,
            reward,
            self.done,
        )

    # ==================================================
    # STATE
    # ==================================================

    def get_state(self):

        centerline = self.road.get_centerline()

        path = self.path_processor.process(
            centerline,
            self.car.x,
            self.car.y,
            self.car.heading,
        )

        state = []

        # --------------------------
        # Path
        # --------------------------

        for point in path:

            state.append(point.x)
            state.append(point.y)

        # --------------------------
        # Padding
        # --------------------------

        while len(path) < self.path_processor.num_points:

            state.append(0.0)
            state.append(0.0)

        # --------------------------
        # Vehicle state
        # --------------------------

        state.append(self.car.velocity)
        state.append(self.car.steering)

        return np.asarray(
            state,
            dtype=np.float32,
        )
    # ==================================================
    # REWARD
    # ==================================================

    def compute_reward(self):

        reward = 0.0

        # ==================================================
        # PATH
        # ==================================================

        centerline = self.road.get_centerline()

        path = self.path_processor.process(
            centerline,
            self.car.x,
            self.car.y,
            self.car.heading,
        )

        # Không có path
        if len(path) == 0:
            return -100.0

        # ==================================================
        # 1. LATERAL ERROR
        # ==================================================

        lateral_error = abs(
            path[0].y
        )

        # Phạt trực tiếp khoảng cách tới centerline
        reward -= lateral_error * 0.05

        # ==================================================
        # 2. IMPROVEMENT
        # ==================================================

        if self.previous_lateral_error is not None:

            error_improvement = (
                self.previous_lateral_error
                - lateral_error
            )

            reward += (
                error_improvement * 0.2
            )

        self.previous_lateral_error = (
            lateral_error
        )

        # ==================================================
        # 3. FORWARD MOVEMENT
        # ==================================================

        reward += self.car.velocity * 0.01

        # ==================================================
        # 4. OUT OF ROAD
        # ==================================================

        if self.done:

            reward -= 100.0

        return reward