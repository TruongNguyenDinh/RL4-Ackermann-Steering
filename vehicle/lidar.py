import math
import pygame


class Lidar:
    def __init__(
        self,
        num_rays: int = 100,
        max_distance: float = 250.0,
    ):
        self.num_rays = num_rays
        self.max_distance = max_distance

        # Góc của các tia (360°)
        self.relative_angles = [
            math.radians(i * 360 / num_rays)
            for i in range(num_rays)
        ]

        self.reset()

    def reset(self):
        self.distances = [self.max_distance] * self.num_rays
        self.hit_points = [None] * self.num_rays

    def scan(self, car, world):
        """
        world.cast_ray(origin, angle, max_distance)
        -> (distance, hit_point)
        """

        self.reset()

        for i, angle in enumerate(self.relative_angles):

            ray_angle = car.heading + angle

            distance, hit = world.cast_ray(
                origin=(car.x, car.y),
                angle=ray_angle,
                max_distance=self.max_distance,
            )

            self.distances[i] = distance
            self.hit_points[i] = hit

    def get_observation(self):
        return self.distances.copy()

    def draw(self, screen, car):
        """
        Nếu đã scan() thì vẽ tới điểm va chạm.
        Nếu chưa scan() thì vẽ tia có chiều dài max_distance.
        """

        for i, angle in enumerate(self.relative_angles):

            start = (car.x, car.y)

            if self.hit_points[i] is None:

                ray_angle = car.heading + angle

                end = (
                    car.x + self.max_distance * math.cos(ray_angle),
                    car.y + self.max_distance * math.sin(ray_angle),
                )

            else:
                end = self.hit_points[i]

            pygame.draw.line(
                screen,
                (0, 255, 0),
                start,
                end,
                1,
            )

            if self.hit_points[i] is not None:
                pygame.draw.circle(
                    screen,
                    (255, 0, 0),
                    (int(end[0]), int(end[1])),
                    3,
                )