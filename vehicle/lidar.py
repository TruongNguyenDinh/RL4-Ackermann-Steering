import math
import pygame


class Lidar:
    def __init__(self, num_rays: int = 100, max_distance: float = 250.0):
        self.num_rays = num_rays
        self.max_distance = max_distance

        self.relative_angles = [
            math.radians(i * 360 / num_rays)
            for i in range(num_rays)
        ]

        self.reset()

    def reset(self):
        self.distances = [self.max_distance] * self.num_rays
        self.hit_points = [None] * self.num_rays

    def scan(self, car, world):
        self.reset()

        origin = (car.x, car.y)

        for i, angle in enumerate(self.relative_angles):
            ray_angle = car.heading + angle

            distance, hit_point = world.cast_ray(
                origin=origin,
                angle=ray_angle,
                max_distance=self.max_distance,
            )

            self.distances[i] = distance
            self.hit_points[i] = hit_point  # phải là point, không phải bool

    def get_observation(self):
        return self.distances.copy()

    def draw(self, screen, car, camera): # ✅ THÊM THAM SỐ CAMERA

        # Chuyển đổi vị trí tâm xe sang tọa độ màn hình
        screen_origin = camera.world_to_screen(pygame.Vector2(car.x, car.y))

        for i, angle in enumerate(self.relative_angles):

            # 1. Tính toán điểm cuối ở tọa độ thế giới (World Coordinates)
            if self.hit_points[i] is None:
                ray_angle = car.heading + angle
                world_end = (
                    car.x + self.max_distance * math.cos(ray_angle),
                    car.y + self.max_distance * math.sin(ray_angle),
                )
            else:
                world_end = self.hit_points[i]

            # chống crash nếu dữ liệu lỗi
            if world_end is None:
                continue

            # 2. Chuyển đổi điểm cuối sang tọa độ màn hình (Screen Coordinates)
            screen_end = camera.world_to_screen(pygame.Vector2(world_end[0], world_end[1]))

            # Vẽ đường tia Lidar
            pygame.draw.line(
                screen,
                (0, 255, 0),
                screen_origin,  # ✅ Dùng screen_origin
                screen_end,     # ✅ Dùng screen_end
                1,
            )

            # Vẽ điểm va chạm
            if self.hit_points[i] is not None:
                pygame.draw.circle(
                    screen,
                    (255, 0, 0),
                    (int(screen_end[0]), int(screen_end[1])), # ✅ Dùng screen_end
                    3,
                )