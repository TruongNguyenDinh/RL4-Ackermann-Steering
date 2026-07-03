import math
import pygame


class Wheel:
    def __init__(
        self,
        offset_x,
        offset_y,
        steerable=False,
        driven=False
    ):
        # ==========================
        # Geometry
        # ==========================
        self.offset_x = offset_x
        self.offset_y = offset_y

        self.length = 20
        self.width = 8

        # ==========================
        # State
        # ==========================
        self.steer_angle = 0.0          # rad
        self.speed = 0.0                # px/s hoặc m/s
        self.angular_speed = 0.0        # rad/s
        self.torque = 0.0               # Nm

        # ==========================
        # Type
        # ==========================
        self.steerable = steerable
        self.driven = driven

    def draw(self, screen, car, camera): # ✅ THÊM THAM SỐ CAMERA

        # ---------------------------------
        # Tâm bánh trong hệ tọa độ thế giới
        # ---------------------------------
        cos_h = math.cos(car.heading)
        sin_h = math.sin(car.heading)

        center_x = (
            car.x
            + self.offset_x * cos_h
            - self.offset_y * sin_h
        )

        center_y = (
            car.y
            + self.offset_x * sin_h
            + self.offset_y * cos_h
        )

        # ---------------------------------
        # Góc bánh
        # ---------------------------------
        angle = car.heading

        if self.steerable:
            angle += self.steer_angle

        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        hl = self.length / 2
        hw = self.width / 2

        corners = [
            (-hl, -hw),
            ( hl, -hw),
            ( hl,  hw),
            (-hl,  hw)
        ]

        world_points = []

        for x, y in corners:

            wx = center_x + x * cos_a - y * sin_a
            wy = center_y + x * sin_a + y * cos_a

            world_points.append((wx, wy))

        # ---------------------------------
        # ✅ CHUYỂN ĐỔI SANG TỌA ĐỘ MÀN HÌNH (SCREEN COORDINATES)
        # ---------------------------------
        screen_points = [
            camera.world_to_screen(pygame.Vector2(p[0], p[1]))
            for p in world_points
        ]

        # ---------------------------------
        # Màu để debug
        # ---------------------------------
        if self.steerable and self.driven:
            color = (255, 140, 0)      # AWD
        elif self.steerable:
            color = (0, 120, 255)      # Bánh lái
        elif self.driven:
            color = (220, 60, 60)      # Bánh chủ động
        else:
            color = (40, 40, 40)

        # Vẽ bằng screen_points thay vì world_points
        pygame.draw.polygon(
            screen,
            color,
            screen_points # ✅ ĐÃ ĐỔI THÀNH SCREEN POINTS
        )