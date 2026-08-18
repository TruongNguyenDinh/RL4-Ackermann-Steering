import math
import pygame


class Lidar:
    def __init__(
        self,
        num_rays: int = 360,
        max_distance: float = 250.0,
    ):
        self.num_rays = num_rays
        self.max_distance = max_distance

        # ==================================================
        # Raw LiDAR geometry: 360°
        # ==================================================

        self.relative_angles = [
            math.radians(i * 360 / num_rays)
            for i in range(num_rays)
        ]

        self.reset()

    # ==================================================
    # RESET
    # ==================================================

    def reset(self):
        self.scanned = [False] * self.num_rays

        self.distances = [
            self.max_distance
            for _ in range(self.num_rays)
        ]

        self.hit_points = [
            None
            for _ in range(self.num_rays)
        ]

    # ==================================================
    # SCAN
    # ==================================================

    def scan(self, car, world, scan_indices=None):
        """
        Raycast các tia được yêu cầu.

        scan_indices:
            None:
                raycast toàn bộ 360 tia.

            list[int]:
                chỉ raycast các tia được chỉ định.
        """

        self.reset()

        origin = (car.x, car.y)

        if scan_indices is None:
            scan_indices = range(self.num_rays)

        for i in scan_indices:

            angle = self.relative_angles[i]

            # Góc tương đối → góc world
            ray_angle = car.heading + angle

            distance, hit_point = world.cast_ray(
                origin=origin,
                angle=ray_angle,
                max_distance=self.max_distance,
            )

            self.scanned[i] = True
            self.distances[i] = distance
            self.hit_points[i] = hit_point

    # ==================================================
    # RAW DATA
    # ==================================================

    def get_raw_scan(self):
        """
        Trả về dữ liệu raw của LiDAR.
        """

        return {
            "angles": self.relative_angles.copy(),
            "distances": self.distances.copy(),
            "scanned": self.scanned.copy(),
        }

    # ==================================================
    # DRAW
    # ==================================================

    def draw(
        self,
        screen,
        car,
        camera,
        show_full_scan=False,
    ):
        """
        Visualization LiDAR.

        show_full_scan=False:
            Chỉ vẽ các tia thực sự được raycast.
            Đây là mặc định.

        show_full_scan=True:
            Vẽ thêm toàn bộ 360 hướng LiDAR để debug.

            Các tia này KHÔNG raycast.
            Chúng chỉ kéo dài tới max_distance.
        """

        screen_origin = camera.world_to_screen(
            pygame.Vector2(car.x, car.y)
        )

        # ==================================================
        # 1. FULL SCAN VISUALIZATION
        # ==================================================
        #
        # Chỉ để nhìn toàn bộ 360°.
        # KHÔNG gọi world.cast_ray().
        #
        # Màu khác với 72 tia thật.
        # ==================================================

        if show_full_scan:

            for i, angle in enumerate(
                self.relative_angles
            ):

                # Tia đã được raycast sẽ được vẽ
                # bằng chế độ bên dưới.
                #
                # Ở đây chỉ vẽ những tia CHƯA được scan.
                if self.scanned[i]:
                    continue

                ray_angle = car.heading + angle

                world_end = (
                    car.x
                    + self.max_distance
                    * math.cos(ray_angle),

                    car.y
                    + self.max_distance
                    * math.sin(ray_angle),
                )

                screen_end = camera.world_to_screen(
                    pygame.Vector2(
                        world_end[0],
                        world_end[1],
                    )
                )

                # Màu riêng cho tia visualization
                pygame.draw.line(
                    screen,
                    (100, 100, 100),
                    screen_origin,
                    screen_end,
                    1,
                )

        # ==================================================
        # 2. ACTUAL SCANNED RAYS
        # ==================================================
        #
        # Đây là các tia thật sự được raycast.
        #
        # Ví dụ:
        # 360 raw rays
        #       ↓
        # 72 scan indices
        #       ↓
        # 72 raycast
        #
        # Đây mới là dữ liệu sensor thực sự.
        # ==================================================

        for i, angle in enumerate(
            self.relative_angles
        ):

            if not self.scanned[i]:
                continue

            # ------------------------------------------
            # Có hit
            # ------------------------------------------

            if self.hit_points[i] is not None:

                world_end = self.hit_points[i]

            # ------------------------------------------
            # Không hit
            # ------------------------------------------

            else:

                ray_angle = car.heading + angle

                world_end = (
                    car.x
                    + self.max_distance
                    * math.cos(ray_angle),

                    car.y
                    + self.max_distance
                    * math.sin(ray_angle),
                )

            # ------------------------------------------
            # World → Screen
            # ------------------------------------------

            screen_end = camera.world_to_screen(
                pygame.Vector2(
                    world_end[0],
                    world_end[1],
                )
            )

            # ------------------------------------------
            # Vẽ tia thật
            # ------------------------------------------

            pygame.draw.line(
                screen,
                (0, 255, 0),
                screen_origin,
                screen_end,
                2,
            )

            # ------------------------------------------
            # Vẽ điểm hit
            # ------------------------------------------

            if self.hit_points[i] is not None:

                pygame.draw.circle(
                    screen,
                    (255, 0, 0),
                    (
                        int(screen_end[0]),
                        int(screen_end[1]),
                    ),
                    3,
                )