import math
import pygame


class PathProcessor:

    def __init__(
        self,
        num_points=10,
        point_spacing=30.0,
    ):
        self.num_points = num_points
        self.point_spacing = point_spacing

    # ==================================================
    # WORLD -> VEHICLE
    # ==================================================
    def world_to_vehicle(
        self,
        point,
        car_x,
        car_y,
        car_heading,
    ):
        dx = point.x - car_x
        dy = point.y - car_y

        cos_h = math.cos(car_heading)
        sin_h = math.sin(car_heading)

        x = (
            cos_h * dx
            + sin_h * dy
        )

        y = (
            -sin_h * dx
            + cos_h * dy
        )

        return pygame.Vector2(x, y)

    # ==================================================
    # PROCESS PATH
    # ==================================================
    def process(
        self,
        centerline,
        car_x,
        car_y,
        car_heading,
    ):
        if len(centerline) == 0:
            return []

        car_position = pygame.Vector2(
            car_x,
            car_y
        )

        # ----------------------------------------------
        # Tìm điểm centerline gần xe nhất
        # ----------------------------------------------

        nearest_index = min(
            range(len(centerline)),
            key=lambda i: centerline[i].distance_to(
                car_position
            )
        )

        # ----------------------------------------------
        # Lấy các điểm phía trước
        # ----------------------------------------------

        selected_points = []

        for i in range(
            nearest_index,
            len(centerline)
        ):

            point = centerline[i]

            if len(selected_points) > 0:

                distance = (
                    point.distance_to(
                        selected_points[-1]
                    )
                )

                if distance < self.point_spacing:
                    continue

            selected_points.append(point)

            if len(selected_points) >= self.num_points:
                break

        # ----------------------------------------------
        # World -> Vehicle
        # ----------------------------------------------

        path = []

        for point in selected_points:

            vehicle_point = (
                self.world_to_vehicle(
                    point,
                    car_x,
                    car_y,
                    car_heading,
                )
            )

            path.append(vehicle_point)

        return path