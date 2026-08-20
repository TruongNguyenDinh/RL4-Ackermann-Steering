import math
import pygame


class PathProcessor:

    def __init__(
        self,
        num_points=10,
        point_spacing=30.0,

        # ==============================================
        # 3D PATH IMAGE
        # ==============================================

        image_width=500,
        image_height=400,

        # FOV của 3D renderer
        fov=90,

        # Camera height
        camera_height=80,

        # Horizon của 3D image
        horizon_y=112,
    ):

        self.num_points = num_points
        self.point_spacing = point_spacing

        self.image_width = image_width
        self.image_height = image_height

        self.fov = math.radians(fov)

        self.camera_height = camera_height
        self.horizon_y = horizon_y

        # ==============================================
        # FOCAL LENGTH
        # ==============================================

        self.focal_length = (
            image_width / 2
        ) / math.tan(
            self.fov / 2
        )

        self.cx = image_width / 2

    # ==================================================
    # 3D IMAGE -> VEHICLE COORDINATE
    # ==================================================

    def image_to_vehicle(
        self,
        x,
        y,
    ):

        # ----------------------------------------------
        # Khoảng cách từ horizon
        # ----------------------------------------------

        dy = (
            y - self.horizon_y
        )

        # Điểm nằm trên horizon
        # không xác định được khoảng cách
        if dy <= 1:

            return None

        # ----------------------------------------------
        # Forward distance
        #
        # Y = H*f / dy
        # ----------------------------------------------

        forward = (
            self.camera_height
            * self.focal_length
            / dy
        )

        # ----------------------------------------------
        # Lateral position
        # ----------------------------------------------

        lateral = (
            (x - self.cx)
            * forward
            / self.focal_length
        )

        # ----------------------------------------------
        # Vehicle coordinate
        #
        # x = lateral
        # y = forward
        # ----------------------------------------------

        return pygame.Vector2(
            lateral,
            forward,
        )

    # ==================================================
    # PROCESS PATH
    # ==================================================

    def process(
        self,
        official_path,
        car_x=None,
        car_y=None,
        car_heading=None,
    ):

        # ==================================================
        # Không có path
        # ==================================================

        if (
            official_path is None
            or len(official_path) == 0
        ):

            return []

        # ==================================================
        # IMAGE -> VEHICLE
        # ==================================================

        vehicle_points = []

        for point in official_path:

            x, y = point

            vehicle_point = (
                self.image_to_vehicle(
                    x,
                    y,
                )
            )

            if vehicle_point is None:
                continue

            vehicle_points.append(
                vehicle_point
            )

        # ==================================================
        # Không còn điểm hợp lệ
        # ==================================================

        if len(vehicle_points) == 0:

            return []

        # ==================================================
        # LẤY ĐIỂM PHÍA TRƯỚC
        #
        # Official Path đã được sắp xếp
        # từ gần -> xa theo Y.
        # ==================================================

        selected_points = []

        for point in vehicle_points:

            # Không lấy điểm phía sau xe
            if point.y <= 0:

                continue

            # ------------------------------------------
            # Kiểm tra khoảng cách giữa các point
            # ------------------------------------------

            if len(selected_points) > 0:

                distance = (
                    point.distance_to(
                        selected_points[-1]
                    )
                )

                if (
                    distance
                    < self.point_spacing
                ):

                    continue

            selected_points.append(
                point
            )

            if (
                len(selected_points)
                >= self.num_points
            ):

                break

        # ==================================================
        # OUTPUT
        # ==================================================

        return selected_points