import math
import pygame


class Path3DRenderer:
    """
    Render path 3D theo kiểu camera perspective nhìn về phía trước.

    Hệ tọa độ:

        X: trái / phải
        Y: phía trước xe
        Z: độ cao

    Mặt đường:
        Z = 0
    """

    def __init__(
        self,
        width=500,
        height=400,

        fov=90,

        camera_height=80,

        lane_width=220,

        horizon_ratio=0.28,
    ):

        self.width = width
        self.height = height

        # ==================================================
        # Camera
        # ==================================================

        self.fov = math.radians(fov)

        self.camera_height = camera_height

        self.lane_width = lane_width

        # Horizon nằm khoảng 28% chiều cao ảnh
        self.horizon_y = (
            self.height * horizon_ratio
        )

        # ==================================================
        # Focal length
        # ==================================================

        self.focal_length = (
            self.width / 2
        ) / math.tan(
            self.fov / 2
        )

        self.perspective_strength = 0.95
    # ==================================================
    # 3D -> SCREEN
    # ==================================================

    def project(
        self,
        x,
        y,
        z=0.0,
    ):
        if y <= 1.0:
            return None

        # Perspective nhưng giảm độ chụm
        perspective = (
            self.focal_length / y
        )

        perspective = (
            1.0
            - self.perspective_strength
            + self.perspective_strength * perspective
        )

        screen_x = (
            self.width / 2
            + x * perspective
        )

        screen_y = (
            self.horizon_y
            + (
                self.camera_height - z
            ) * (
                self.focal_length / y
            )
        )

        return pygame.Vector2(
            screen_x,
            screen_y
        )
    # ==================================================
    # Lane boundary
    # ==================================================

    def get_lane_boundaries(
        self,
        path,
    ):
        """
        Từ center path tạo ra 2 mép lane.

        Không lấy X ± width/2 đơn giản,
        vì path có thể cong.

        Ta dùng hướng tiếp tuyến của path
        để tính vector pháp tuyến.
        """

        left = []
        right = []

        if len(path) < 2:
            return left, right

        half_width = (
            self.lane_width / 2
        )

        for i, point in enumerate(path):

            x, y, z = point

            # --------------------------------------------------
            # Tangent
            # --------------------------------------------------

            if i == 0:

                x1, y1, _ = path[0]
                x2, y2, _ = path[1]

            elif i == len(path) - 1:

                x1, y1, _ = path[-2]
                x2, y2, _ = path[-1]

            else:

                x1, y1, _ = path[i - 1]
                x2, y2, _ = path[i + 1]

            dx = x2 - x1
            dy = y2 - y1

            length = math.hypot(
                dx,
                dy
            )

            if length < 1e-6:
                continue

            dx /= length
            dy /= length

            # --------------------------------------------------
            # Normal
            #
            # tangent = (dx, dy)
            #
            # normal = (-dy, dx)
            # --------------------------------------------------

            nx = -dy
            ny = dx

            left.append(
                (
                    x + nx * half_width,
                    y + ny * half_width,
                    z
                )
            )

            right.append(
                (
                    x - nx * half_width,
                    y - ny * half_width,
                    z
                )
            )

        return left, right

    # ==================================================
    # Draw polyline
    # ==================================================

    def draw_line(
        self,
        surface,
        points,
        color,
        width=3,
    ):

        projected = []

        for x, y, z in points:

            p = self.project(
                x,
                y,
                z
            )

            if p is not None:
                projected.append(p)

        if len(projected) >= 2:

            pygame.draw.lines(
                surface,
                color,
                False,
                projected,
                width
            )

        return projected

    # ==================================================
    # Draw
    # ==================================================

    def draw(
        self,
        surface,
        path,
    ):

        # ==================================================
        # SKY
        # ==================================================

        surface.fill(
            (120, 180, 220)
        )


        # ==================================================
        # ROAD
        # ==================================================

        horizon_y = int(
            self.horizon_y
        )

        pygame.draw.rect(
            surface,
            (90, 90, 90),
            (
                0,
                horizon_y,
                self.width,
                self.height - horizon_y
            )
        )


        # ==================================================
        # HORIZON
        # ==================================================

        pygame.draw.line(
            surface,
            (160, 160, 160),
            (
                0,
                horizon_y
            ),
            (
                self.width,
                horizon_y
            ),
            2
        )


        # ==================================================
        # NO PATH
        # ==================================================

        if len(path) < 2:
            return


        # ==================================================
        # LANE BOUNDARIES
        # ==================================================

        left_boundary, right_boundary = (
            self.get_lane_boundaries(
                path
            )
        )


        # ==================================================
        # DRAW LEFT LANE
        # ==================================================

        self.draw_line(
            surface,
            left_boundary,
            (220, 220, 220),
            5
        )


        # ==================================================
        # DRAW RIGHT LANE
        # ==================================================

        self.draw_line(
            surface,
            right_boundary,
            (220, 220, 220),
            5
        )


        # ==================================================
        # CENTER PATH
        # ==================================================

        center_projected = []


        for x, y, z in path:

            p = self.project(
                x,
                y,
                z
            )

            if p is not None:

                center_projected.append(
                    p
                )


        # ==================================================
        # DRAW CENTER PATH
        # ==================================================

        if len(center_projected) >= 2:

            pygame.draw.lines(
                surface,
                (255, 40, 40),
                False,
                center_projected,
                2
            )


        # ==================================================
        # DRAW WAYPOINTS
        # ==================================================

        for p in center_projected:

            pygame.draw.circle(
                surface,
                (255, 40, 40),
                (
                    int(p.x),
                    int(p.y)
                ),
                4
            )


        # ==================================================
        # VEHICLE REFERENCE
        # ==================================================

        car_x = self.width // 2

        car_y = self.height - 20


        pygame.draw.polygon(
            surface,
            (60, 120, 255),
            [
                (
                    car_x,
                    car_y - 15
                ),
                (
                    car_x - 10,
                    car_y + 10
                ),
                (
                    car_x + 10,
                    car_y + 10
                ),
            ]
        )