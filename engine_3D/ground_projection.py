import math


class GroundProjection:

    def __init__(
        self,
        image_width=480,
        image_height=270,
        fov=90,

        camera_height=80,

        # Horizon thấp hơn
        horizon_y=90,
    ):

        self.image_width = image_width
        self.image_height = image_height

        self.fov = math.radians(fov)

        self.camera_height = camera_height

        self.horizon_y = horizon_y

        # ==================================================
        # Focal length
        # ==================================================

        self.focal_length = (
            image_width / 2
        ) / math.tan(
            self.fov / 2
        )

        self.cx = image_width / 2

    # ==================================================
    # Pixel -> Ground
    # ==================================================

    def project_to_ground(
        self,
        u,
        v,
    ):

        dx = u - self.cx

        dy = v - self.horizon_y

        if dy <= 1:
            return None

        # Khoảng cách phía trước
        Y = (
            self.camera_height
            * self.focal_length
            / dy
        )

        # Độ lệch trái / phải
        X = (
            dx
            * Y
            / self.focal_length
        )

        return (
            X,
            Y,
            0.0
        )

    # ==================================================
    # Path
    # ==================================================

    def project_path(
        self,
        waypoints,
    ):

        path_3d = []

        for point in waypoints:

            x = point[0]
            y = point[1]

            # Nếu có valid flag
            if len(point) >= 3:

                valid = point[2]

                if not valid:
                    continue

            point_3d = self.project_to_ground(
                x,
                y
            )

            if point_3d is not None:

                path_3d.append(
                    point_3d
                )

        return path_3d