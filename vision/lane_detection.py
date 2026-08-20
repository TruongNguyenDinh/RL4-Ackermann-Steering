import cv2
import numpy as np
import pygame


class LaneDetector:

    def __init__(
        self,
        num_waypoints=20,
        y_start=260,
        y_end=80,
    ):

        self.num_waypoints = num_waypoints
        self.y_start = y_start
        self.y_end = y_end

    # ==================================================
    # PYGAME -> OPENCV
    # ==================================================

    def surface_to_image(self, surface):

        array = pygame.surfarray.array3d(surface)

        image = np.transpose(
            array,
            (1, 0, 2)
        )

        return cv2.cvtColor(
            image,
            cv2.COLOR_RGB2BGR
        )

    # ==================================================
    # DETECT ROAD
    # ==================================================

    def get_road_mask(self, image):

        hsv = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2HSV
        )

        # Road hiện tại màu xám
        lower = np.array(
            [0, 0, 40]
        )

        upper = np.array(
            [180, 80, 180]
        )

        mask = cv2.inRange(
            hsv,
            lower,
            upper
        )

        return mask

    # ==================================================
    # CENTER PATH
    # ==================================================

    def extract_path(self, surface):

        image = self.surface_to_image(
            surface
        )

        mask = self.get_road_mask(
            image
        )

        height, width = mask.shape

        y_start = min(
            self.y_start,
            height - 1
        )

        y_end = max(
            self.y_end,
            0
        )

        # ==================================================
        # TẠO CÁC MỐC Y CỐ ĐỊNH
        # ==================================================

        sample_ys = np.linspace(
            y_start,
            y_end,
            self.num_waypoints
        ).astype(int)

        detected = []

        # ==================================================
        # TÌM CENTER Ở TỪNG Y
        # ==================================================

        for y in sample_ys:

            row = mask[y]

            xs = np.where(
                row > 0
            )[0]

            if len(xs) < 2:

                detected.append(
                    (
                        None,
                        y
                    )
                )

                continue

            # --------------------------------------------------
            # Tách các vùng road
            # --------------------------------------------------

            groups = []

            start = xs[0]
            previous = xs[0]

            for x in xs[1:]:

                if x - previous > 5:

                    groups.append(
                        (
                            start,
                            previous
                        )
                    )

                    start = x

                previous = x

            groups.append(
                (
                    start,
                    previous
                )
            )

            # --------------------------------------------------
            # Chọn vùng rộng nhất
            # --------------------------------------------------

            if len(groups) == 0:

                detected.append(
                    (
                        None,
                        y
                    )
                )

                continue

            best = max(
                groups,
                key=lambda g: g[1] - g[0]
            )

            left_x = best[0]
            right_x = best[1]

            center_x = (
                left_x + right_x
            ) / 2

            detected.append(
                (
                    center_x,
                    y
                )
            )

        # ==================================================
        # INTERPOLATE NHỮNG ĐIỂM BỊ MẤT
        # ==================================================

        valid = [
            (x, y)
            for x, y in detected
            if x is not None
        ]

        if len(valid) < 2:
            return []

        valid_x = np.array(
            [
                x
                for x, y in valid
            ],
            dtype=float
        )

        valid_y = np.array(
            [
                y
                for x, y in valid
            ],
            dtype=float
        )

        result = []

        for x, y in detected:

            if x is not None:

                result.append(
                    (
                        float(x),
                        float(y)
                    )
                )

            else:

                x_interp = np.interp(
                    y,
                    valid_y[::-1],
                    valid_x[::-1]
                )

                result.append(
                    (
                        float(x_interp),
                        float(y)
                    )
                )

        return result