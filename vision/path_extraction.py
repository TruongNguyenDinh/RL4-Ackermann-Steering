import cv2
import numpy as np
import pygame


class PathExtractor:
    """
    Trích xuất Official Path từ ảnh 3D.

    Input:
        Pygame Surface của 3D PATH.

    Output:
        List[(x, y)] là các waypoint của Official Path.

    Chỉ lấy phần path màu đỏ thực sự tồn tại.
    Không tự sinh path ở vùng không nhìn thấy.
    """

    def __init__(
        self,
        num_waypoints=20,
        y_start=390,
        y_end=100,
        max_missing=3,
    ):

        self.num_waypoints = num_waypoints

        self.y_start = y_start
        self.y_end = y_end

        self.max_missing = max_missing

    # ==================================================
    # Pygame -> OpenCV
    # ==================================================

    def surface_to_image(self, surface):

        array = pygame.surfarray.array3d(
            surface
        )

        image = np.transpose(
            array,
            (1, 0, 2)
        )

        # Pygame: RGB
        # OpenCV: BGR
        image = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2BGR
        )

        return image

    # ==================================================
    # Detect RED
    # ==================================================

    def detect_path_mask(self, image):

        # --------------------------------------------------
        # Cách 1: HSV
        # --------------------------------------------------

        hsv = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2HSV
        )

        lower_red_1 = np.array(
            [0, 70, 70],
            dtype=np.uint8
        )

        upper_red_1 = np.array(
            [15, 255, 255],
            dtype=np.uint8
        )

        lower_red_2 = np.array(
            [165, 70, 70],
            dtype=np.uint8
        )

        upper_red_2 = np.array(
            [180, 255, 255],
            dtype=np.uint8
        )

        mask1 = cv2.inRange(
            hsv,
            lower_red_1,
            upper_red_1
        )

        mask2 = cv2.inRange(
            hsv,
            lower_red_2,
            upper_red_2
        )

        mask = cv2.bitwise_or(
            mask1,
            mask2
        )

        # ==================================================
        # Morphology
        # ==================================================

        kernel = np.ones(
            (3, 3),
            np.uint8
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel
        )

        return mask

    # ==================================================
    # Find RED pixels at Y
    # ==================================================

    def find_x_at_y(
        self,
        mask,
        y,
    ):

        height, width = mask.shape

        if y < 0 or y >= height:
            return None

        row = mask[y]

        xs = np.where(
            row > 0
        )[0]

        if len(xs) == 0:
            return None

        # --------------------------------------------------
        # Center của path tại Y
        # --------------------------------------------------

        return float(
            np.mean(xs)
        )

    # ==================================================
    # Extract
    # ==================================================

    def extract_path(
        self,
        surface,
    ):

        image = self.surface_to_image(
            surface
        )

        mask = self.detect_path_mask(
            image
        )

        height, width = mask.shape

        # ==================================================
        # Clamp vùng scan
        # ==================================================

        y_start = min(
            int(self.y_start),
            height - 1
        )

        y_end = max(
            int(self.y_end),
            0
        )

        # Đảm bảo gần -> xa
        if y_start < y_end:

            y_start, y_end = (
                y_end,
                y_start
            )

        # ==================================================
        # Scan dày hơn
        #
        # Không lấy đúng num_waypoints ngay từ đầu.
        # Lấy nhiều sample trước để xác định vùng path.
        # ==================================================

        scan_count = max(
            self.num_waypoints * 3,
            50
        )

        ys = np.linspace(
            y_start,
            y_end,
            scan_count
        ).astype(int)

        detected = []

        missing_count = 0

        # ==================================================
        # Detect
        # ==================================================

        for y in ys:

            x = self.find_x_at_y(
                mask,
                y
            )

            # --------------------------------------------------
            # Không thấy đỏ
            # --------------------------------------------------

            if x is None:

                missing_count += 1

                if (
                    missing_count
                    > self.max_missing
                    and len(detected) > 0
                ):
                    break

                continue

            # --------------------------------------------------
            # Có path
            # --------------------------------------------------

            missing_count = 0

            detected.append(
                (
                    x,
                    float(y)
                )
            )

        # ==================================================
        # Không tìm thấy
        # ==================================================

        if not detected:
            return []

        # ==================================================
        # Nếu chỉ có 1 điểm
        # ==================================================

        if len(detected) == 1:

            return [
                (
                    int(detected[0][0]),
                    int(detected[0][1])
                )
            ]

        # ==================================================
        # Lấy vùng path thực sự
        # ==================================================

        x_values = np.array(
            [
                p[0]
                for p in detected
            ],
            dtype=float
        )

        y_values = np.array(
            [
                p[1]
                for p in detected
            ],
            dtype=float
        )

        # ==================================================
        # Resample
        #
        # CHỈ nằm trong vùng path đã detect.
        # ==================================================

        sample_count = min(
            self.num_waypoints,
            len(detected)
        )

        sample_y = np.linspace(
            y_values.max(),
            y_values.min(),
            sample_count
        )

        # np.interp cần X tăng dần
        sort_index = np.argsort(
            y_values
        )

        sorted_y = y_values[
            sort_index
        ]

        sorted_x = x_values[
            sort_index
        ]

        sample_x = np.interp(
            sample_y,
            sorted_y,
            sorted_x
        )

        # ==================================================
        # Result
        # ==================================================

        result = []

        for x, y in zip(
            sample_x,
            sample_y
        ):

            result.append(
                (
                    int(round(x)),
                    int(round(y))
                )
            )

        return result