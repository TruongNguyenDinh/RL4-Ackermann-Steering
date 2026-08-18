import math
import numpy as np


class LidarProcessor:
    def __init__(
        self,
        num_features: int = 72,
        front_fov_deg: float = 180.0,
        max_distance: float = 250.0,
    ):
        self.num_features = num_features
        self.front_fov_deg = front_fov_deg
        self.max_distance = max_distance

        self.half_fov_deg = front_fov_deg / 2.0

    # ==================================================
    # CHỌN CÁC TIA CẦN RAYCAST
    # ==================================================

    def get_scan_indices(self, relative_angles):
        """
        Chọn num_features tia phân bố đều trong
        vùng phía trước của xe.

        Ví dụ:

            -90° ---------------- +90°
                    72 tia

        relative_angles:
            Toàn bộ các hướng mà LiDAR có thể quét.
            Ví dụ 360 tia: 0°, 1°, ..., 359°.

        Returns
        -------
        list[int]
            Index của các tia cần raycast.
        """

        relative_angles = np.asarray(
            relative_angles,
            dtype=np.float32,
        )

        # ----------------------------------------------
        # Chuyển góc về [-180°, +180°]
        # ----------------------------------------------

        angles_deg = np.degrees(relative_angles)

        angles_deg = (
            (angles_deg + 180.0) % 360.0
        ) - 180.0

        # ----------------------------------------------
        # Tạo 72 góc mục tiêu
        #
        # -90° → +90°
        # ----------------------------------------------

        target_angles = np.linspace(
            -self.half_fov_deg,
            self.half_fov_deg,
            self.num_features,
        )

        scan_indices = []

        # ----------------------------------------------
        # Với mỗi góc mục tiêu,
        # tìm tia LiDAR gần nhất
        # ----------------------------------------------

        for target_angle in target_angles:

            angle_difference = np.abs(
                angles_deg - target_angle
            )

            index = np.argmin(
                angle_difference
            )

            scan_indices.append(
                int(index)
            )

        return scan_indices

    # ==================================================
    # XỬ LÝ KHOẢNG CÁCH
    # ==================================================

    def process(self, distances):
        """
        Chuyển các khoảng cách LiDAR đã chọn
        thành features cho RL.

        distances:
            Phải có num_features phần tử.

        Returns:
            np.ndarray shape = (num_features,)
            Giá trị [0, 1].
        """

        distances = np.asarray(
            distances,
            dtype=np.float32,
        )

        # ----------------------------------------------
        # Kiểm tra số lượng
        # ----------------------------------------------

        if len(distances) != self.num_features:
            raise ValueError(
                f"Expected {self.num_features} "
                f"LiDAR values, "
                f"got {len(distances)}."
            )

        # ----------------------------------------------
        # Xử lý NaN / Inf
        # ----------------------------------------------

        distances = np.nan_to_num(
            distances,
            nan=self.max_distance,
            posinf=self.max_distance,
            neginf=self.max_distance,
        )

        # ----------------------------------------------
        # Giới hạn khoảng cách
        # ----------------------------------------------

        distances = np.clip(
            distances,
            0.0,
            self.max_distance,
        )

        # ----------------------------------------------
        # Normalize
        #
        # 0 m   -> 0.0
        # 250 m -> 1.0
        # ----------------------------------------------

        features = (
            distances / self.max_distance
        )

        return features.astype(np.float32)