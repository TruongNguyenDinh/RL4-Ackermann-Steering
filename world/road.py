import math
import random
from collections import deque
import pygame


class Road:
    def __init__(
        self,
        width: float = 200,
        segment_length: float = 180,
        samples_per_segment: int = 20,
    ):
        self.width = width
        self.segment_length = segment_length
        self.samples_per_segment = samples_per_segment

        # Lưu giữ lịch sử toàn bộ các điểm điều khiển
        self.history_control_points: list[pygame.Vector2] = []

        # Sliding window
        self.max_active_segments = 10
        max_points = self.max_active_segments * self.samples_per_segment

        self.active_control_points = deque(
            maxlen=self.max_active_segments
        )

        self.centerline = deque(maxlen=max_points)
        self.left_boundary = deque(maxlen=max_points)
        self.right_boundary = deque(maxlen=max_points)

        # Vạch kẻ đường
        self.left_lane_marking = deque(maxlen=max_points)
        self.right_lane_marking = deque(maxlen=max_points)

        self.extend_distance = 600

        # Momentum cho hướng đường
        self.current_heading = 0.0
        self.turn_rate = 0.0

        # Cache polygon cho cast_ray
        self._cached_polygon = []

    def get_start(self):
        return self.history_control_points[0]

    # ==================================================
    # Public API
    # ==================================================

    def generate(self, start=(200, 350), heading=0.0):
        self.active_control_points.clear()
        self.history_control_points.clear()

        self.current_heading = heading
        self.turn_rate = 0.0

        point = pygame.Vector2(start)

        self.active_control_points.append(point.copy())
        self.history_control_points.append(point.copy())

        # Tạo sẵn các điểm ban đầu
        for _ in range(self.max_active_segments - 1):
            self._append_control_point_incremental()

        self._rebuild_sliding_window()

    def update(self, position: pygame.Vector2):
        if len(self.active_control_points) < 2:
            return

        extended = False

        # Sinh thêm đường khi xe/camera đến gần cuối
        while position.distance_to(
            self.active_control_points[-1]
        ) < self.extend_distance:

            self._append_control_point_incremental()
            extended = True

        if extended:
            self._rebuild_sliding_window()

    def _append_control_point_incremental(self):
        """
        Sinh điểm mới dựa trên quán tính (momentum).
        """

        last = self.active_control_points[-1]

        # Noise cho hướng đường
        noise = random.uniform(-0.08, 0.08)
        self.turn_rate += noise

        # Giới hạn độ cong
        self.turn_rate = max(
            -0.25,
            min(0.25, self.turn_rate)
        )

        # Cộng dồn hướng
        self.current_heading += self.turn_rate

        direction = pygame.Vector2(
            math.cos(self.current_heading),
            math.sin(self.current_heading)
        )

        new_point = (
            last
            + direction * self.segment_length
        )

        self.active_control_points.append(
            new_point.copy()
        )

        self.history_control_points.append(
            new_point.copy()
        )

    # ==================================================
    # Sliding Window
    # ==================================================

    def _rebuild_sliding_window(self):
        """
        Tính toán lại geometry trong sliding window.
        """

        self.centerline.clear()
        self.left_boundary.clear()
        self.right_boundary.clear()

        self.left_lane_marking.clear()
        self.right_lane_marking.clear()

        self._generate_centerline()
        self._generate_boundaries()
        self._generate_lane_markings()

        # Cache polygon cho LiDAR / collision
        self._cached_polygon = (
            list(self.left_boundary)
            + list(reversed(self.right_boundary))
        )

    # ==================================================
    # Catmull-Rom
    # ==================================================

    def _catmull_rom(
        self,
        p0,
        p1,
        p2,
        p3,
        t
    ):
        t2 = t * t
        t3 = t2 * t

        return 0.5 * (
            (2 * p1)
            + (-p0 + p2) * t
            + (
                2 * p0
                - 5 * p1
                + 4 * p2
                - p3
            ) * t2
            + (
                -p0
                + 3 * p1
                - 3 * p2
                + p3
            ) * t3
        )

    def _generate_centerline(self):
        if len(self.active_control_points) < 4:
            return

        pts = (
            [self.active_control_points[0]]
            + list(self.active_control_points)
            + [self.active_control_points[-1]]
        )

        for i in range(1, len(pts) - 2):

            p0 = pts[i - 1]
            p1 = pts[i]
            p2 = pts[i + 1]
            p3 = pts[i + 2]

            for j in range(self.samples_per_segment):

                t = j / self.samples_per_segment

                point = self._catmull_rom(
                    p0,
                    p1,
                    p2,
                    p3,
                    t
                )

                self.centerline.append(point)

        self.centerline.append(
            self.active_control_points[-1].copy()
        )

    # ==================================================
    # Road Boundaries
    # ==================================================

    def _generate_boundaries(self):

        n = len(self.centerline)

        if n < 2:
            return

        half_width = self.width / 2

        for i in range(n):

            if i == 0:

                tangent = (
                    self.centerline[1]
                    - self.centerline[0]
                )

            elif i == n - 1:

                tangent = (
                    self.centerline[n - 1]
                    - self.centerline[n - 2]
                )

            else:

                tangent = (
                    self.centerline[i + 1]
                    - self.centerline[i - 1]
                )

            if tangent.length_squared() == 0:
                continue

            tangent = tangent.normalize()

            normal = pygame.Vector2(
                -tangent.y,
                tangent.x
            )

            center = self.centerline[i]

            self.left_boundary.append(
                center + normal * half_width
            )

            self.right_boundary.append(
                center - normal * half_width
            )

    # ==================================================
    # Lane Markings
    # ==================================================

    def _generate_lane_markings(self):

        self.left_lane_marking.clear()
        self.right_lane_marking.clear()

        # Vạch nằm hơi vào phía trong mặt đường
        marking_offset = 5.0

        for center, left, right in zip(
            self.centerline,
            self.left_boundary,
            self.right_boundary
        ):

            # Hướng từ boundary -> center
            left_dir = center - left
            right_dir = center - right

            if left_dir.length_squared() > 0:
                left_dir = left_dir.normalize()

            if right_dir.length_squared() > 0:
                right_dir = right_dir.normalize()

            self.left_lane_marking.append(
                left + left_dir * marking_offset
            )

            self.right_lane_marking.append(
                right + right_dir * marking_offset
            )

    # ==================================================
    # Render
    # ==================================================

    def draw(
        self,
        screen,
        camera,
        show_control_points=True,
        show_centerline=True,
        show_lane_markings=True,
    ):

        n = len(self.left_boundary)

        if n < 2:
            return

        # --------------------------------------------------
        # Road surface
        # --------------------------------------------------

        for i in range(n - 1):

            p1 = camera.world_to_screen(
                self.left_boundary[i]
            )

            p2 = camera.world_to_screen(
                self.right_boundary[i]
            )

            p3 = camera.world_to_screen(
                self.right_boundary[i + 1]
            )

            p4 = camera.world_to_screen(
                self.left_boundary[i + 1]
            )

            pygame.draw.polygon(
                screen,
                (70, 70, 70),
                [p1, p2, p3, p4]
            )

        # --------------------------------------------------
        # Lane markings
        # --------------------------------------------------

        if (
            show_lane_markings
            and len(self.left_lane_marking) > 1
        ):

            left_marking = [
                camera.world_to_screen(p)
                for p in self.left_lane_marking
            ]

            right_marking = [
                camera.world_to_screen(p)
                for p in self.right_lane_marking
            ]

            pygame.draw.lines(
                screen,
                (255, 255, 255),
                False,
                left_marking,
                4
            )

            pygame.draw.lines(
                screen,
                (255, 255, 255),
                False,
                right_marking,
                4
            )

        # --------------------------------------------------
        # Centerline
        # --------------------------------------------------

        if (
            show_centerline
            and len(self.centerline) > 1
        ):

            screen_center = [
                camera.world_to_screen(p)
                for p in self.centerline
            ]

            pygame.draw.lines(
                screen,
                (255, 220, 0),
                False,
                screen_center,
                2
            )

        # --------------------------------------------------
        # Control points
        # --------------------------------------------------

        if show_control_points:

            for point in self.active_control_points:

                pygame.draw.circle(
                    screen,
                    (255, 0, 0),
                    camera.world_to_screen(point),
                    5
                )

    # ==================================================
    # LiDAR / Collision
    # ==================================================

    def get_boundary_polygon(self):
        return self._cached_polygon
    def get_centerline(self):
        """
        Trả về các điểm centerline đã được nội suy.

        Các điểm nằm trong tọa độ World.
        """
        return list(self.centerline)