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

        # [YÊU CẦU 2] Lưu giữ lịch sử toàn bộ các điểm điều khiển (không bao giờ xóa)
        self.history_control_points: list[pygame.Vector2] = []

        # [YÊU CẦU 1] Chuyển đổi list sang deque để popleft() đạt độ phức tạp O(1)
        # Giới hạn số lượng điểm đang hoạt động (sliding window) để tính toán luôn là O(1)
        self.max_active_segments = 10 
        max_points = self.max_active_segments * self.samples_per_segment
        
        self.active_control_points = deque(maxlen=self.max_active_segments)
        self.centerline = deque(maxlen=max_points)
        self.left_boundary = deque(maxlen=max_points)
        self.right_boundary = deque(maxlen=max_points)

        self.extend_distance = 600

        # [YÊU CẦU 5] Thuộc tính quán tính (Momentum) cho hướng đường
        self.current_heading = 0.0
        self.turn_rate = 0.0
        self._cached_polygon = []
        print(width)
    def get_start(self):
        return self.control_points[0]
    # ==================================================
    # Public API
    # ==================================================

    def generate(self, start=(200, 350), heading=0.0):
        self.active_control_points.clear()
        self.history_control_points.clear()
        
        self.current_heading = heading
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
        # Nếu vị trí camera/xe đến gần điểm cuối, sinh thêm đoạn mới
        while position.distance_to(self.active_control_points[-1]) < self.extend_distance:
            self._append_control_point_incremental()
            extended = True
            
        if extended:
            # Rebuild lại trên một tập dữ liệu nhỏ (deque cố định) -> Đảm bảo chạy nhanh O(1)
            self._rebuild_sliding_window()

    def _append_control_point_incremental(self):
        """
        Sinh điểm mới dựa trên quán tính (momentum), giảm thiểu gập góc đột ngột.
        """
        last = self.active_control_points[-1]

        # [YÊU CẦU 5] Thêm nhiễu (noise) vào lực bẻ lái, duy trì quán tính
        noise = random.uniform(-0.08, 0.08)
        self.turn_rate += noise
        
        # Giới hạn góc bẻ tối đa để đường không cuộn thành hình tròn quá gắt
        self.turn_rate = max(-0.25, min(0.25, self.turn_rate)) 
        
        # Cộng dồn vào hướng đi hiện tại
        self.current_heading += self.turn_rate

        direction = pygame.Vector2(
            math.cos(self.current_heading), 
            math.sin(self.current_heading)
        )
        
        new_point = last + direction * self.segment_length

        self.active_control_points.append(new_point.copy())
        self.history_control_points.append(new_point.copy()) # Lưu lại vào lịch sử

    # ==================================================
    # Tính toán (Sliding Window O(1))
    # ==================================================

    def _rebuild_sliding_window(self):
        """Tính toán lại chỉ cho các điểm nằm trong giới hạn của deque"""
        self.centerline.clear()
        self.left_boundary.clear()
        self.right_boundary.clear()

        self._generate_centerline()
        self._generate_boundaries()

    def _catmull_rom(self, p0, p1, p2, p3, t):
        t2 = t * t
        t3 = t2 * t
        return 0.5 * (
            (2 * p1) +
            (-p0 + p2) * t +
            (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 +
            (-p0 + 3 * p1 - 3 * p2 + p3) * t3
        )

    def _generate_centerline(self):
        if len(self.active_control_points) < 4:
            return

        # Dùng kỹ thuật nhân bản điểm đầu cuối trong window để Catmull-Rom nối mượt
        pts = [self.active_control_points[0]] + list(self.active_control_points) + [self.active_control_points[-1]]

        for i in range(1, len(pts) - 2):
            p0, p1, p2, p3 = pts[i - 1], pts[i], pts[i + 1], pts[i + 2]

            for j in range(self.samples_per_segment):
                t = j / self.samples_per_segment
                self.centerline.append(self._catmull_rom(p0, p1, p2, p3, t))

        self.centerline.append(self.active_control_points[-1].copy())

    def _generate_boundaries(self):
        n = len(self.centerline)
        if n < 2:
            return

        half_width = self.width / 2

        for i in range(n):
            # [YÊU CẦU 4] Giữ nguyên sai phân trung tâm: P[i+1] - P[i-1]
            if i == 0:
                tangent = self.centerline[1] - self.centerline[0]
            elif i == n - 1:
                tangent = self.centerline[n - 1] - self.centerline[n - 2]
            else:
                tangent = self.centerline[i + 1] - self.centerline[i - 1]

            if tangent.length_squared() == 0:
                continue

            tangent = tangent.normalize()
            normal = pygame.Vector2(-tangent.y, tangent.x)
            center = self.centerline[i]

            self.left_boundary.append(center + normal * half_width)
            self.right_boundary.append(center - normal * half_width)

    # ==================================================
    # Render với Quads
    # ==================================================

    def draw(self, screen, camera, show_control_points=True, show_centerline=True):
        n = len(self.left_boundary)
        if n < 2:
            return

        # [YÊU CẦU 3] Thay vì 1 polygon lớn, vẽ từng Quad (tứ giác) để chống giật lag
        for i in range(n - 1):
            p1 = camera.world_to_screen(self.left_boundary[i])
            p2 = camera.world_to_screen(self.right_boundary[i])
            p3 = camera.world_to_screen(self.right_boundary[i + 1])
            p4 = camera.world_to_screen(self.left_boundary[i + 1])
            
            # Vẽ 1 quad gồm 4 điểm nối P(i) và P(i+1)
            pygame.draw.polygon(screen, (70, 70, 70), [p1, p2, p3, p4])

        # Vẽ tim đường
        if show_centerline and len(self.centerline) > 1:
            screen_center = [camera.world_to_screen(p) for p in self.centerline]
            pygame.draw.lines(screen, (255, 220, 0), False, screen_center, 2)

        # Vẽ điểm điều khiển đang active
        if show_control_points:
            for point in self.active_control_points:
                pygame.draw.circle(
                    screen, 
                    (255, 0, 0), 
                    camera.world_to_screen(point), 
                    5
                )
    def _rebuild_sliding_window(self):
        self.centerline.clear()
        self.left_boundary.clear()
        self.right_boundary.clear()

        self._generate_centerline()
        self._generate_boundaries()

        # Cache polygon 1 lần duy nhất, thay vì dựng lại ở mỗi lần cast_ray
        self._cached_polygon = list(self.left_boundary) + list(reversed(self.right_boundary))

    def get_boundary_polygon(self):
        return self._cached_polygon