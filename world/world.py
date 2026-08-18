import pygame
import math

from world.camera import Camera


class World:
    def __init__(self, road, car):
        self.road = road
        self.car = car

        self.camera = Camera(1400, 900)

        # debug flags
        self.show_debug = True

    # ==================================================
    # RESET WORLD
    # ==================================================
    def reset(self):
        self.car.scan(self)
        self.road.generate((200, 350), 0)

        start = self.road.active_control_points[0]

        self.car.reset(start.x, start.y, 0)

        # camera phải set theo car ngay lập tức
        self.camera.update((start.x, start.y))

    # ==================================================
    # UPDATE WORLD (RL STEP)
    # ==================================================
    def update(self, dt, action):

        self.car.update(action, dt)

        self.road.update(pygame.Vector2(self.car.x, self.car.y))

        self.car.scan(self)   # ✅ FIX QUAN TRỌNG

        self.camera.update((self.car.x, self.car.y))

    # ==================================================
    # RENDER
    # ==================================================
    def draw(self, screen):
        screen.fill((35, 120, 35))

        # ----------------------------
        # ROAD
        # ----------------------------
        self.road.draw(
            screen,
            self.camera,
            show_control_points=False,
            show_centerline=True
        )

        # ----------------------------
        # CAR
        # ----------------------------
        self.car.draw(screen, self.camera)  # ✅ TRUYỀN CAMERA XUỐNG DƯỚI

        # ----------------------------
        # DEBUG INFO
        # ----------------------------
        if self.show_debug:
            self._draw_debug(screen)

    # ==================================================
    # DEBUG HUD
    # ==================================================
    def _draw_debug(self, screen):
        font = pygame.font.SysFont("consolas", 18)

        info = [
            f"Speed: {self.car.velocity:.2f}",
            f"Steering: {math.degrees(self.car.steering):.2f} deg",
            f"Pos: ({self.car.x:.1f}, {self.car.y:.1f})",
            f"Road CP: {len(self.road.active_control_points)}",
        ]

        y = 10
        for line in info:
            surf = font.render(line, True, (0, 0, 0))
            screen.blit(surf, (10, y))
            y += 20

    # ==================================================
    # RL STATE HOOK
    # ==================================================
    def get_state(self):
        return self.car.get_state()

    # ==================================================
    # OPTIONAL: reward function hook (cho RL sau này)
    # ==================================================
    def compute_reward(self):
        """
        Placeholder cho RL
        """
        reward = 0.0

        # đi nhanh
        reward += self.car.velocity * 0.01

        # sau này thêm:
        # - distance to centerline
        # - collision penalty
        # - heading alignment

        return reward
    def cast_ray(self, origin, angle, max_distance=500, coarse_step=25.0):
        direction = pygame.Vector2(math.cos(angle), math.sin(angle))
        origin = pygame.Vector2(origin)

        # Nếu xuất phát đã ngoài đường -> trả ngay, khỏi march
        if not self._is_on_road(origin):
            return 0.0, (origin.x, origin.y)

        # ---- Bước 1: march thô để tìm đoạn chứa điểm va chạm ----
        prev_dist = 0.0
        dist = coarse_step
        hit = False

        while dist <= max_distance:
            point = origin + direction * dist
            if not self._is_on_road(point):
                hit = True
                break
            prev_dist = dist
            dist += coarse_step

        if not hit:
            return max_distance, None

        # ---- Bước 2: binary search refine trong [prev_dist, dist] ----
        lo, hi = prev_dist, dist
        hi_point = origin + direction * hi

        for _ in range(6):  # 6 vòng ~ độ chính xác dưới 1 đơn vị, đủ mượt
            mid = (lo + hi) * 0.5
            mid_point = origin + direction * mid
            if self._is_on_road(mid_point):
                lo = mid
            else:
                hi, hi_point = mid, mid_point

        return hi, (hi_point.x, hi_point.y)
    def is_inside_polygon(self, point, poly):
        inside = False
        j = len(poly) - 1

        for i in range(len(poly)):
            xi, yi = poly[i].x, poly[i].y
            xj, yj = poly[j].x, poly[j].y

            intersect = ((yi > point.y) != (yj > point.y)) and \
                        (point.x < (xj - xi) * (point.y - yi) / (yj - yi + 1e-9) + xi)

            if intersect:
                inside = not inside

            j = i

        return inside
    def _is_on_road(self, point):
        return self.is_inside_polygon(point, self.road.get_boundary_polygon())