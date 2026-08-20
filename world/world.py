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
        # check if car is out of road (for RL)
        self.done = False
    # ==================================================
    # RESET WORLD
    # ==================================================
    def reset(self):
        self.road.generate((200, 350), 0)

        start = self.road.active_control_points[0]

        self.car.reset(start.x, start.y, 0)

        # camera phải set theo car ngay lập tức
        self.camera.update((start.x, start.y))
        # car phải set lại trạng thái done
        self.done = False
    # ==================================================
    # UPDATE WORLD (RL STEP)
    # ==================================================
    def update(self, dt, action):

        if self.done==True:
            self.reset()
            return

        self.car.update(action, dt)

        self.road.update(
            pygame.Vector2(
                self.car.x,
                self.car.y
            )
        )

        if self.is_car_out_of_road():
            self.done = True
            return

        self.camera.update(
            (self.car.x, self.car.y)
        )

    # ==================================================
    # RENDER
    # ==================================================
    def draw(self, screen):
        screen.fill((35, 120, 35))

        self.road.draw(
            screen, self.camera,
            show_control_points=False,
            show_centerline=False
        )

        self.car.draw(screen, self.camera)
        
        # ĐÃ XÓA: Phần code self.car.camera.render(...) và draw_fov(...) ở đây
    # ==================================================
    # RENDER UI (Vẽ text sau khi đã chụp camera)
    # ==================================================
    def draw_ui(self, screen):
        if self.show_debug:
            self._draw_debug(screen)

        if self.done:
            font = pygame.font.SysFont("consolas", 30)
            text = font.render("OUT OF ROAD", True, (255, 0, 0))
            screen.blit(text, (500, 50))
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
    # CHECK IF CAR IS OUT OF ROAD (CHO RL SAU NÀY)
    # ==================================================
    def is_car_out_of_road(self):

        corners = self.car.get_corners()

        centerline = self.road.get_centerline()

        if len(centerline) < 2:
            return False

        half_width = self.road.width / 2

        for x, y in corners:

            point = pygame.Vector2(x, y)

            min_distance = float("inf")

            for center in centerline:

                distance = point.distance_to(center)

                if distance < min_distance:
                    min_distance = distance

            if min_distance > half_width:
                return True

        return False
    # ==================================================
    # HELPER: point in polygon (cho RL SAU NÀY)
    # ==================================================
    def _point_in_polygon(self, point, polygon):

        inside = False

        x = point.x
        y = point.y

        j = len(polygon) - 1

        for i in range(len(polygon)):

            xi = polygon[i].x
            yi = polygon[i].y

            xj = polygon[j].x
            yj = polygon[j].y

            intersect = (
                (yi > y) != (yj > y)
                and
                x < (
                    (xj - xi)
                    * (y - yi)
                    / (yj - yi + 1e-12)
                    + xi
                )
            )

            if intersect:
                inside = not inside

            j = i

        return inside