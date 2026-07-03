import pygame


class Camera:
    """
    Camera follow mượt (lerp) - cực quan trọng cho RL + visual ổn định
    """
    def __init__(self, width, height):
        self.offset = pygame.Vector2(0, 0)
        self.width = width
        self.height = height

        self.smoothness = 0.08  # càng nhỏ càng mượt

    def update(self, target_pos):
        target = pygame.Vector2(target_pos)

        desired = pygame.Vector2(
            self.width / 2 - target.x,
            self.height / 2 - target.y
        )

        self.offset += (desired - self.offset) * self.smoothness

    def world_to_screen(self, pos):
        return pygame.Vector2(pos) + self.offset


# ==================================================