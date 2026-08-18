import pygame

from env.env import CarEnv
from process.path_processor import PathProcessor


WIDTH = 1400
HEIGHT = 900
FPS = 60


pygame.init()

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Path Processing Test")

clock = pygame.time.Clock()


# ==========================
# ENV
# ==========================

env = CarEnv(
    width=WIDTH,
    height=HEIGHT
)

env.reset()

processor = PathProcessor(
    num_points=15
)


# ==========================
# LOOP
# ==========================

running = True

while running:

    clock.tick(FPS)

    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:

            if event.key == pygame.K_SPACE:
                env.reset()

    # ==========================
    # INPUT
    # ==========================

    keys = pygame.key.get_pressed()

    steering = 0.0
    throttle = 0.0

    if keys[pygame.K_LEFT]:
        steering = -1.0

    if keys[pygame.K_RIGHT]:
        steering = 1.0

    if keys[pygame.K_UP]:
        throttle = 1.0

    if keys[pygame.K_DOWN]:
        throttle = -1.0

    action = (
        steering,
        throttle
    )

    env.step(action)

    # ==========================
    # GET PATH
    # ==========================

    centerline = env.road.get_centerline()

    path = processor.process(
        centerline,
        env.car.x,
        env.car.y,
        env.car.heading
    )

    # ==========================
    # DRAW
    # ==========================

    env.world.draw(screen)

    # --------------------------
    # Draw processed path
    # --------------------------

    for point in path:

        # Vehicle -> World
        cos_h = pygame.math.Vector2(
            1, 0
        ).rotate_rad(
            env.car.heading
        )

        sin_h = pygame.math.Vector2(
            0, 1
        ).rotate_rad(
            env.car.heading
        )

        world_x = (
            env.car.x
            + point.x * cos_h.x
            + point.y * sin_h.x
        )

        world_y = (
            env.car.y
            + point.x * cos_h.y
            + point.y * sin_h.y
        )

        screen_pos = env.world.camera.world_to_screen(
            pygame.Vector2(
                world_x,
                world_y
            )
        )

        pygame.draw.circle(
            screen,
            (255, 0, 0),
            screen_pos,
            5
        )

    # ==========================
    # DEBUG
    # ==========================

    font = pygame.font.SysFont(
        "consolas",
        18
    )

    info = [
        f"Car: ({env.car.x:.1f}, {env.car.y:.1f})",
        f"Heading: {env.car.heading:.2f}",
        f"Path points: {len(path)}",
    ]

    y = 10

    for text in info:

        surface = font.render(
            text,
            True,
            (0, 0, 0)
        )

        screen.blit(
            surface,
            (10, y)
        )

        y += 22

    pygame.display.flip()


pygame.quit()