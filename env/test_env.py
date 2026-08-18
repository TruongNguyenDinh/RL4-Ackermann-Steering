import pygame

from env.env import CarEnv


WIDTH = 1400
HEIGHT = 900
FPS = 60


pygame.init()

screen = pygame.display.set_mode(
    (WIDTH, HEIGHT)
)

pygame.display.set_caption(
    "CarEnv Test"
)

clock = pygame.time.Clock()


# ==========================
# ENV
# ==========================

env = CarEnv(
    width=WIDTH,
    height=HEIGHT
)

state = env.reset()


# ==========================
# LOOP
# ==========================

running = True

while running:

    dt = clock.tick(FPS) / 1000.0

    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:

            if event.key == pygame.K_SPACE:
                state = env.reset()

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

    # ==========================
    # ENV STEP
    # ==========================

    state, reward, done = env.step(
        action
    )

    # ==========================
    # AUTO RESET
    # ==========================

    if done:
        state = env.reset()

    # ==========================
    # RENDER
    # ==========================

    env.world.draw(screen)

    # ==========================
    # DEBUG
    # ==========================

    font = pygame.font.SysFont(
        "consolas",
        18
    )

    info = [
        f"Reward: {reward:.3f}",
        f"Done: {done}",
        f"State: {state}",
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