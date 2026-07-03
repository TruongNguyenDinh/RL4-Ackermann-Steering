import pygame

from world.world import World
from world.road import Road
from vehicle.car import Car   # giả sử bạn có class này

WIDTH = 1400
HEIGHT = 900

pygame.init()

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Road Generator")

clock = pygame.time.Clock()

# ==========================
# INIT COMPONENTS
# ==========================
road = Road(
    width=120,
    segment_length=180,
    samples_per_segment=20
)

car = Car(
    x=WIDTH // 2,
    y=HEIGHT // 2,
)

world = World(road, car)

world.reset()

running = True

# action mặc định (steering, throttle)
action = (0.0, 0.0)

while running:

    dt = clock.tick(60) / 1000

    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            running = False

        # reset world
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                world.reset()

    # ==========================
    # INPUT TEST (tạm thời)
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

    action = (steering, throttle)

    # ==========================
    # UPDATE WORLD
    # ==========================
    world.update(dt, action)

    # ==========================
    # DRAW
    # ==========================
    world.draw(screen)

    pygame.display.flip()

pygame.quit()