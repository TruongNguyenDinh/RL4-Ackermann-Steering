import pygame

from .car import Car


WIDTH = 1000
HEIGHT = 700

pygame.init()

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Bicycle Model Test")

clock = pygame.time.Clock()
font = pygame.font.SysFont("consolas", 20)

car = Car(
    x=WIDTH // 2,
    y=HEIGHT // 2,
)

running = True

while running:

    dt = clock.tick(60) / 1000

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # ==========================
    # Steering theo chuột
    # ==========================


    # ==========================
    # Keyboard
    # ==========================
    throttle = 0.0

    keys = pygame.key.get_pressed()

    if keys[pygame.K_UP]:
        throttle = 1.0          # tiến

    elif keys[pygame.K_DOWN]:
        throttle = -1.0         # lùi

    mouse_x, mouse_y = pygame.mouse.get_pos()

    steering = car.steer_to_point(
        mouse_x,
        mouse_y
    )
    car.update(
        (steering, throttle),
        dt
    )

    # ==========================
    # Draw
    # ==========================
    screen.fill((240, 240, 240))

    car.draw(screen)
    # ==========================
    # Debug Info
    # ==========================

    info = [
        f"Vehicle Speed : {car.velocity:7.2f}",
        f"Steering      : {car.steering:7.2f} rad ({car.steering * 180 / 3.14159:6.2f} deg)",
        "",
        f"Front Left Angle : {car.front_left.steer_angle * 180 / 3.14159:6.2f}",
        f"Front Right Angle: {car.front_right.steer_angle * 180 / 3.14159:6.2f}",
        "",
        f"Front Left Speed : {car.front_left.speed:7.2f}",
        f"Front Right Speed: {car.front_right.speed:7.2f}",
        f"Rear Left Speed  : {car.rear_left.speed:7.2f}",
        f"Rear Right Speed : {car.rear_right.speed:7.2f}",
    ]

    y = 10

    for line in info:
        surface = font.render(line, True, (20, 20, 20))
        screen.blit(surface, (10, y))
        y += 24
    pygame.display.flip()

pygame.quit()