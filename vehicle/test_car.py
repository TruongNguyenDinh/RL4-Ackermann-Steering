import pygame

from world.world import World
from world.road import Road
from vehicle.car import Car
from vehicle.camera import VehicleCamera


WIDTH = 1400
HEIGHT = 900

pygame.init()

screen = pygame.display.set_mode(
    (WIDTH, HEIGHT)
)

pygame.display.set_caption(
    "Road Generator + Vehicle Camera"
)

clock = pygame.time.Clock()
font = pygame.font.SysFont("consolas", 20)


# ==================================================
# INIT COMPONENTS
# ==================================================

road = Road(
    width=220,
    segment_length=180,
    samples_per_segment=20
)

car = Car(
    x=WIDTH // 2,
    y=HEIGHT // 2,
)

world = World(
    road,
    car
)

# ==================================================
# VEHICLE CAMERA
# ==================================================

vehicle_camera = VehicleCamera(
    width=480,
    height=270,
    fov=90,
    view_distance=400,
)

world.reset()


# ==================================================
# MAIN LOOP
# ==================================================

running = True

action = (0.0, 0.0)


while running:

    # ==================================================
    # 1. CLOCK & DT (Bắt buộc để game không chạy quá nhanh)
    # ==================================================
    dt = clock.tick(60) / 1000

    # ==================================================
    # 2. EVENTS (Bắt buộc để không bị treo cửa sổ)
    # ==================================================
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                world.reset()

    # ==================================================
    # 3. INPUT (Điều khiển xe)
    # ==================================================
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

    # ==================================================
    # 4. UPDATE WORLD (Cập nhật vật lý)
    # ==================================================
    world.update(dt, action)

    # ==================================================
    # 5. DRAW WORLD (Chỉ vẽ đường và xe)
    # ==================================================
    world.draw(screen)

    # ==================================================
    # 6. VEHICLE CAMERA IMAGE
    # Chụp ngay bây giờ! Khi screen HOÀN TOÀN SẠCH, chưa có UI
    # ==================================================
    camera_image = vehicle_camera.render(
        screen,
        car,
        world.camera
    )

    # ==================================================
    # 7. DRAW UI & FOV
    # Bây giờ mới vẽ các thành phần phụ lên trên cùng
    # ==================================================
    world.draw_ui(screen) # Gọi hàm bạn đã tách trong world.py

    vehicle_camera.draw_fov(
        screen,
        car,
        world.camera.world_to_screen
    )

    # ==================================================
    # 8. CAMERA PREVIEW (Khung PIP)
    # ==================================================
    preview_x = WIDTH - 500
    preview_y = 20
    preview_width = 480
    preview_height = 270

    # Border
    pygame.draw.rect(
        screen,
        (20, 20, 20),
        (
            preview_x - 5,
            preview_y - 5,
            preview_width + 10,
            preview_height + 10,
        ),
    )

    # Camera image
    screen.blit(
        camera_image,
        (preview_x, preview_y)
    )

    # Label
    label = font.render("VEHICLE CAMERA", True, (255, 255, 255))
    screen.blit(
        label,
        (preview_x + 10, preview_y + 10)
    )

    # ==================================================
    # 9. DISPLAY
    # ==================================================
    pygame.display.flip()


pygame.quit()