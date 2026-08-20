import pygame

from world.world import World
from world.road import Road
from vehicle.car import Car
from vehicle.camera import VehicleCamera

from vision.lane_detection import LaneDetector
from vision.path_extraction import PathExtractor

from engine_3D.path_3d import Path3DRenderer
from engine_3D.ground_projection import GroundProjection


# ==================================================
# WINDOW
# ==================================================

WIDTH = 1400
HEIGHT = 900

pygame.init()

screen = pygame.display.set_mode(
    (WIDTH, HEIGHT)
)

pygame.display.set_caption(
    "Vision -> 2D Waypoints -> 3D Path -> Official Path"
)

clock = pygame.time.Clock()

font = pygame.font.SysFont(
    "consolas",
    20
)


# ==================================================
# WORLD
# ==================================================

road = Road(
    width=70,
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
    view_distance=500,
)


# ==================================================
# 2D LANE DETECTOR
# ==================================================

lane_detector = LaneDetector(
    num_waypoints=20,
    y_start=245,
    y_end=80,
)


# ==================================================
# 3D GROUND PROJECTION
# ==================================================

ground_projection = GroundProjection(
    image_width=480,
    image_height=270,
    fov=90,

    camera_height=80,

    horizon_y=40,
)


# ==================================================
# 3D PATH RENDERER
# ==================================================

path_renderer = Path3DRenderer(
    width=500,
    height=400,

    fov=90,

    camera_height=80,

    lane_width=70,

    horizon_ratio=0.28,
)


# ==================================================
# PATH EXTRACTOR
# ==================================================

path_extractor = PathExtractor(
    num_waypoints=20,

    # 3D image coordinates
    y_start=390,
    y_end=100,
)


# ==================================================
# 3D SURFACE
# ==================================================

path_surface = pygame.Surface(
    (
        500,
        400
    )
)


# ==================================================
# RESET
# ==================================================

world.reset()


# ==================================================
# MAIN LOOP
# ==================================================

running = True

action = (
    0.0,
    0.0
)


while running:

    # ==================================================
    # 1. CLOCK
    # ==================================================

    dt = clock.tick(60) / 1000


    # ==================================================
    # 2. EVENTS
    # ==================================================

    for event in pygame.event.get():

        if event.type == pygame.QUIT:

            running = False

        if event.type == pygame.KEYDOWN:

            if event.key == pygame.K_SPACE:

                world.reset()


    # ==================================================
    # 3. INPUT
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


    action = (
        steering,
        throttle
    )


    # ==================================================
    # 4. UPDATE WORLD
    # ==================================================

    world.update(
        dt,
        action
    )


    # ==================================================
    # 5. DRAW WORLD
    # ==================================================

    world.draw(
        screen
    )


    # ==================================================
    # 6. VEHICLE CAMERA
    # ==================================================

    camera_image = vehicle_camera.render(
        screen,
        car,
        world.camera
    )


    # ==================================================
    # 7. 2D LANE DETECTION
    # ==================================================

    waypoints_2d = lane_detector.extract_path(
        camera_image
    )


    # ==================================================
    # 8. 2D -> 3D
    # ==================================================

    path_3d = ground_projection.project_path(
        waypoints_2d
    )


    # ==================================================
    # 9. DRAW 2D WAYPOINTS
    # ==================================================

    for x, y in waypoints_2d:

        pygame.draw.circle(
            camera_image,
            (255, 0, 0),
            (
                int(x),
                int(y)
            ),
            5
        )


    # ==================================================
    # 10. RENDER 3D PATH
    # ==================================================

    path_surface.fill(
        (30, 30, 30)
    )

    path_renderer.draw(
        path_surface,
        path_3d
    )


    # ==================================================
    # 11. EXTRACT OFFICIAL PATH
    #
    # Lấy path từ chính ảnh 3D.
    # ==================================================

    official_path = path_extractor.extract_path(
        path_surface
    )


    # ==================================================
    # 12. DRAW OFFICIAL PATH
    #
    # XANH = path được PathExtractor lấy lại
    # ĐỎ  = path gốc từ 3D renderer
    # ==================================================

    for x, y in official_path:

        pygame.draw.circle(
            path_surface,
            (0, 255, 0),
            (
                int(x),
                int(y)
            ),
            5
        )


    # ==================================================
    # 13. DRAW WORLD UI
    # ==================================================

    world.draw_ui(
        screen
    )


    # ==================================================
    # 14. DRAW VEHICLE CAMERA FOV
    # ==================================================

    vehicle_camera.draw_fov(
        screen,
        car,
        world.camera.world_to_screen
    )


    # ==================================================
    # 15. 3D PATH VIEW
    # ==================================================

    path_x = 20
    path_y = 20

    path_width = 500
    path_height = 400


    # --------------------------------------------------
    # Border
    # --------------------------------------------------

    pygame.draw.rect(
        screen,
        (20, 20, 20),
        (
            path_x - 5,
            path_y - 5,
            path_width + 10,
            path_height + 10
        )
    )


    # --------------------------------------------------
    # 3D image
    # --------------------------------------------------

    screen.blit(
        path_surface,
        (
            path_x,
            path_y
        )
    )


    # --------------------------------------------------
    # Label
    # --------------------------------------------------

    label_3d = font.render(
        "3D PATH",
        True,
        (255, 255, 255)
    )

    screen.blit(
        label_3d,
        (
            path_x + 10,
            path_y + 10
        )
    )


    # ==================================================
    # 16. VEHICLE CAMERA PREVIEW
    # ==================================================

    preview_x = WIDTH - 500
    preview_y = 20

    preview_width = 480
    preview_height = 270


    # --------------------------------------------------
    # Border
    # --------------------------------------------------

    pygame.draw.rect(
        screen,
        (20, 20, 20),
        (
            preview_x - 5,
            preview_y - 5,
            preview_width + 10,
            preview_height + 10
        )
    )


    # --------------------------------------------------
    # Camera image
    # --------------------------------------------------

    screen.blit(
        camera_image,
        (
            preview_x,
            preview_y
        )
    )


    # --------------------------------------------------
    # Label
    # --------------------------------------------------

    camera_label = font.render(
        "VEHICLE CAMERA",
        True,
        (255, 255, 255)
    )

    screen.blit(
        camera_label,
        (
            preview_x + 10,
            preview_y + 10
        )
    )


    # ==================================================
    # 17. DEBUG INFO
    # ==================================================

    waypoint_text = font.render(
        f"2D Waypoints : {len(waypoints_2d)}",
        True,
        (255, 255, 255)
    )

    screen.blit(
        waypoint_text,
        (
            preview_x + 10,
            preview_y + 40
        )
    )


    path3d_text = font.render(
        f"3D Points    : {len(path_3d)}",
        True,
        (255, 255, 255)
    )

    screen.blit(
        path3d_text,
        (
            preview_x + 10,
            preview_y + 65
        )
    )


    official_text = font.render(
        f"Official Path: {len(official_path)}",
        True,
        (0, 255, 0)
    )

    screen.blit(
        official_text,
        (
            preview_x + 10,
            preview_y + 90
        )
    )


    # ==================================================
    # 18. DISPLAY
    # ==================================================

    pygame.display.flip()


# ==================================================
# EXIT
# ==================================================

pygame.quit()