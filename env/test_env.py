import pygame
import numpy as np

from env.env import CarEnv


# ==================================================
# CONFIG
# ==================================================

WIDTH = 1400
HEIGHT = 900

FPS = 60

STATE_DIM = 22


# ==================================================
# COLORS
# ==================================================

RED = (255, 0, 0)
GREEN = (0, 255, 0)

WHITE = (255, 255, 255)
BLACK = (20, 20, 20)

WARNING_RED = (220, 30, 30)


# ==================================================
# INIT PYGAME
# ==================================================

pygame.init()

screen = pygame.display.set_mode(
    (WIDTH, HEIGHT)
)

pygame.display.set_caption(
    "CarEnv Test - Vision -> Official Path -> RL"
)

clock = pygame.time.Clock()

font = pygame.font.SysFont(
    "consolas",
    18
)


# ==================================================
# ENVIRONMENT
# ==================================================

env = CarEnv(
    width=WIDTH,
    height=HEIGHT,
)


# ==================================================
# RESET
# ==================================================

state = env.reset()

state = np.asarray(
    state,
    dtype=np.float32
)


# ==================================================
# INITIAL DEBUG
# ==================================================

print("=" * 60)
print("CarEnv Test")
print("=" * 60)

print(
    f"State shape : {state.shape}"
)

print(
    f"State dim   : {len(state)}"
)

print(
    f"Expected    : {STATE_DIM}"
)

print(
    f"2D path     : {len(env.waypoints_2d)}"
)

print(
    f"3D path     : {len(env.path_3d)}"
)

print(
    f"Official    : {len(env.official_path)}"
)

print(
    f"RL path     : {len(env.processed_path)}"
)

print("=" * 60)


# ==================================================
# STATE CHECK
# ==================================================

if len(state) != STATE_DIM:

    raise ValueError(
        f"Invalid state dimension: "
        f"{len(state)} "
        f"(expected {STATE_DIM})"
    )


# ==================================================
# LOOP
# ==================================================

running = True


while running:

    # ==================================================
    # CLOCK
    # ==================================================

    clock.tick(FPS)


    # ==================================================
    # EVENTS
    # ==================================================

    for event in pygame.event.get():

        if event.type == pygame.QUIT:

            running = False


        if event.type == pygame.KEYDOWN:

            # ------------------------------------------
            # RESET
            # ------------------------------------------

            if event.key == pygame.K_SPACE:

                state = env.reset()

                state = np.asarray(
                    state,
                    dtype=np.float32
                )

                print()
                print("=" * 60)
                print("Environment reset")

                print(
                    f"State dim   : "
                    f"{len(state)}"
                )

                print(
                    f"2D path     : "
                    f"{len(env.waypoints_2d)}"
                )

                print(
                    f"3D path     : "
                    f"{len(env.path_3d)}"
                )

                print(
                    f"Official    : "
                    f"{len(env.official_path)}"
                )

                print(
                    f"RL path     : "
                    f"{len(env.processed_path)}"
                )

                print("=" * 60)


    # ==================================================
    # INPUT
    # ==================================================

    keys = pygame.key.get_pressed()

    steering = 0.0
    throttle = 0.0


    # --------------------------------------------------
    # Steering
    # --------------------------------------------------

    if keys[pygame.K_LEFT]:

        steering = -1.0

    elif keys[pygame.K_RIGHT]:

        steering = 1.0


    # --------------------------------------------------
    # Throttle
    # --------------------------------------------------

    if keys[pygame.K_UP]:

        throttle = 1.0

    elif keys[pygame.K_DOWN]:

        throttle = -1.0


    action = (
        steering,
        throttle
    )


    # ==================================================
    # ENV STEP
    # ==================================================

    state, reward, done = env.step(
        action
    )

    state = np.asarray(
        state,
        dtype=np.float32
    )


    # ==================================================
    # STATE CHECK
    # ==================================================

    if len(state) != STATE_DIM:

        print(
            f"WARNING: "
            f"state dim = {len(state)}"
        )


    # ==================================================
    # RENDER WORLD
    #
    # KHÔNG VẼ RL PATH Ở ĐÂY
    # ==================================================

    screen.fill(
        (240, 240, 240)
    )

    env.world.draw(
        screen
    )


    # ==================================================
    # WORLD UI
    # ==================================================

    env.world.draw_ui(
        screen
    )


    # ==================================================
    # VEHICLE CAMERA FOV
    # ==================================================

    env.vehicle_camera.draw_fov(
        screen,
        env.car,
        env.world.camera.world_to_screen
    )


    # ==================================================
    # VEHICLE CAMERA
    #
    # 2D WAYPOINTS = ĐỎ
    #
    # Đây là path được lấy trực tiếp từ
    # LaneDetector.
    # ==================================================

    camera_image = env.camera_image


    if camera_image is not None:

        # --------------------------------------------------
        # Vẽ 2D waypoints
        # --------------------------------------------------

        for x, y in env.waypoints_2d:

            pygame.draw.circle(
                camera_image,
                RED,
                (
                    int(x),
                    int(y)
                ),
                5
            )


        # --------------------------------------------------
        # Camera preview
        # --------------------------------------------------

        camera_x = WIDTH - 500
        camera_y = 20

        camera_width = 480
        camera_height = 270


        # --------------------------------------------------
        # Border
        # --------------------------------------------------

        pygame.draw.rect(
            screen,
            BLACK,
            (
                camera_x - 5,
                camera_y - 5,
                camera_width + 10,
                camera_height + 10
            )
        )


        # --------------------------------------------------
        # Image
        # --------------------------------------------------

        screen.blit(
            camera_image,
            (
                camera_x,
                camera_y
            )
        )


        # --------------------------------------------------
        # Label
        # --------------------------------------------------

        camera_label = font.render(
            "VEHICLE CAMERA",
            True,
            WHITE
        )

        screen.blit(
            camera_label,
            (
                camera_x + 10,
                camera_y + 10
            )
        )


        # --------------------------------------------------
        # Camera debug
        # --------------------------------------------------

        camera_info = [

            f"2D Waypoints : "
            f"{len(env.waypoints_2d)}",

        ]


        info_y = camera_y + 40


        for text in camera_info:

            surface = font.render(
                text,
                True,
                WHITE
            )

            screen.blit(
                surface,
                (
                    camera_x + 10,
                    info_y
                )
            )

            info_y += 21


    # ==================================================
    # 3D PATH
    #
    # Official Path = XANH
    #
    # env.path_surface đã được env tạo.
    # Ở đây chỉ hiển thị nó.
    # ==================================================

    path_x = WIDTH - 500
    path_y = 330

    path_width = 500
    path_height = 400


    # --------------------------------------------------
    # Border
    # --------------------------------------------------

    pygame.draw.rect(
        screen,
        BLACK,
        (
            path_x - 5,
            path_y - 5,
            path_width + 10,
            path_height + 10
        )
    )


    # --------------------------------------------------
    # 3D Surface
    # --------------------------------------------------

    if env.path_surface is not None:

        screen.blit(
            env.path_surface,
            (
                path_x,
                path_y
            )
        )


    # --------------------------------------------------
    # Label
    # --------------------------------------------------

    path_label = font.render(
        "3D PATH / OFFICIAL PATH",
        True,
        WHITE
    )

    screen.blit(
        path_label,
        (
            path_x + 10,
            path_y + 10
        )
    )


    # ==================================================
    # DEBUG INFORMATION
    # ==================================================

    state_dim = len(state)

    path2d_count = len(
        env.waypoints_2d
    )

    path3d_count = len(
        env.path_3d
    )

    official_count = len(
        env.official_path
    )

    rl_path_count = len(
        env.processed_path
    )

    velocity = env.car.velocity

    steering_angle = env.car.steering


    # ==================================================
    # DEBUG TEXT
    # ==================================================

    info = [

        "=== CAR ENV ===",

        f"State Dim : "
        f"{state_dim} / {STATE_DIM}",

        "",

        f"2D Path   : "
        f"{path2d_count}",

        f"3D Path   : "
        f"{path3d_count}",

        f"Official  : "
        f"{official_count}",

        f"RL Path   : "
        f"{rl_path_count}",

        "",

        f"Reward    : "
        f"{reward:8.3f}",

        f"Done      : "
        f"{done}",

        "",

        f"Velocity  : "
        f"{velocity:8.2f}",

        f"Steering  : "
        f"{steering_angle:8.3f}",

        "",

        "RL STATE:",

    ]


    # ==================================================
    # SHOW RL STATE
    #
    # 20 path coordinates
    # + velocity
    # + steering
    #
    # = 22 dimensions
    # ==================================================

    for i in range(10):

        x_index = i * 2

        y_index = x_index + 1


        if y_index < len(state):

            info.append(
                f"P{i:02d}: "
                f"x={state[x_index]:7.3f} "
                f"y={state[y_index]:7.3f}"
            )


    # ==================================================
    # DRAW DEBUG TEXT
    # ==================================================

    x = 10
    y = 10


    for text in info:

        surface = font.render(
            text,
            True,
            BLACK
        )

        screen.blit(
            surface,
            (
                x,
                y
            )
        )

        y += 21


    # ==================================================
    # DONE MESSAGE
    # ==================================================

    if done:

        message = font.render(
            "DONE - PRESS SPACE TO RESET",
            True,
            WARNING_RED
        )

        screen.blit(
            message,
            (
                10,
                HEIGHT - 35
            )
        )


    # ==================================================
    # DISPLAY
    # ==================================================

    pygame.display.flip()


# ==================================================
# CLOSE
# ==================================================

env.close()

pygame.quit()