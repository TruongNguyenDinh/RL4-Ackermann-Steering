import pygame
import numpy as np
import torch

from env.env import CarEnv
from rl.agent import SACAgent


# ==================================================
# CONFIG
# ==================================================

WIDTH = 1400
HEIGHT = 900

FPS = 60

STATE_DIM = 22
ACTION_DIM = 2

MODEL_PATH = "model/best_model_v2.1.pth"


# ==================================================
# INIT PYGAME
# ==================================================

pygame.init()

screen = pygame.display.set_mode(
    (WIDTH, HEIGHT)
)

pygame.display.set_caption(
    "SAC Self-Driving - V2 Test"
)

clock = pygame.time.Clock()

font = pygame.font.SysFont(
    "consolas",
    18
)


# ==================================================
# DEVICE
# ==================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("=" * 60)
print("SAC Self-Driving Model Test")
print("=" * 60)

print(
    f"Device      : {device}"
)

print(
    f"Model       : {MODEL_PATH}"
)

print(
    f"State dim   : {STATE_DIM}"
)

print(
    f"Action dim  : {ACTION_DIM}"
)

if torch.cuda.is_available():

    print(
        f"GPU         : "
        f"{torch.cuda.get_device_name(0)}"
    )

print("=" * 60)


# ==================================================
# ENVIRONMENT
# ==================================================

env = CarEnv(
    width=WIDTH,
    height=HEIGHT,
)


# ==================================================
# AGENT
# ==================================================

agent = SACAgent(
    state_dim=STATE_DIM,
    action_dim=ACTION_DIM,
    device=device,
)


# ==================================================
# LOAD MODEL
# ==================================================

print(
    "Loading model..."
)

agent.load(
    MODEL_PATH
)

print(
    "Model loaded."
)

print("=" * 60)


# ==================================================
# RESET
# ==================================================

state = env.reset()

state = np.asarray(
    state,
    dtype=np.float32
)


if state.shape != (
    STATE_DIM,
):

    raise ValueError(
        f"Invalid state shape: "
        f"{state.shape}, "
        f"expected ({STATE_DIM},)"
    )


# ==================================================
# VARIABLES
# ==================================================

running = True

episode_reward = 0.0

episode_steps = 0

episode = 1

last_action = np.zeros(
    ACTION_DIM,
    dtype=np.float32
)


# ==================================================
# MAIN LOOP
# ==================================================

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

                episode += 1

                episode_reward = 0.0

                episode_steps = 0

                last_action = np.zeros(
                    ACTION_DIM,
                    dtype=np.float32
                )

                print(
                    f"Episode {episode} reset"
                )


            # ------------------------------------------
            # EXIT
            # ------------------------------------------

            if event.key == pygame.K_ESCAPE:

                running = False


    # ==================================================
    # MODEL ACTION
    # ==================================================

    action = agent.select_action(
        state,
        evaluate=True,
    )

    action = np.asarray(
        action,
        dtype=np.float32
    )

    last_action = action.copy()


    # ==================================================
    # ENV STEP
    # ==================================================

    next_state, reward, done = env.step(
        action
    )

    next_state = np.asarray(
        next_state,
        dtype=np.float32
    )

    reward = float(
        reward
    )

    state = next_state

    episode_reward += reward

    episode_steps += 1


    # ==================================================
    # AUTO RESET
    # ==================================================

    if done:

        print(
            f"Episode: {episode:4d} | "
            f"Steps: {episode_steps:5d} | "
            f"Reward: {episode_reward:9.3f}"
        )

        state = env.reset()

        state = np.asarray(
            state,
            dtype=np.float32
        )

        episode += 1

        episode_reward = 0.0

        episode_steps = 0


    # ==================================================
    # RENDER WORLD
    #
    # KHÔNG VẼ RL PATH Ở WORLD
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
    # 🔴 2D WAYPOINTS
    # ==================================================

    camera_image = env.camera_image.copy()


    for x, y in env.waypoints_2d:

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
    # CAMERA PREVIEW
    # ==================================================

    camera_x = WIDTH - 500
    camera_y = 20

    camera_width = 480
    camera_height = 270


    pygame.draw.rect(
        screen,
        (20, 20, 20),
        (
            camera_x - 5,
            camera_y - 5,
            camera_width + 10,
            camera_height + 10
        )
    )


    screen.blit(
        camera_image,
        (
            camera_x,
            camera_y
        )
    )


    camera_label = font.render(
        "VEHICLE CAMERA",
        True,
        (255, 255, 255)
    )

    screen.blit(
        camera_label,
        (
            camera_x + 10,
            camera_y + 10
        )
    )


    # ==================================================
    # 3D PATH
    #
    # 🟢 OFFICIAL PATH
    # ==================================================

    path_surface = env.path_surface.copy()


    for x, y in env.official_path:

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
    # 3D PATH PREVIEW
    # ==================================================

    path_x = WIDTH - 500
    path_y = 330

    path_width = 500
    path_height = 400


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


    screen.blit(
        path_surface,
        (
            path_x,
            path_y
        )
    )


    path_label = font.render(
        "3D PATH / OFFICIAL",
        True,
        (255, 255, 255)
    )

    screen.blit(
        path_label,
        (
            path_x + 10,
            path_y + 10
        )
    )


    # ==================================================
    # DEBUG
    # ==================================================

    # --------------------------------------------------
    # Lateral error
    # --------------------------------------------------

    if len(env.processed_path) > 0:

        lateral_error = abs(
            env.processed_path[0].y
        )

    else:

        lateral_error = 0.0


    # --------------------------------------------------
    # Steering
    # --------------------------------------------------

    steering = env.car.steering


    # --------------------------------------------------
    # Velocity
    # --------------------------------------------------

    velocity = env.car.velocity


    info = [

        "=== MODEL TEST ===",

        f"Episode       : {episode}",
        f"Steps         : {episode_steps}",

        "",

        f"Reward        : {episode_reward:8.3f}",

        "",

        "=== VEHICLE ===",

        f"Lateral Error : {lateral_error:8.3f}",
        f"Steering      : {steering:8.3f}",
        f"Velocity      : {velocity:8.3f}",

        "",

        "=== PATH ===",

        f"2D Path       : {len(env.waypoints_2d)}",
        f"3D Path       : {len(env.path_3d)}",
        f"Official Path : {len(env.official_path)}",
        f"RL Path       : {len(env.processed_path)}",

        "",

        "=== ACTION ===",

        f"Steering Act  : {last_action[0]:8.3f}",
        f"Throttle Act  : {last_action[1]:8.3f}",

    ]


    # ==================================================
    # STATE DISPLAY
    # ==================================================

    for i in range(10):

        x_index = i * 2

        y_index = x_index + 1


        if y_index < len(state):

            info.append(
                f"P{i:02d}: "
                f"x={state[x_index]:7.2f} "
                f"y={state[y_index]:7.2f}"
            )


    # ==================================================
    # DRAW INFO
    # ==================================================

    x = 10
    y = 10

    for text in info:

        surface = font.render(
            text,
            True,
            (20, 20, 20)
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
    # DISPLAY
    # ==================================================

    pygame.display.flip()


# ==================================================
# CLOSE
# ==================================================

env.close()

pygame.quit()

print()
print("=" * 60)
print("Model test finished")
print("=" * 60)