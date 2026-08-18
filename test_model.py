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

MODEL_PATH = "model/best_model.pth"


# ==================================================
# INIT PYGAME
# ==================================================

pygame.init()

screen = pygame.display.set_mode(
    (WIDTH, HEIGHT)
)

pygame.display.set_caption(
    "SAC Best Model Test"
)

clock = pygame.time.Clock()


# ==================================================
# ENV
# ==================================================

env = CarEnv(
    width=WIDTH,
    height=HEIGHT,
)


# ==================================================
# AGENT
# ==================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

agent = SACAgent(
    state_dim=STATE_DIM,
    action_dim=ACTION_DIM,
    device=device,
)

agent.load(
    MODEL_PATH
)

print("=" * 60)
print("SAC MODEL TEST")
print("=" * 60)
print(f"Device: {device}")
print(f"Model: {MODEL_PATH}")
print("=" * 60)


# ==================================================
# RESET
# ==================================================

state = env.reset()

state = np.asarray(
    state,
    dtype=np.float32
)


# ==================================================
# LOOP
# ==================================================

running = True

episode_reward = 0.0
episode_steps = 0
episode = 1

while running:

    clock.tick(FPS)

    # ==================================================
    # EVENTS
    # ==================================================

    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            running = False

        # SPACE -> reset episode
        if event.type == pygame.KEYDOWN:

            if event.key == pygame.K_SPACE:

                print(
                    f"Episode {episode} reset"
                )

                state = env.reset()

                state = np.asarray(
                    state,
                    dtype=np.float32
                )

                episode_reward = 0.0
                episode_steps = 0

    # ==================================================
    # AI ACTION
    # ==================================================

    action = agent.select_action(
        state,
        evaluate=True,
    )

    action = np.asarray(
        action,
        dtype=np.float32
    )

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

    state = next_state

    episode_reward += reward
    episode_steps += 1

    # ==================================================
    # RENDER
    # ==================================================

    env.world.draw(
        screen
    )

    # ==================================================
    # DEBUG HUD
    # ==================================================

    font = pygame.font.SysFont(
        "consolas",
        18
    )

    info = [
        f"Episode: {episode}",
        f"Steps: {episode_steps}",
        f"Reward: {episode_reward:.2f}",
        f"Steering: {action[0]:.3f}",
        f"Throttle: {action[1]:.3f}",
        f"Done: {done}",
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
            (500, y)
        )

        y += 22

    pygame.display.flip()

    # ==================================================
    # EPISODE END
    # ==================================================

    if done:

        print(
            f"Episode {episode} | "
            f"Steps: {episode_steps} | "
            f"Reward: {episode_reward:.3f}"
        )

        episode += 1

        state = env.reset()

        state = np.asarray(
            state,
            dtype=np.float32
        )

        episode_reward = 0.0
        episode_steps = 0


pygame.quit()