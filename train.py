import os
import random

import numpy as np
import torch

from env.env import CarEnv
from rl.agent import SACAgent
from rl.replay_buffer import ReplayBuffer


# ==================================================
# CONFIG
# ==================================================

WIDTH = 1400
HEIGHT = 900

STATE_DIM = 22
ACTION_DIM = 2

EPISODES = 3000
MAX_STEPS = 2000

BUFFER_SIZE = 100_000
BATCH_SIZE = 256

WARMUP_STEPS = 5_000

UPDATE_AFTER = 1
UPDATE_EVERY = 1

SAVE_EVERY = 100

MODEL_DIR = "model"

SEED = 42


# ==================================================
# SEED
# ==================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ==================================================
# DEVICE
# ==================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 60)
print("SAC Self-Driving Training")
print("=" * 60)
print(f"Device: {device}")

if torch.cuda.is_available():
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
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
# REPLAY BUFFER
# ==================================================

replay_buffer = ReplayBuffer(
    state_dim=STATE_DIM,
    action_dim=ACTION_DIM,
    capacity=BUFFER_SIZE,
)


# ==================================================
# MODEL DIRECTORY
# ==================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True,
)


# ==================================================
# TRAINING
# ==================================================

total_steps = 0

best_reward = -float("inf")


for episode in range(1, EPISODES + 1):

    # ----------------------------------------------
    # Reset environment
    # ----------------------------------------------

    state = env.reset()

    state = np.asarray(
        state,
        dtype=np.float32,
    )

    episode_reward = 0.0
    episode_steps = 0

    # ----------------------------------------------
    # Episode
    # ----------------------------------------------

    for step in range(MAX_STEPS):

        # ==========================================
        # Action
        # ==========================================

        if total_steps < WARMUP_STEPS:

            action = np.random.uniform(
                -1.0,
                1.0,
                size=ACTION_DIM,
            ).astype(np.float32)

        else:

            action = agent.select_action(
                state,
                evaluate=False,
            )

            action = np.asarray(
                action,
                dtype=np.float32,
            )

        # ==========================================
        # Environment
        # ==========================================

        next_state, reward, done = env.step(
            action
        )

        next_state = np.asarray(
            next_state,
            dtype=np.float32,
        )

        reward = float(reward)

        done = bool(done)

        # ==========================================
        # Replay Buffer
        # ==========================================

        replay_buffer.add(
            state,
            action,
            reward,
            next_state,
            done,
        )

        # ==========================================
        # Update
        # ==========================================

        if (
            total_steps >= UPDATE_AFTER
            and len(replay_buffer) >= BATCH_SIZE
            and total_steps % UPDATE_EVERY == 0
        ):

            info = agent.update(
                replay_buffer,
                batch_size=BATCH_SIZE,
            )

        # ==========================================
        # State
        # ==========================================

        state = next_state

        episode_reward += reward
        episode_steps += 1
        total_steps += 1

        # ==========================================
        # Episode finished
        # ==========================================

        if done:
            break

    # ==================================================
    # LOG
    # ==================================================

    print(
        f"Episode: {episode:4d} | "
        f"Steps: {episode_steps:4d} | "
        f"Reward: {episode_reward:9.3f} | "
        f"Buffer: {len(replay_buffer):6d}"
    )

    # ==================================================
    # BEST MODEL
    # ==================================================

    if episode_reward > best_reward:

        best_reward = episode_reward

        best_path = os.path.join(
            MODEL_DIR,
            "best_model.pth",
        )

        agent.save(
            best_path
        )

        print(
            f"  -> Best model saved "
            f"(reward={best_reward:.3f})"
        )

    # ==================================================
    # CHECKPOINT
    # ==================================================

    if episode % SAVE_EVERY == 0:

        checkpoint_path = os.path.join(
            MODEL_DIR,
            f"checkpoint_{episode}.pth",
        )

        agent.save(
            checkpoint_path
        )

        print(
            f"  -> Checkpoint saved: "
            f"{checkpoint_path}"
        )


# ==================================================
# FINAL MODEL
# ==================================================

final_path = os.path.join(
    MODEL_DIR,
    "final_model.pth",
)

agent.save(
    final_path
)

print()
print("=" * 60)
print("Training finished")
print(f"Best reward : {best_reward:.3f}")
print(f"Final model : {final_path}")
print("=" * 60)