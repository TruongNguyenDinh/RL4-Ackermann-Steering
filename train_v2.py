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


# ==================================================
# STATE
# ==================================================

# 10 waypoints
# mỗi waypoint = (x, y)
#
# 10 × 2 = 20
#
# velocity      = 1
# steering      = 1
#
# TOTAL = 22

STATE_DIM = 22

ACTION_DIM = 2


# ==================================================
# TRAINING
# ==================================================

EPISODES = 3000

MAX_STEPS = 2000


# ==================================================
# REPLAY BUFFER
# ==================================================

BUFFER_SIZE = 100_000

BATCH_SIZE = 256


# ==================================================
# WARMUP
#
# Model cũ đã được train tốt.
#
# Không cần random exploration ban đầu.
# SAC sẽ lấy action từ policy cũ.
# ==================================================

WARMUP_STEPS = 0


# ==================================================
# UPDATE
# ==================================================

UPDATE_AFTER = 1

UPDATE_EVERY = 1


# ==================================================
# SAVE
# ==================================================

SAVE_EVERY = 100


# ==================================================
# MODEL
# ==================================================

MODEL_DIR = "model"

# Model cũ
PRETRAINED_MODEL = os.path.join(
    MODEL_DIR,
    "best_model.pth"
)


# ==================================================
# VERSION
# ==================================================

VERSION = "v2.1"


# ==================================================
# SEED
# ==================================================

SEED = 42


# ==================================================
# SEED
# ==================================================

random.seed(
    SEED
)

np.random.seed(
    SEED
)

torch.manual_seed(
    SEED
)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(
        SEED
    )


# ==================================================
# DEVICE
# ==================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ==================================================
# HEADER
# ==================================================

print("=" * 60)

print(
    "SAC Self-Driving Training - V2"
)

print("=" * 60)

print(
    f"Device      : {device}"
)

print(
    f"State dim   : {STATE_DIM}"
)

print(
    f"Action dim  : {ACTION_DIM}"
)

print(
    f"Episodes    : {EPISODES}"
)

print(
    f"Max steps   : {MAX_STEPS}"
)

print(
    f"Warmup      : {WARMUP_STEPS}"
)

print(
    f"Pretrained  : {PRETRAINED_MODEL}"
)

print(
    f"Output      : *_{VERSION}.pth"
)


if torch.cuda.is_available():

    print(
        f"GPU         : "
        f"{torch.cuda.get_device_name(0)}"
    )

print("=" * 60)


# ==================================================
# MODEL DIRECTORY
# ==================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ==================================================
# ENVIRONMENT
# ==================================================

env = CarEnv(
    width=WIDTH,
    height=HEIGHT
)


# ==================================================
# CHECK ENV STATE
# ==================================================

initial_state = env.reset()

initial_state = np.asarray(
    initial_state,
    dtype=np.float32
)


print(
    f"Initial state shape: "
    f"{initial_state.shape}"
)


if initial_state.shape != (
    STATE_DIM,
):

    raise ValueError(
        "\n"
        f"Expected state shape "
        f"({STATE_DIM},), "
        f"but CarEnv returned "
        f"{initial_state.shape}\n"
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
# AGENT
# ==================================================

agent = SACAgent(
    state_dim=STATE_DIM,
    action_dim=ACTION_DIM,
    device=device,
)


# ==================================================
# LOAD OLD MODEL
# ==================================================

if not os.path.exists(
    PRETRAINED_MODEL
):

    raise FileNotFoundError(
        "\n"
        "Pretrained model not found:\n"
        f"{PRETRAINED_MODEL}\n"
        "\n"
        "Make sure the old model exists "
        "before starting V2 training."
    )


print(
    "Loading pretrained model..."
)

print(
    f"  -> {PRETRAINED_MODEL}"
)


agent.load(
    PRETRAINED_MODEL
)


print(
    "Pretrained model loaded."
)

print("=" * 60)


# ==================================================
# REPLAY BUFFER
#
# Không dùng buffer cũ.
#
# Vì environment/path input đã thay đổi.
# ==================================================

replay_buffer = ReplayBuffer(
    state_dim=STATE_DIM,
    action_dim=ACTION_DIM,
    capacity=BUFFER_SIZE,
)


# ==================================================
# TRAINING
# ==================================================

total_steps = 0

best_reward = -float(
    "inf"
)


# ==================================================
# EPISODES
# ==================================================

for episode in range(
    1,
    EPISODES + 1
):

    # ==================================================
    # RESET
    # ==================================================

    state = env.reset()

    state = np.asarray(
        state,
        dtype=np.float32
    )


    # --------------------------------------------------
    # State validation
    # --------------------------------------------------

    if state.shape != (
        STATE_DIM,
    ):

        raise ValueError(
            f"Invalid state shape: "
            f"{state.shape}"
        )


    episode_reward = 0.0

    episode_steps = 0


    # ==================================================
    # EPISODE
    # ==================================================

    for step in range(
        MAX_STEPS
    ):

        # ==================================================
        # ACTION
        # ==================================================

        if (
            total_steps
            < WARMUP_STEPS
        ):

            # ------------------------------------------
            # Random exploration
            # ------------------------------------------

            action = np.random.uniform(
                -1.0,
                1.0,
                size=ACTION_DIM
            ).astype(
                np.float32
            )

        else:

            # ------------------------------------------
            # PRETRAINED POLICY
            # ------------------------------------------

            action = (
                agent.select_action(
                    state,
                    evaluate=False
                )
            )

            action = np.asarray(
                action,
                dtype=np.float32
            )


        # ==================================================
        # ENVIRONMENT
        # ==================================================

        next_state, reward, done = (
            env.step(
                action
            )
        )


        next_state = np.asarray(
            next_state,
            dtype=np.float32
        )

        reward = float(
            reward
        )

        done = bool(
            done
        )


        # ==================================================
        # STATE CHECK
        # ==================================================

        if next_state.shape != (
            STATE_DIM,
        ):

            raise ValueError(
                f"Invalid next_state "
                f"shape: "
                f"{next_state.shape}"
            )


        # ==================================================
        # REPLAY BUFFER
        # ==================================================

        replay_buffer.add(
            state,
            action,
            reward,
            next_state,
            done
        )


        # ==================================================
        # SAC UPDATE
        # ==================================================

        if (
            total_steps >= UPDATE_AFTER
            and
            len(replay_buffer)
            >= BATCH_SIZE
            and
            total_steps
            % UPDATE_EVERY
            == 0
        ):

            info = agent.update(
                replay_buffer,
                batch_size=BATCH_SIZE
            )


        # ==================================================
        # UPDATE STATE
        # ==================================================

        state = next_state

        episode_reward += reward

        episode_steps += 1

        total_steps += 1


        # ==================================================
        # DONE
        # ==================================================

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
    # BEST MODEL V2
    # ==================================================

    if (
        episode_reward
        > best_reward
    ):

        best_reward = (
            episode_reward
        )


        best_path = os.path.join(
            MODEL_DIR,
            f"best_model_{VERSION}.pth"
        )


        agent.save(
            best_path
        )


        print(
            "  -> Best model saved: "
            f"{best_path} "
            f"(reward="
            f"{best_reward:.3f})"
        )


    # ==================================================
    # CHECKPOINT V2
    # ==================================================

    if (
        episode
        % SAVE_EVERY
        == 0
    ):

        checkpoint_path = (
            os.path.join(
                MODEL_DIR,
                f"checkpoint_{episode}_{VERSION}.pth"
            )
        )


        agent.save(
            checkpoint_path
        )


        print(
            "  -> Checkpoint saved: "
            f"{checkpoint_path}"
        )


# ==================================================
# FINAL MODEL V2
# ==================================================

final_path = os.path.join(
    MODEL_DIR,
    f"final_model_{VERSION}.pth"
)


agent.save(
    final_path
)


# ==================================================
# FINISHED
# ==================================================

print()

print("=" * 60)

print(
    "V2 Training finished"
)

print(
    f"Best reward : "
    f"{best_reward:.3f}"
)

print(
    f"Final model : "
    f"{final_path}"
)

print(
    f"Total steps : "
    f"{total_steps}"
)

print("=" * 60)


# ==================================================
# CLOSE
# ==================================================

env.close()