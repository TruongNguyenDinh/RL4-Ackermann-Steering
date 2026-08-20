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
# WARM START MODE
#
# QUAN TRỌNG: reward function đã đổi hoàn toàn về
# scale/tính chất so với lúc model cũ được train.
# Nếu load nguyên actor+critic cũ, critic sẽ dự đoán
# sai be bét ngay từ đầu (đã thấy trong log trước:
# full 2000-step episode nhưng reward -2660), khiến
# policy sụp nhanh.
#
# 3 lựa chọn, chọn 1 qua WARM_START_MODE:
#
#   "none"        - Train from scratch, KHÔNG load gì.
#                    An toàn nhất, khuyến nghị mặc định
#                    sau khi đổi reward function.
#
#   "actor_only"  - Chỉ load actor (policy) từ model cũ
#                    làm điểm khởi đầu hành vi, RESET
#                    hoàn toàn critic + optimizer state.
#                    Cần SACAgent hỗ trợ load actor riêng
#                    (xem hàm _load_actor_only bên dưới -
#                    PHẢI kiểm tra khớp với rl/agent.py
#                    thực tế của bạn, đây chỉ là khung).
#
#   "full"        - Load actor+critic như cũ. CHỈ dùng
#                    khi reward function gần như không đổi
#                    scale. Không khuyến nghị ở thời điểm
#                    này.
# ==================================================

WARM_START_MODE = "none"


# ==================================================
# WARMUP
#
# Vì không còn dùng nguyên policy cũ ngay từ đầu (mode
# "none" hoặc "actor_only" reset critic), nên cần một
# giai đoạn random exploration thật để replay buffer có
# dữ liệu đa dạng cho critic học lại từ đầu. Đặt 0 chỉ
# hợp lý khi WARM_START_MODE = "full".
# ==================================================

WARMUP_STEPS = 5_000 if WARM_START_MODE != "full" else 0


# ==================================================
# UPDATE
# ==================================================

UPDATE_AFTER = 1_000
UPDATE_EVERY = 1


# ==================================================
# SAVE
# ==================================================

SAVE_EVERY = 100

# Số episode gần nhất dùng để tính reward trung bình
# trượt (moving average) khi xét "best model". Dùng
# trung bình thay vì 1 episode đơn lẻ để tránh save
# nhầm model chỉ vì 1 episode ăn may.
BEST_MODEL_WINDOW = 10


# ==================================================
# MODEL
# ==================================================

MODEL_DIR = "model"

PRETRAINED_MODEL = os.path.join(
    MODEL_DIR,
    "best_model.pth"
)


# ==================================================
# VERSION
# ==================================================

VERSION = "v2"


# ==================================================
# SEED
# ==================================================

SEED = 42

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


# ==================================================
# HEADER
# ==================================================

print("=" * 60)
print("SAC Self-Driving Training - V2")
print("=" * 60)
print(f"Device          : {device}")
print(f"State dim       : {STATE_DIM}")
print(f"Action dim      : {ACTION_DIM}")
print(f"Episodes        : {EPISODES}")
print(f"Max steps       : {MAX_STEPS}")
print(f"Warm start mode : {WARM_START_MODE}")
print(f"Warmup steps    : {WARMUP_STEPS}")
print(f"Pretrained      : {PRETRAINED_MODEL}")
print(f"Output          : *_{VERSION}.pth")

if torch.cuda.is_available():
    print(f"GPU             : {torch.cuda.get_device_name(0)}")

print("=" * 60)


# ==================================================
# MODEL DIRECTORY
# ==================================================

os.makedirs(MODEL_DIR, exist_ok=True)


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
initial_state = np.asarray(initial_state, dtype=np.float32)

print(f"Initial state shape: {initial_state.shape}")

if initial_state.shape != (STATE_DIM,):
    raise ValueError(
        "\n"
        f"Expected state shape ({STATE_DIM},), "
        f"but CarEnv returned {initial_state.shape}\n"
    )

print(f"2D path     : {len(env.waypoints_2d)}")
print(f"3D path     : {len(env.path_3d)}")
print(f"Official    : {len(env.official_path)}")
print(f"RL path     : {len(env.processed_path)}")
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
# WARM START
# ==================================================

def _load_actor_only(agent, path, device):
    """
    Chỉ nạp trọng số của actor/policy network từ
    checkpoint cũ, KHÔNG đụng tới critic/target
    network/optimizer state.

    LƯU Ý: hàm này giả định agent.save() lưu một dict
    dạng {"actor": ..., "critic": ..., ...} - đây là
    cấu trúc phổ biến nhưng CẦN đối chiếu với đúng
    rl/agent.py bạn đang dùng. Nếu SACAgent.save/load
    lưu theo cấu trúc khác, sửa hàm này cho khớp trước
    khi dùng WARM_START_MODE = "actor_only".
    """

    checkpoint = torch.load(path, map_location=device)

    if "actor" not in checkpoint:
        raise KeyError(
            "\n"
            "Checkpoint không có key 'actor'. Cấu trúc "
            "checkpoint thực tế:\n"
            f"{list(checkpoint.keys())}\n"
            "Cần sửa _load_actor_only() cho khớp với "
            "rl/agent.py trước khi dùng actor_only mode."
        )

    agent.actor.load_state_dict(checkpoint["actor"])

    # Critic, target networks, optimizer state đều GIỮ
    # NGUYÊN như lúc SACAgent() vừa khởi tạo (random),
    # để critic học lại từ đầu theo đúng thang reward mới.


if WARM_START_MODE == "none":

    print("Warm start: KHONG load model cu. Train from scratch.")

elif WARM_START_MODE == "actor_only":

    if not os.path.exists(PRETRAINED_MODEL):
        raise FileNotFoundError(
            f"\nPretrained model not found:\n{PRETRAINED_MODEL}\n"
        )

    print("Warm start: chi load ACTOR tu model cu, reset critic...")
    print(f"  -> {PRETRAINED_MODEL}")

    _load_actor_only(agent, PRETRAINED_MODEL, device)

    print("Da load actor. Critic/optimizer giu nguyen (random init).")

elif WARM_START_MODE == "full":

    if not os.path.exists(PRETRAINED_MODEL):
        raise FileNotFoundError(
            f"\nPretrained model not found:\n{PRETRAINED_MODEL}\n"
        )

    print("Warm start: load FULL actor+critic tu model cu...")
    print(f"  -> {PRETRAINED_MODEL}")

    agent.load(PRETRAINED_MODEL)

    print("Da load full model.")

else:

    raise ValueError(
        f"WARM_START_MODE khong hop le: {WARM_START_MODE}"
    )

print("=" * 60)


# ==================================================
# REPLAY BUFFER
#
# Luon tao moi - khong tai su dung buffer cu, vi reward
# function/environment da thay doi.
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

best_reward = -float("inf")

# Dung de tinh moving average reward cho viec save best model
recent_rewards = []

# Dung de theo doi trung binh cac thanh phan reward moi
# episode, giup phat hien nhanh neu 1 term nao dang ap
# dao cac term con lai (vd heading_penalty qua lon do
# sai don vi).
episode_breakdown_sums = {}


# ==================================================
# EPISODES
# ==================================================

for episode in range(1, EPISODES + 1):

    # ==================================================
    # RESET
    # ==================================================

    state = env.reset()
    state = np.asarray(state, dtype=np.float32)

    if state.shape != (STATE_DIM,):
        raise ValueError(f"Invalid state shape: {state.shape}")

    episode_reward = 0.0
    episode_steps = 0

    episode_breakdown_sums = {}

    # ==================================================
    # EPISODE
    # ==================================================

    for step in range(MAX_STEPS):

        # ==================================================
        # ACTION
        # ==================================================

        if total_steps < WARMUP_STEPS:

            # Random exploration that su - can thiet de
            # replay buffer co du lieu da dang cho critic
            # hoc lai tu dau khi khong con dung nguyen
            # policy cu.

            action = np.random.uniform(
                -1.0, 1.0, size=ACTION_DIM
            ).astype(np.float32)

        else:

            action = agent.select_action(state, evaluate=False)
            action = np.asarray(action, dtype=np.float32)

        # ==================================================
        # ENVIRONMENT
        # ==================================================

        next_state, reward, done = env.step(action)

        next_state = np.asarray(next_state, dtype=np.float32)
        reward = float(reward)
        done = bool(done)

        if next_state.shape != (STATE_DIM,):
            raise ValueError(
                f"Invalid next_state shape: {next_state.shape}"
            )

        # ==================================================
        # REWARD BREAKDOWN LOGGING (debug reward hacking)
        # ==================================================

        breakdown = {}

        if hasattr(env, "get_reward_breakdown"):
            breakdown = env.get_reward_breakdown()

        for key, value in breakdown.items():
            episode_breakdown_sums[key] = (
                episode_breakdown_sums.get(key, 0.0) + value
            )

        # ==================================================
        # REPLAY BUFFER
        # ==================================================

        replay_buffer.add(state, action, reward, next_state, done)

        # ==================================================
        # SAC UPDATE
        # ==================================================

        if (
            total_steps >= UPDATE_AFTER
            and len(replay_buffer) >= BATCH_SIZE
            and total_steps % UPDATE_EVERY == 0
        ):

            info = agent.update(replay_buffer, batch_size=BATCH_SIZE)

        # ==================================================
        # UPDATE STATE
        # ==================================================

        state = next_state
        episode_reward += reward
        episode_steps += 1
        total_steps += 1

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

    # In trung binh moi thanh phan reward / step, de de
    # phat hien term nao dang bat thuong (vd heading_penalty
    # luon o muc kich tran K_HEADING * pi -> nghi sai don vi).

    if episode_breakdown_sums and episode_steps > 0:

        parts = [
            f"{key}={value / episode_steps:+.4f}"
            for key, value in episode_breakdown_sums.items()
        ]

        print("  -> Avg/step: " + " | ".join(parts))

    # ==================================================
    # MOVING AVERAGE REWARD (cho best model, on dinh hon
    # so voi xet tung episode rieng le)
    # ==================================================

    recent_rewards.append(episode_reward)

    if len(recent_rewards) > BEST_MODEL_WINDOW:
        recent_rewards.pop(0)

    moving_avg_reward = sum(recent_rewards) / len(recent_rewards)

    # ==================================================
    # BEST MODEL V2
    #
    # Chi xet tu window thu BEST_MODEL_WINDOW tro di, de
    # tranh save nham model ngay tu episode dau tien (chua
    # co du du lieu de moving average co y nghia).
    # ==================================================

    if (
        len(recent_rewards) >= BEST_MODEL_WINDOW
        and moving_avg_reward > best_reward
    ):

        best_reward = moving_avg_reward

        best_path = os.path.join(
            MODEL_DIR, f"best_model_{VERSION}.pth"
        )

        agent.save(best_path)

        print(
            "  -> Best model saved: "
            f"{best_path} "
            f"(moving_avg_reward={best_reward:.3f}, "
            f"window={BEST_MODEL_WINDOW})"
        )

    # ==================================================
    # CHECKPOINT V2
    # ==================================================

    if episode % SAVE_EVERY == 0:

        checkpoint_path = os.path.join(
            MODEL_DIR, f"checkpoint_{episode}_{VERSION}.pth"
        )

        agent.save(checkpoint_path)

        print(f"  -> Checkpoint saved: {checkpoint_path}")


# ==================================================
# FINAL MODEL V2
# ==================================================

final_path = os.path.join(MODEL_DIR, f"final_model_{VERSION}.pth")

agent.save(final_path)


# ==================================================
# FINISHED
# ==================================================

print()
print("=" * 60)
print("V2 Training finished")
print(f"Best moving avg reward : {best_reward:.3f}")
print(f"Final model            : {final_path}")
print(f"Total steps             : {total_steps}")
print("=" * 60)


# ==================================================
# CLOSE
# ==================================================

env.close()