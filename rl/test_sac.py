import numpy as np

from rl.agent import SACAgent
from rl.replay_buffer import ReplayBuffer


STATE_DIM = 22
ACTION_DIM = 2

BUFFER_SIZE = 10_000
BATCH_SIZE = 64


# ==========================
# Agent
# ==========================

agent = SACAgent(
    state_dim=STATE_DIM,
    action_dim=ACTION_DIM,
)


print("Device:", agent.device)


# ==========================
# Replay Buffer
# ==========================

buffer = ReplayBuffer(
    state_dim=STATE_DIM,
    action_dim=ACTION_DIM,
    capacity=BUFFER_SIZE,
)


# ==========================
# Fake transitions
# ==========================

for _ in range(1000):

    state = np.random.randn(
        STATE_DIM
    ).astype(np.float32)

    action = np.random.uniform(
        -1.0,
        1.0,
        ACTION_DIM
    ).astype(np.float32)

    reward = np.random.randn()

    next_state = np.random.randn(
        STATE_DIM
    ).astype(np.float32)

    done = np.random.choice(
        [0.0, 1.0]
    )

    buffer.add(
        state,
        action,
        reward,
        next_state,
        done,
    )


# ==========================
# Update
# ==========================

for i in range(10):

    info = agent.update(
        buffer,
        batch_size=BATCH_SIZE,
    )

    print(
        f"Update {i}:",
        info
    )


# ==========================
# Action test
# ==========================

state = np.zeros(
    STATE_DIM,
    dtype=np.float32
)

action = agent.select_action(
    state
)

print()
print("State shape:", state.shape)
print("Action:", action)
print("Action shape:", action.shape)