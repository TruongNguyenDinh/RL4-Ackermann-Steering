import numpy as np


class ReplayBuffer:

    def __init__(
        self,
        state_dim,
        action_dim,
        capacity=100_000,
    ):
        self.capacity = capacity

        self.states = np.zeros(
            (capacity, state_dim),
            dtype=np.float32
        )

        self.actions = np.zeros(
            (capacity, action_dim),
            dtype=np.float32
        )

        self.rewards = np.zeros(
            (capacity, 1),
            dtype=np.float32
        )

        self.next_states = np.zeros(
            (capacity, state_dim),
            dtype=np.float32
        )

        self.dones = np.zeros(
            (capacity, 1),
            dtype=np.float32
        )

        self.ptr = 0
        self.size = 0

    def add(
        self,
        state,
        action,
        reward,
        next_state,
        done,
    ):

        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done

        self.ptr = (
            self.ptr + 1
        ) % self.capacity

        self.size = min(
            self.size + 1,
            self.capacity
        )

    def sample(
        self,
        batch_size,
    ):

        indices = np.random.randint(
            0,
            self.size,
            size=batch_size
        )

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
        )

    def __len__(self):

        return self.size