import random
from collections import deque, namedtuple
import numpy as np

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, replay_batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = replay_batch_size
        self.experience = namedtuple("Experience", field_names=["ob_seq", "ac", "re", "next_ob", "done"])
        self.seed = random.seed(seed)

    def add(self, ob_seq, ac, re, next_ob, done):
        """Add a new experience to memory."""
        e = self.experience(ob_seq, ac, re, next_ob, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=1)
        obs = np.vstack([e.ob_seq for e in experiences if e is not None])
        acs = np.vstack([e.ac for e in experiences if e is not None])
        res = np.vstack([e.re for e in experiences if e is not None])
        next_obs = np.vstack([e.next_ob for e in experiences if e is not None])
        dones = (np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8))
        return (obs, acs, res, next_obs, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)