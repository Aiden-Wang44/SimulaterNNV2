import numpy as np
from dataclasses import dataclass
@dataclass
class TransitionState:
    estimated_state: np.ndarray
    privileged_state: np.ndarray
    action: np.ndarray
    reward: float
    next_estimated_state: np.ndarray
    next_privileged_state: np.ndarray
    done: bool

class  ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer : list[TransitionState] =[]
        self.position = 0
    def add(self,estimated_state: np.ndarray, privileged_state: np.ndarray, action:np.ndarray, reward:float, next_estimated_state: np.ndarray, next_privileged_state: np.ndarray, done: bool ):
        transition = TransitionState(np.asarray(estimated_state, dtype = np.float32), np.asarray(privileged_state,dtype = np.float32), np.asarray(action,dtype = np.float32),float(reward), np.asarray(next_estimated_state, dtype = np.float32), np.asarray(next_privileged_state, dtype = np.float32), bool(done))
        if len(self.buffer)<self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity
    def sample(self, batch_size: int)->list[TransitionState]:
        indices = np.random.randint(0,len(self.buffer), size=batch_size)
        return [self.buffer[i] for i in indices]
    def __len__(self):
        return len(self.buffer)
