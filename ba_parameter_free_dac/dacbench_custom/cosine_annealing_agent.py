import math
from time import time
from dacbench.abstract_agent import AbstractDACBenchAgent
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

class CosineAnnealingWRAgent(AbstractDACBenchAgent):
    """Agent using cosine annea."""

    def __init__(self, env, T_0, eta_min=0, base_lr=0.1, t_mult=1):
        """Initialize the Agent."""
        self.eta_min = eta_min
        self.base_lr = base_lr
        self.current_lr = base_lr
        self.last_epoch = -1

        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = t_mult
        self.T_cur = self.last_epoch
        self.epoch = 0
        super().__init__(env)

    def act(self, state=None, reward=None):
        """Returns the next action."""
        # current_time_ms = time()
        self.epoch += 1
        
        if self.epoch is None and self.last_epoch < 0:
            self.epoch = 0

        if self.epoch is None:
            self.epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.T_cur = self.T_cur - self.T_i
                self.T_i = self.T_i * self.T_mult
        else:
            if self.epoch < 0:
                raise ValueError("Expected non-negative epoch, but got {}".format(self.epoch))
            if self.epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = self.epoch % self.T_0
                else:
                    n = int(math.log((self.epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.T_cur = self.epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = self.epoch
        self.last_epoch = math.floor(self.epoch)

        self.current_lr = self.eta_min + (self.base_lr - self.eta_min) * (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
        # print("CAWR: {}".format(time()-current_time_ms))
        return self.current_lr

    def train(self, state=None, reward=None):  # noqa: D102
        pass

    def end_episode(self, state=None, reward=None):  # noqa: D102
        pass
