import math
from dacbench.abstract_agent import AbstractDACBenchAgent

class CosineAnnealingWRAgent(AbstractDACBenchAgent):
    r"""Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr, :math:`T_{cur}`
    is the number of epochs since the last restart and :math:`T_{i}` is the number
    of epochs between two warm restarts in SGDR:

    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 +
        \cos\left(\frac{T_{cur}}{T_{i}}\pi\right)\right)

    When :math:`T_{cur}=T_{i}`, set :math:`\eta_t = \eta_{min}`.
    When :math:`T_{cur}=0` after restart, set :math:`\eta_t=\eta_{max}`.

    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_.

    Args:
        T_0 (int): Number of iterations until the first restart.
        T_mult (int, optional): A factor by which :math:`T_{i}` increases after a restart. Default: 1.
        eta_min (float, optional): Minimum learning rate. Default: 0.
        base_lr (float, optional): Initial learning rate. Default: 0.1.
        last_epoch (int, optional): The index of the last epoch. Default: -1.
        verbose (bool | str): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

            .. deprecated:: 2.2
                ``verbose`` is deprecated. Please use ``get_last_lr()`` to access the
                learning rate.

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983

        Adapted from torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
    """

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
        return self.current_lr

    def train(self, state=None, reward=None):  # noqa: D102
        pass

    def end_episode(self, state=None, reward=None):  # noqa: D102
        pass
