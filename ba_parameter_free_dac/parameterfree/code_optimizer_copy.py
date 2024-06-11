import torch
from torchmin import minimize
from torch.optim.optimizer import Optimizer, required, _use_grad_for_differentiable

__all__ = ['CODE', 'code']

class CODE(Optimizer):
    """Implements the CODE algorithm."""

    def __init__(self, params, alpha: float = 100, eps: float = 1e-8, weight_decay: float = 0):
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= alpha:
            raise ValueError("Invalid alpha value: {}".format(alpha))
        if not 0.0 <= weight_decay:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(weight_decay=weight_decay, lr=1.0)
        self._alpha = alpha
        self._eps = eps

        super(CODE, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure = None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        lr = max(group['lr'] for group in self.param_groups)
        
        for group in self.param_groups:
            for unflattened_p in group['params']:
                if unflattened_p.grad is None:
                    continue
                grad = unflattened_p.grad
                if grad.is_sparse:
                    raise RuntimeError('CODE does not support sparse gradients')

                state = self.state[unflattened_p]

                p = torch.flatten(unflattened_p)
                grad = torch.flatten(grad)

                # State initialization
                if len(state) == 0:
                    # Sum of the negative gradients (theta)
                    state['theta_sum'] = torch.zeros_like(p).detach()
                    # Sum of the minimal ht
                    state['H'] = torch.ones(p.shape[1:2] or 1).detach()
                    # Reward/wealth of the algorithm
                    state['wealth'] = torch.ones(p.shape[1:2] or 1).detach()

                theta_sum, H, wealth = (
                    state['theta_sum'],
                    state['H'],
                    state['wealth']
                )

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])
                    
                grad=grad*lr

                def wealth_update(h):
                    logarithm = torch.log(1+h/H)
                    exponent = grad.mul(theta_sum).sum(dim=0).mul(logarithm)
                    exponent.add((torch.norm(grad,dim=0)**2).mul(h+H.mul(logarithm)))

                    return torch.exp(-exponent).mul(wealth)
                
                def Psi(h):
                    return wealth_update(h).div(H.add(h)).mul(theta_sum.sub(h*grad))
                
                def Phi(h):
                    #h = torch.from_numpy(h)
                    return loss + torch.dot(torch.flatten(grad), torch.flatten(Psi(h).sub(p)))
                
                # compute the minimizing ht
                h_min = torch.min(torch.ones_like(H), minimize(Phi, H, method='newton-cg').x)

                # update wealth
                wealth = wealth_update(h_min)
                # update H
                H.add_(h_min)
                # update theta sum of weighted gradients
                theta_sum.sub_(grad.mul(h_min))

                # update model parameters
                p.data.copy_(theta_sum.mul(wealth.div(H)))
                                
        return loss