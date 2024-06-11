import torch
from torchmin import minimize_constr
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
                    state['theta_sum'] = p.clone().detach()
                    # Sum of the minimal ht
                    state['H'] = torch.ones(1).detach()
                    # Reward/wealth of the algorithm
                    state['wealth'] = torch.ones(1).detach()

                theta_sum, H, wealth = (
                    state['theta_sum'],
                    state['H'],
                    state['wealth']
                )

                #if group['weight_decay'] != 0:
                #    grad = grad.add(p, alpha=group['weight_decay'])
                    
                #grad=grad*lr
                #print(grad)
                grad_norm = torch.norm(grad)
                if(grad_norm>1):
                    grad.div_(grad_norm)

                def update_wealth(h):
                    exponent = -grad.dot(theta_sum).mul(torch.log(1+h/H))
                    exponent.add_(grad.norm().square().mul(h.add(H.mul(torch.log(H.div(H.add(h)))))))
                    return torch.exp(exponent).mul(wealth)

                def update_parameters(h):
                    factor = update_wealth(h).div(H.add(h))
                    return factor.mul(theta_sum.sub(h.mul(grad)))
                
                def tangent(h):
                    return (loss + grad.dot(update_parameters(h).sub(p)) - 0)

                # compute the minimizing ht
                h_min = torch.min(torch.ones_like(H), minimize_constr(tangent, torch.zeros(1), bounds={'lb': -H, 'ub': H}).x)
                
                # update model parameters
                unflattened_p.data.copy_(update_parameters(h_min))
                
                # update wealth
                wealth_tmp = update_wealth(h_min)
                if torch.any(torch.isnan(wealth_tmp)):
                    print("alarm")

                wealth = wealth_tmp
                # update H
                H.add_(h_min)
                # update theta sum of weighted gradients
                theta_sum.sub_(h_min.mul(grad))

                if torch.any(torch.isnan(unflattened_p)):
                    print("this is bad")

        return loss