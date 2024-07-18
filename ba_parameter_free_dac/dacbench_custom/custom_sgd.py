"""Custom SGD environment."""
from __future__ import annotations

import numpy as np
import torch

from dacbench.envs import SGDEnv
from dacbench.envs.env_utils import sgd_utils
from dacbench.envs.env_utils.sgd_utils import random_torchvision_loader

'''
def _optimizer_action(
    optimizer: torch.optim.Optimizer, action: float, use_momentum: bool
) -> None:
    for g in optimizer.param_groups:
        g["lr"] = action[0]
        if use_momentum:
            print("Momentum")
            g["betas"] = (action[1], 0.999)
    return optimizer


def test(
    model,
    loss_function,
    loader,
    batch_size,
    batch_percentage: float = 1.0,
    device="cpu",
):
    """Evaluate given `model` on `loss_function`.

    Percentage defines how much percentage of the data shall be used.
    If nothing given the whole data is used.

    Returns:
        test_losses: Batch validation loss per data point
        test_accuracies: Batch validation accuracy per batch
    """
    nmb_sets = batch_percentage * (len(loader.dataset) / batch_size)
    model.eval()
    test_losses = []
    test_accuracies = []
    i = 0

    with torch.no_grad():
        for data, target in loader:
            d_data, d_target = data.to(device), target.to(device)
            output = model(d_data)
            _, preds = output.max(dim=1)
            test_losses.append(loss_function(output, d_target))
            test_accuracies.append(torch.sum(preds == target) / len(target))
            i += 1
            if i >= nmb_sets:
                break
    return torch.cat(test_losses), torch.tensor(test_accuracies)


def forward_backward(model, loss_function, loader, device="cpu"):
    """Do a forward and a backward pass for given `model` for `loss_function`.

    Returns:
        loss: Mini batch training loss per data point
    """
    model.train()
    (data, target) = next(iter(loader))
    data, target = data.to(device), target.to(device)
    output = model(data)
    loss = loss_function(output, target)
    loss.mean().backward()
    return loss.mean().detach()


def run_epoch(model, loss_function, loader, optimizer, device="cpu"):
    """Run a single epoch of training for given `model` with `loss_function`.

    Returns:
        loss: mean loss in last batch
        running_loss: mean loss across epoch
    """
    last_loss = None
    running_loss = 0
    model.train()
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        print(data.shape, target.shape)
        output = model(data)
        loss = loss_function(output, target)
        loss.mean().backward()
        optimizer.step()
        last_loss = loss
        running_loss += last_loss.mean().item()
    return last_loss.mean().detach(), running_loss / len(loader)
'''

class CustomSGDEnv(SGDEnv):
    """The SGD DAC Environment implements the problem of dynamically configuring
    the learning rate hyperparameter of a neural network optimizer
    (more specifically, torch.optim.AdamW) for a supervised learning task.
    While training, the model is evaluated after every epoch.

    Actions correspond to learning rate values in [0,+inf[
    For observation space check `observation_space` method docstring.
    For instance space check the `SGDInstance` class docstring
    Reward:
        negative loss of model on test_loader of the instance       if done
        crash_penalty of the instance                               if crashed
        0                                                           otherwise
    """

    metadata = {"render_modes": ["human"]}  # noqa: RUF012

    def __init__(self, config, optimizer_type):
        """Init env."""
        super(CustomSGDEnv, self).__init__(config)
        self.optimizer_type = optimizer_type

    def reset(self, seed=None, options=None):
        """Initialize the neural network, data loaders, etc. for given/random next task.
        Also perform a single forward/backward pass,
        not yet updating the neural network parameters.
        """
        if options is None:
            options = {}
        super().reset_(seed)

        # Use generator
        rng = np.random.RandomState(self.initial_seed)
        if self.use_generator:
            (
                self.model,
                self.optimizer_params,
                self.batch_size,
                self.crash_penalty,
            ) = sgd_utils.random_instance(rng, self.datasets)
        elif self.torchub_model[0]:
            hub_model = torch.hub.load(
                self.torchub_model[0],
                self.torchub_model[1],
                pretrained=self.torchub_model[2],
            )
            self.model = torch.nn.Sequential(hub_model, torch.nn.Softmax(dim=0))
        else:
            # Load model from config file
            self.model = sgd_utils.create_model(
                self.config.get("layer_specification"), len(self.datasets[0].classes)
            )

        self.learning_rate = None
        self.optimizer_type = torch.optim.AdamW
        self.info = {}
        self._done = False

        self.model.to(self.device)
        # custom optimizer initialization
        if self.optimizer_params is not None:
            self.optimizer: torch.optim.Optimizer = self.optimizer_type(
            **self.optimizer_params, params=self.model.parameters()
            )
        else:
            self.optimizer: torch.optim.Optimizer = self.optimizer_type(params=self.model.parameters())
        ###
        self.loss = 0
        self.test_losses = None

        self.validation_loss = 0
        self.validation_accuracy = 0
        self.min_validation_loss = None

        if self.epoch_mode:
            self.average_loss = 0

        return self.get_state(self), {}