"""Custom SGD environment."""
from __future__ import annotations

import numpy as np
import torch

from dacbench import AbstractMADACEnv
from dacbench.envs.env_utils import sgd_utils
from dacbench.envs.env_utils.sgd_utils import random_torchvision_loader


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
        #print(data.shape, target.shape)
        output = model(data)
        loss = loss_function(output, target)
        loss.mean().backward()
        optimizer.step()
        last_loss = loss
        running_loss += last_loss.mean().item()
    return last_loss.mean().detach(), running_loss / len(loader)


class CustomSGDEnv(AbstractMADACEnv):
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
        super().__init__(config)
        self.epoch_mode = config.get("epoch_mode", True)
        self.device = config.get("device")

        self.learning_rate = None
        self.optimizer_type = optimizer_type
        self.optimizer_params = config.get("optimizer_params")
        self.batch_size = config.get("training_batch_size")
        self.model = config.get("model")
        self.crash_penalty = config.get("crash_penalty")
        self.loss_function = config.loss_function(**config.loss_function_kwargs)
        self.dataset_name = config.get("dataset_name")
        self.use_momentum = config.get("use_momentum")
        self.use_generator = config.get("model_from_dataset")
        self.torchub_model = config.get("torch_hub_model", (False, None, False))

        # Use default reward function, if no specific function is given
        self.get_reward = config.get("reward_function", self.get_default_reward)

        # Use default state function, if no specific function is given
        self.get_state = config.get("state_method", self.get_default_state)

        # Get loaders for instance
        self.datasets, loaders = random_torchvision_loader(
            config.get("seed"),
            config.get("instance_set_path"),
            self.dataset_name,
            self.batch_size,
            config.get("fraction_of_dataset"),
            config.get("train_validation_ratio"),
        )
        self.train_loader, self.validation_loader, self.test_loader = loaders

    def step(self, action: float):
        """Update the parameters of the neural network using the given learning rate lr,
        in the direction specified by AdamW, and if not done (crashed/cutoff reached),
        performs another forward/backward pass (update only in the next step).
        """
        print(self.n_steps)
        print(self.c_step)
        truncated = super().step_()
        info = {}
        if isinstance(action, float):
            action = [action]
        self.optimizer = _optimizer_action(self.optimizer, action, self.use_momentum)

        if self.epoch_mode:
            self.loss, self.average_loss = run_epoch(
                self.model,
                self.loss_function,
                self.train_loader,
                self.optimizer,
                self.device,
            )
        else:
            train_args = [
                self.model,
                self.loss_function,
                self.train_loader,
                self.device,
            ]
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.loss = forward_backward(*train_args)

        crashed = (
            not torch.isfinite(self.loss).any()
            or not torch.isfinite(
                torch.nn.utils.parameters_to_vector(self.model.parameters())
            ).any()
        )
        self.loss = self.loss.numpy().item()

        if crashed:
            self._done = True
            return (
                self.get_state(self),
                self.crash_penalty,
                False,
                True,
                info,
            )

        self._done = truncated

        if (
            self.n_steps % len(self.train_loader) == 0 or self._done
        ):  # Calculate validation loss at the end of an epoch
            batch_percentage = 1.0
        else:
            batch_percentage = 0.1

        val_args = [
            self.model,
            self.loss_function,
            self.validation_loader,
            self.batch_size,
            batch_percentage,
            self.device,
        ]
        validation_loss, validation_accuracy = test(*val_args)

        self.validation_loss = validation_loss.mean().detach().numpy()
        self.validation_accuracy = validation_accuracy.mean().detach().numpy()
        if (
            self.min_validation_loss is None
            or self.validation_loss <= self.min_validation_loss
        ):
            self.min_validation_loss = self.validation_loss

        if self._done:
            val_args = [
                self.model,
                self.loss_function,
                self.test_loader,
                self.batch_size,
                1.0,
                self.device,
            ]
            self.test_losses, self.test_accuracies = test(*val_args)

        reward = self.get_reward(self)

        return self.get_state(self), reward, False, truncated, info

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
            self.model = torch.hub.load(
                self.torchub_model[0],
                self.torchub_model[1],
                pretrained=self.torchub_model[2],
            )
        else:
            # Load model from config file
            self.model = sgd_utils.create_model(
                self.config.get("layer_specification"), len(self.datasets[0].classes)
            )

        self.learning_rate = None
        #self.optimizer_type = torch.optim.AdamW
        self.info = {}
        self._done = False

        self.model.to(self.device)
        if self.optimizer_params is not None:
            self.optimizer: torch.optim.Optimizer = self.optimizer_type(
            **self.optimizer_params, params=self.model.parameters()
            )
        else:
            self.optimizer: torch.optim.Optimizer = self.optimizer_type(params=self.model.parameters())

        self.loss = 0
        self.test_losses = None

        self.validation_loss = 0
        self.validation_accuracy = 0
        self.min_validation_loss = None

        if self.epoch_mode:
            self.average_loss = 0

        return self.get_state(self), {}

    def get_default_reward(self, _) -> float:
        """The default reward function.

        Args:
            _ (_type_): Empty parameter, which can be used when overriding

        Returns:
            float: The calculated reward
        """
        if self.test_losses is not None:
            reward = self.test_losses.sum().item() / len(self.test_loader.dataset)
        else:
            reward = 0.0
        return -reward

    def get_default_state(self, _) -> dict:
        """Default state function.

        Args:
            _ (_type_): Empty parameter, which can be used when overriding

        Returns:
            dict: The current state
        """
        state = {
            "step": self.c_step,
            "loss": self.loss,
            "validation_loss": self.validation_loss,
            "validation_accuracy": self.validation_accuracy,
            "done": self._done,
        }
        if self.epoch_mode:
            state["average_loss"] = self.average_loss

        if self._done and self.test_losses is not None:
            state["test_losses"] = self.test_losses
            state["test_accuracies"] = self.test_accuracies
        return state

    def render(self, mode="human"):
        """Render progress."""
        if mode == "human":
            epoch = 1 + self.c_step // len(self.train_loader)
            epoch_cutoff = self.n_steps // len(self.train_loader)
            batch = 1 + self.c_step % len(self.train_loader)
            print(
                f"prev_lr {self.optimizer.param_groups[0]['lr'] if self.n_steps > 0 else None}, "  # noqa: E501
                f"epoch {epoch}/{epoch_cutoff}, "
                f"batch {batch}/{len(self.train_loader)}, "
                f"batch_loss {self.loss}, "
                f"val_loss {self.validation_loss}"
            )
        else:
            raise NotImplementedError
