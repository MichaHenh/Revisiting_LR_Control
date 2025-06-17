"""Custom SGD environment with customizable optimizer_type. Class is adapted from dacbench.envs.SGDEnv"""
from __future__ import annotations

import numpy as np
import torch

from time import time
from dacbench.envs import SGDEnv
from dacbench import AbstractMADACEnv
from dacbench.envs.env_utils import sgd_utils
from dacbench.envs.env_utils.sgd_utils import random_torchvision_loader
from dacbench_custom import custom_models
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
from dacbench_custom.libsvmloader import fetch_libsvm


class LIBSVMDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.todense() if hasattr(X, 'todense') else X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.y = self.y - self.y.min()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def random_libsvm_loader(
    seed: int,
    name: str | None,
    batch_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    
    try:
        X, y = fetch_libsvm(name, normalize=True)

        train_loader = DataLoader(LIBSVMDataset(X, y), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(LIBSVMDataset(X, y), batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(LIBSVMDataset(X, y), batch_size=batch_size, shuffle=True)

        return train_loader, val_loader, test_loader
    except:
        return None, None, None

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
            target = target.to(preds.device)
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
    """Run a single epoch of training for given `model` with `loss_function`."""
    last_loss = None
    running_loss = 0
    model.train()
    print(f"Started epoch on {device} with {len(loader)}")
    for data, target in loader:
        #print("iteration")
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output, target)
        loss.mean().backward()
        optimizer.step()
        last_loss = loss
        running_loss += last_loss.mean().item()
    return last_loss.mean().detach(), running_loss / len(loader)

def run_epoch_stormplus(model, loss_function, loader, optimizer, device="cpu"):
    r"""Run a single stormplus-type epoch of training for given `model` with `loss_function` using step and correction step.
        This encompasses an estimator step in addition to optimizer steps."""
    last_loss = None
    running_loss = 0

    model.train()

    # ONLY FOR THE FIRST BATCH: We need to compute the initial estimator d_1, 
    # which is the first (mini-batch) stochastic gradient g_1. To set the estimator
    # we need to call compute_step() with the first batch.
    (data, label) = next(iter(loader))
    # data, label = data.to(device), label.to(device)
    optimizer.zero_grad()
    output = model(data)
    loss = loss_function(output, label)
    loss.mean().backward()
    optimizer.compute_estimator(normalized_norm=True)

    for _, (data, label) in enumerate(loader, 0):
        # data, label = data.to(device), label.to(device)

        # main optimization step
        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output, label)
        loss.mean().backward()

        # uses \tilde g_t from the backward() call above
        # uses d_t already saved as parameter group state from previous iteration
        optimizer.step()

        # update tracked losses
        last_loss = loss
        running_loss += last_loss.mean().item()

        # makes the second pass, backpropagation for the NEXT iterate using the current data batch
        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output, label)
        loss.mean().backward()

        # updates estimate d_{t+1} for the next iteration, saves g_{t+1} for next iteration
        optimizer.compute_estimator(normalized_norm=True)

    return last_loss.mean().detach(), running_loss / len(loader)


class CustomSGDEnv(SGDEnv):
    """The SGD DAC Environment implements the problem of dynamically configuring
    the learning rate hyperparameter of a neural network optimizer
    (can be specified in as optimizer_type) for a supervised learning task.
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
        torch.manual_seed(config['seed'])
        
        # START of SGDEnv init
        super().__init__(config)
        self.epoch_mode = config.get("epoch_mode", True)
        self.device = config.get("device")

        self.learning_rate = None
        self.optimizer_type = torch.optim.AdamW
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

        self.train_loader, self.validation_loader, self.test_loader = random_libsvm_loader(
            config.get("seed"),
            self.dataset_name,
            self.batch_size)

        if self.train_loader is None:
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
        # END of SGDEnv init

        self.optimizer_type = optimizer_type
        self.use_validation = config['use_validation'] if 'use_validation' in config else True
        self.use_testing = config['use_testing'] if 'use_testing' in config else True
        self.use_validation_as_test = config['use_validation_as_test'] if 'use_validation_as_test' in config else False
        self.use_run_epoch_stormplus = config['use_run_epoch_stormplus'] if 'use_run_epoch_stormplus' in config else False
        self.custom_model = config.get("custom_model", None)

    def step(self, action: float):
        """Update the parameters of the neural network using the given learning rate lr,
        in the direction specified by AdamW, and if not done (crashed/cutoff reached),
        performs another forward/backward pass (update only in the next step).
        """
        truncated = super().step_()
        info = {}
        if isinstance(action, float):
            action = [action]
        self.optimizer = _optimizer_action(self.optimizer, action, self.use_momentum)
        if self.epoch_mode:
            self.loss, self.average_loss = (run_epoch_stormplus if self.use_run_epoch_stormplus else run_epoch)(
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
        self.loss = self.loss.cpu().numpy().item()

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
        
        """Option to disable evaluation of validation set"""
        if self.use_validation:
            if (
                (self.c_step % len(self.train_loader) == 0) or self._done
            ):  # Calculate validation loss at the end of an epoch
                batch_percentage = 1.0
            else:
                batch_percentage = 0.1 if self.epoch_mode else 0 # dont validate every batch step in batch mode

            if batch_percentage > 0:
                val_args = [
                self.model,
                self.loss_function,
                self.validation_loader,
                self.batch_size,
                batch_percentage,
                self.device,
                ]
                validation_loss, validation_accuracy = test(*val_args)

                self.validation_loss = validation_loss.mean().detach().cpu().numpy()
                self.validation_accuracy = validation_accuracy.mean().detach().numpy()
                if (
                    self.min_validation_loss is None
                    or self.validation_loss <= self.min_validation_loss
                ):
                    self.min_validation_loss = self.validation_loss

        """Option to disable evaluation of test set and option to use the validation set as test set."""
        if self.use_testing or self.use_validation_as_test:
            if self._done:
                val_args = [
                    self.model,
                    self.loss_function,
                    self.test_loader if not self.use_validation_as_test else self.validation_loader,
                    self.batch_size,
                    1.0,
                    self.device,
                ]
                self.test_losses, self.test_accuracies = test(*val_args)

        reward = self.get_reward(self)

        print("step {}/{}".format(self.c_step, self.n_steps))

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
            hub_model = torch.hub.load(
                # local model loading for offline mode
                torch.hub.get_dir() + '/' +
                self.torchub_model[0],
                self.torchub_model[1],
                pretrained=self.torchub_model[2],
                source='local'
            )
            self.model = torch.nn.Sequential(hub_model, torch.nn.LogSoftmax(dim=1))
        elif self.custom_model:
            custom_model = custom_models.get_model(self.custom_model)
            self.model = torch.nn.Sequential(custom_model, torch.nn.LogSoftmax(dim=-1))
        else:
            # Load model from config file
            self.model = sgd_utils.create_model(
                self.config.get("layer_specification"), len(self.datasets[0].classes)
            )

        self.learning_rate = None
        self.info = {}
        self._done = False

        self.model.to(self.device)
        """Custom optimizer initialization with or without hyperparameters"""
        if self.optimizer_params is not None:
            self.optimizer: torch.optim.Optimizer = self.optimizer_type(
            **self.optimizer_params, params=self.model.parameters()
            )
        else:
            self.optimizer: torch.optim.Optimizer = self.optimizer_type(params=self.model.parameters())
        ###
        self.loss = 0
        self.test_losses = None
        self.test_accuracies = None

        self.validation_loss = 0
        self.validation_accuracy = 0
        self.min_validation_loss = None

        if self.epoch_mode:
            self.average_loss = 0

        return self.get_state(self), {}
