import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.optim import AdamW
from ConfigSpace import Configuration, ConfigurationSpace, Float
from smac import HyperparameterOptimizationFacade as HPOFacade
from smac import Scenario
from dacbench.runner import run_benchmark
from parameterfree.parameter_free_sgd_benchmark import ParameterFreeSGDBenchmark
from dacbench.agents import StaticAgent
from dacbench.wrappers import PerformanceTrackingWrapper
from dacbench.benchmarks import SGDBenchmark

# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 10)
        )

    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=0)
        lr = Float("lr", (0, 0.1), default=1e-3)
        cs.add_hyperparameters([lr])

        return cs

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

    def train_smac(self, config: Configuration, seed: int = 0) -> float:
        """Returns the y value of a quadratic function with a minimum we know to be at x=0."""
        lr = config["lr"]

        loss_fn = nn.CrossEntropyLoss()
        optimizer = AdamW(self.parameters(), lr=lr)

        num_batches = len(train_dataloader)
        self.train()
        losses = 0
        for batch, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device), y.to(device)

            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)
            losses += loss.item()

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        ls = losses / num_batches
        print(f"Current loss: {ls}")


        return ls

    

model = NeuralNetwork().to(device)
print(model)

scenario = Scenario(model.configspace, deterministic=True, n_trials=10)

smac = HPOFacade(
        scenario,
        model.train_smac,  # We pass the target function here
        overwrite=True,  # Overrides any previous results that are found that are inconsistent with the meta-data
    )

incumbent = smac.optimize()

print(f"Incumbent: {incumbent}")

# Get cost of default configuration
default_cost = smac.validate(model.configspace.get_default_configuration())
print(f"Default cost: {default_cost}")

# Let's calculate the cost of the incumbent
incumbent_cost = smac.validate(incumbent)
print(f"Incumbent cost: {incumbent_cost}")


### Test Model using DACBench



# Result output path
path = "dacbench_tabular"

bench_env = PerformanceTrackingWrapper(SGDBenchmark().get_benchmark())

# Run SGD benchmark
run_benchmark(bench_env, StaticAgent(bench_env, incumbent["lr"]), 30)
print(bench_env.get_performance())
bench_env.render_performance()