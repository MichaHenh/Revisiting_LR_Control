import torch
import torch.optim as optim
from parameterfree.cocob_optimizer import COCOB
from parameterfree.code_optimizer import CODE

# Define the function to be minimized
def function_to_minimize(x):
    return (x - 3)**2

# Initialize the parameter to be optimized
x = torch.tensor([2.0], requires_grad=True)  # Start with x=0

# Initialize the optimizer (SGD in this case)
optimizer = CODE([x])

# Number of optimization steps
num_steps = 1000

for step in range(num_steps):

    def closure():
        optimizer.zero_grad()
        ls = function_to_minimize(x)
        ls.backward()
        return ls


    # Zero the gradients
    optimizer.zero_grad()

    # Compute the loss
    loss = function_to_minimize(x)

    # Perform backpropagation to compute gradients
    loss.backward()

    # Update the parameter
    optimizer.step(closure)

    # Print the progress
    if step % 10 == 0:
        print(f'Step {step}, x = {x.item()}, loss = {loss.item()}')

print(f'Optimized x = {x.item()}, loss = {function_to_minimize(x).item()}')
