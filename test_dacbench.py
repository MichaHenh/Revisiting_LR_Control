from dacbench.runner import run_benchmark
from dacbench.benchmarks import SGDBenchmark
from dacbench.agents import RandomAgent

# Function to create an agent fulfilling the DACBench Agent interface
# In this case: a simple random agent
def make_agent(env):
    return RandomAgent(env)


# Result output path
path = "dacbench_tabular"

bench_env = SGDBenchmark().get_benchmark()

# Run SGD benchmark
run_benchmark(bench_env, make_agent, 2)