from dacbench.runner import run_benchmark
from parameterfree.parameter_free_sgd_benchmark import ParameterFreeSGDBenchmark
from dacbench.agents import RandomAgent
from dacbench.wrappers import PerformanceTrackingWrapper
from dacbench.benchmarks import SGDBenchmark
import parameterfree.cocob_optimizer as cocob_optimizer

# Result output path
path = "dacbench_tabular"

#bench_env = PerformanceTrackingWrapper(ParameterFreeSGDBenchmark(cocob_optimizer.COCOB).get_benchmark())
bench_env = PerformanceTrackingWrapper(SGDBenchmark().get_benchmark())

# Run SGD benchmark
run_benchmark(bench_env, RandomAgent(bench_env), 30)
print(bench_env.get_performance())
bench_env.render_performance()