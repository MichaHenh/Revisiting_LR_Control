from dacbench.wrappers import PerformanceTrackingWrapper

class CustomTrackingWrapper(PerformanceTrackingWrapper):

    def __init__(self, env, performance_interval=None, track_instance_performance=True, logger=None):
        super().__init__(env, performance_interval, track_instance_performance, logger)
        self.step_losses = []

    def step(self, action):
        state, reward, terminated, truncated, info = super().step(action)

        self.step_losses.append(self.loss.mean().item())

        if terminated or truncated:
            if self.logger is not None:
                self.logger.log(
                    "step_losses",
                    self.step_losses,
                )

            self.step_losses = []

        return state, reward, terminated, truncated, info
