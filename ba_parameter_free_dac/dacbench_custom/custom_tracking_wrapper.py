from dacbench.wrappers import PerformanceTrackingWrapper

class CustomTrackingWrapper(PerformanceTrackingWrapper):

    def __init__(self, env, performance_interval=None, track_instance_performance=True, logger=None):
        super().__init__(env, performance_interval, track_instance_performance, logger)
        self.training_losses = []
        self.avg_training_losses = []
        self.val_losses = []
        self.val_accuracies = []

    def step(self, action):
        print("step")
        state, reward, terminated, truncated, info = super().step(action)

        self.training_losses.append(self.loss)
        self.avg_training_losses.append(self.average_loss)
        self.val_losses.append(self.validation_loss.item())
        self.val_accuracies.append(self.validation_accuracy.item())

        if terminated or truncated:
            if self.logger is not None:
                self.logger.log(
                    "training_losses",
                    self.training_losses,
                )
                self.logger.log(
                    "avg_training_losses",
                    self.avg_training_losses,
                )
                self.logger.log(
                    "validation_losses",
                    self.val_losses,
                )
                self.logger.log(
                    "validation_accuracies",
                    self.val_accuracies,
                )
                self.logger.log(
                    "test_loss",
                    self.test_losses.mean().item(),
                )
                self.logger.log(
                    "test_accuracy",
                    self.test_accuracies.mean().item(),
                )

            self.training_losses = []
            self.avg_training_losses = []
            self.val_losses = []
            self.val_accuracies = []

        return state, reward, terminated, truncated, info
