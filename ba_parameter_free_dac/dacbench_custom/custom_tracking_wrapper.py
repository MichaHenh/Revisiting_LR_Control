from dacbench.wrappers import PerformanceTrackingWrapper
from time import time

class CustomTrackingWrapper(PerformanceTrackingWrapper):

    def __init__(self, env, performance_interval=None, track_instance_performance=True, track_effective_lr=False, logger=None):
        super().__init__(env, performance_interval, track_instance_performance, logger)
        self.track_effective_lr = track_effective_lr

        self.training_losses = []
        self.avg_training_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.actions = []
        self.step_times = []
        self.effective_lrs = []
        
        self.last_step_start = None
        self.average_loss = None

    def step(self, action):
        print("step")
        
        current_time_ms = round(time()*1000)
        if self.last_step_start is not None:
            self.step_times.append(current_time_ms - self.last_step_start)
        self.last_step_start = current_time_ms

        state, reward, terminated, truncated, info = super().step(action)


        self.training_losses.append(self.loss)
        if(self.average_loss): self.avg_training_losses.append(self.average_loss)

        if self.epoch_mode or self.c_step % len(self.train_loader) == 0 or self._done:
            self.val_losses.append(self.validation_loss.item())
            self.val_accuracies.append(self.validation_accuracy.item())
            
        self.actions.append(action)
        if(self.track_effective_lr and self.optimizer.avg_effective_lr):
            self.effective_lrs.append(self.optimizer.avg_effective_lr.item())

        if terminated or truncated:
            if self.logger is not None:
                self.logger.log(
                    "actions",
                    self.actions,
                )
                self.logger.log(
                    "step_times",
                    self.step_times,
                )
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
                self.logger.log(
                    "effective_lrs",
                    self.effective_lrs,
                )

            self.training_losses = []
            self.avg_training_losses = []
            self.val_losses = []
            self.val_accuracies = []
            self.actions = []
            self.step_times = []

        return state, reward, terminated, truncated, info
