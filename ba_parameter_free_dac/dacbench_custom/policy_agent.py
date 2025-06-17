from dacbench.abstract_agent import AbstractDACBenchAgent

class PolicyAgent(AbstractDACBenchAgent):
    r"""Agent representing a Policy. Every time act is called, we take one step in policy.
    Args:
        env: DACBench environment
        policy: array of policy values
    """

    def __init__(self, env, policy):
        self.policy = policy
        self.act_counter = -1
        super().__init__(env)

    def act(self, state=None, reward=None):
        self.act_counter += 1
        return self.policy[self.act_counter]
    
    def reconfigure(self, policy):
        self.policy = policy

    def reset(self):
        self.act_counter = -1

    def train(self, state=None, reward=None):  # noqa: D102
        pass

    def end_episode(self, state=None, reward=None):  # noqa: D102
        pass

    def __str__(self) -> str:
        return str(self.policy)