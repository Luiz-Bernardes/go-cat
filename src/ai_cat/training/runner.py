class TrainingRunner:
    def __init__(self, environment, agent):
        self.environment = environment
        self.agent = agent

    def run(self, episodes=1000, max_steps=100):
        if episodes <= 0:
            raise ValueError(
                "Episodes must be greater than zero"
            )

        if max_steps <= 0:
            raise ValueError(
                "Maximum steps must be greater than zero"
            )

        for _ in range(episodes):
            self.environment.reset()
            self.agent.reset_memory()
            self.agent.observe(self.environment)

            for _ in range(max_steps):
                state = self.agent.get_state(self.environment)
                action = self.agent.choose_action(state)

                reward, done = self.environment.step(action)

                self.agent.observe(self.environment)
                next_state = self.agent.get_state(self.environment)

                self.agent.learn(
                    state,
                    action,
                    reward,
                    next_state,
                    done,
                )

                if done:
                    break

            self.agent.decay_exploration()
