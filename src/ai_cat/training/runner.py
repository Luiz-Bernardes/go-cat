class TrainingRunner:
    def __init__(self, environment=None, agent=None, environments=None):
        if environments is not None:
            if not environments:
                raise ValueError("Environments cannot be empty")

            self.environments = list(environments)

        elif environment is not None:
            self.environments = [environment]

        else:
            raise ValueError("An environment is required")

        if agent is None:
            raise ValueError("An agent is required")

        self.environment = self.environments[0]
        self.agent = agent

    def run(self, episodes=1000, max_steps=100):
        if episodes <= 0:
            raise ValueError("Episodes must be greater than zero")

        if max_steps <= 0:
            raise ValueError("Maximum steps must be greater than zero")

        for episode in range(episodes):
            environment = self.environments[
                episode % len(self.environments)
            ]

            environment.reset()

            self.agent.reset_memory()
            self.agent.observe(environment)

            for _ in range(max_steps):
                state = self.agent.get_state(environment)
                action = self.agent.choose_action(state)

                reward, done = environment.step(action)

                self.agent.observe(environment)

                next_state = self.agent.get_state(environment)

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
