from ai_cat.simulation.metrics import SimulationMetrics


class SimulationRunner:
    def __init__(self, environment, agent):
        self.environment = environment
        self.agent = agent

    def run(self, max_steps=100):
        if max_steps <= 0:
            raise ValueError(
                "Maximum steps must be greater than zero"
            )

        metrics = SimulationMetrics()

        self.environment.reset()
        self.agent.reset_memory()
        self.agent.observe(self.environment)

        metrics.record_position(
            self.environment.cat_position
        )

        for _ in range(max_steps):
            state = self.agent.get_state(self.environment)
            action = self.agent.choose_action(state)

            reward, done = self.environment.step(action)

            metrics.record_step()

            self.agent.observe(self.environment)

            metrics.record_position(
                self.environment.cat_position
            )

            if self.agent.movement_history.get_cycle_length() is not None:
                metrics.record_cycle()

            if done:
                metrics.record_food()
                break

        return metrics
