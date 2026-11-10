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

        cycle_active = False

        for _ in range(max_steps):
            state = self.agent.get_state(self.environment)
            action = self.agent.choose_action(state)

            reward, done = self.environment.step(action)

            metrics.record_step()

            self.agent.observe(self.environment)

            metrics.record_position(
                self.environment.cat_position
            )

            cycle_detected = (
                self.agent.movement_history.get_cycle_length() is not None
            )

            if cycle_detected:
                metrics.record_cycle()

                if not cycle_active:
                    metrics.record_cycle_event()

            cycle_active = cycle_detected


            if done:
                metrics.record_food()
                break

        return metrics
