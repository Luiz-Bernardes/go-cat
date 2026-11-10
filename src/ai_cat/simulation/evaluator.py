class SimulationEvaluator:
    def __init__(self, runner):
        self.runner = runner

    def evaluate(self, episodes=100, max_steps=100):
        if episodes <= 0:
            raise ValueError(
                "Episodes must be greater than zero"
            )

        if max_steps <= 0:
            raise ValueError(
                "Maximum steps must be greater than zero"
            )

        results = []

        for _ in range(episodes):
            metrics = self.runner.run(max_steps=max_steps)
            results.append(metrics.summary())

        successful_episodes = sum(
            result["foods_found"] > 0
            for result in results
        )

        return {
            "episodes": episodes,
            "successful_episodes": successful_episodes,
            "success_rate": successful_episodes / episodes,
            "average_steps": sum(
                result["steps"] for result in results
            ) / episodes,
            "average_unique_positions": sum(
                result["unique_positions"] for result in results
            ) / episodes,
            "average_cycles_detected": sum(
                result["cycles_detected"] for result in results
            ) / episodes,
            "average_cycle_events": sum(
                result["cycle_events"] for result in results
            ) / episodes,
        }
