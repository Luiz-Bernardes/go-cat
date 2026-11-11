from ai_cat.agents.agent import Agent
from ai_cat.environments.grid_world import Environment
from ai_cat.training.runner import TrainingRunner
from ai_cat.simulation.runner import SimulationRunner
from ai_cat.simulation.evaluator import SimulationEvaluator


def run_experiment():
    environment = Environment()
    agent = Agent()

    training_runner = TrainingRunner(environment, agent)
    simulation_runner = SimulationRunner(environment, agent)
    evaluator = SimulationEvaluator(simulation_runner)

    # Avaliação sem exploração aleatória.
    original_exploration_rate = agent.learning.exploration_rate
    agent.learning.exploration_rate = 0

    before = evaluator.evaluate(
        episodes=100,
        max_steps=100,
    )

    # Restaurar exploração para treinamento.
    agent.learning.exploration_rate = original_exploration_rate

    training_runner.run(
        episodes=1000,
        max_steps=100,
    )

    # Avaliação final sem exploração aleatória.
    agent.learning.exploration_rate = 0

    after = evaluator.evaluate(
        episodes=100,
        max_steps=100,
    )

    print("\n=== BEFORE TRAINING ===")
    for key, value in before.items():
        print(f"{key}: {value}")

    print("\n=== AFTER TRAINING ===")
    for key, value in after.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    run_experiment()
