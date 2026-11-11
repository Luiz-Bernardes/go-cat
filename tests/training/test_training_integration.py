from ai_cat.agents.agent import Agent
from ai_cat.environments.grid_world import Environment
from ai_cat.training.runner import TrainingRunner


def test_training_updates_q_values():
    environment = Environment()
    agent = Agent()

    runner = TrainingRunner(environment, agent)

    runner.run(
        episodes=100,
        max_steps=100,
    )

    q_table = agent.learning.q_table

    assert len(q_table) > 0

    assert any(
        value != 0
        for actions in q_table.values()
        for value in actions.values()
    )