from ai_cat.simulation.forgetting_experiment import (
    evaluate_agent,
)
from ai_cat.agents.agent import Agent
from ai_cat.environments.grid_world import Environment


def test_evaluation_preserves_exploration_rate():
    agent = Agent()

    environment = Environment(
        foods={(1, 0)},
        obstacles=set(),
    )

    agent.learning.exploration_rate = 0.5

    evaluate_agent(agent, environment)

    assert agent.learning.exploration_rate == 0.5
