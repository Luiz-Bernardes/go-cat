from ai_cat.agents.agent import Agent
from ai_cat.environments.grid_world import Environment


def test_different_environments_can_produce_same_state():
    environment_a = Environment(
        foods={(8, 8)},
        obstacles=set(),
    )

    environment_b = Environment(
        foods={(5, 5)},
        obstacles=set(),
    )

    agent_a = Agent()
    agent_b = Agent()

    agent_a.observe(environment_a)
    agent_b.observe(environment_b)

    state_a = agent_a.get_state(environment_a)
    state_b = agent_b.get_state(environment_b)

    assert state_a == state_b
