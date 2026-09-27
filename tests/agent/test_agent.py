from ai_cat.agents.agent import Agent
from ai_cat.environments.grid_world import Environment

def test_agent_creates_q_values_for_new_state():
    agent = Agent()

    state = (0, 0)

    q_values = agent.get_q_values(state)

    assert state in agent.learning.q_table
    assert q_values == {
        "up": 0.0,
        "down": 0.0,
        "left": 0.0,
        "right": 0.0,
    }

def test_agent_starts_with_empty_q_table():
    agent = Agent()

    assert agent.learning.q_table == {}

def test_agent_gets_state_from_environment():
    environment = Environment()
    agent = Agent()

    environment.cat_position = [2, 1]

    state = agent.get_state(environment)

    assert state == (
        (2, 1),
        ("empty", "empty", "empty", "obstacle"),
    )