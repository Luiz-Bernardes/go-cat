from ai_cat.agents.agent import Agent


def test_agent_creates_q_values_for_new_state():
    agent = Agent()

    state = (0, 0)

    q_values = agent.get_q_values(state)

    assert state in agent.q_table
    assert q_values == {
        "up": 0.0,
        "down": 0.0,
        "left": 0.0,
        "right": 0.0,
    }


def test_agent_starts_with_empty_q_table():
    agent = Agent()

    assert agent.q_table == {}