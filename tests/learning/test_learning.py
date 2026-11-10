from ai_cat.agents.agent import Agent


def test_agent_learns_from_bad_experience():
    agent = Agent()

    state = (0, 0)
    next_state = (0, 1)

    agent.learn(
        state=state,
        action="down",
        reward=-1,
        next_state=next_state,
    )

    q_values = agent.get_q_values(state)

    assert q_values["down"] == -0.1


def test_agent_learns_from_good_experience():
    agent = Agent()

    state = (0, 0)
    next_state = (0, 1)

    agent.learn(
        state=state,
        action="right",
        reward=10,
        next_state=next_state,
    )

    q_values = agent.get_q_values(state)

    assert q_values["right"] == 1.0


def test_agent_does_not_bootstrap_terminal_state():
    agent = Agent()

    state = ("initial",)
    next_state = ("terminal",)

    agent.learning.q_table[state] = {
        "up": 0.0,
        "down": 0.0,
        "left": 0.0,
        "right": 0.0,
    }

    agent.learning.q_table[next_state] = {
        "up": 100.0,
        "down": 100.0,
        "left": 100.0,
        "right": 100.0,
    }

    agent.learn(
        state,
        "right",
        10,
        next_state,
        done=True,
    )

    expected_q = agent.learning.learning_rate * 10

    assert agent.learning.q_table[state]["right"] == expected_q
