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