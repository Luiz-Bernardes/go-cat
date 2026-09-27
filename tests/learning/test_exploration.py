from ai_cat.agents.agent import Agent

def test_agent_exploits_best_action_when_exploration_is_zero():
    agent = Agent()

    state = (0, 0)

    agent.learning.q_table[state] = {
        "up": 1.0,
        "down": 2.0,
        "left": 0.5,
        "right": 10.0,
    }

    agent.learning.exploration_rate = 0.0

    for _ in range(10):
        assert agent.choose_action(state) == "right"