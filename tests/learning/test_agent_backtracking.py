from ai_cat.agents.agent import Agent
from ai_cat.environments.grid_world import Environment


def test_agent_can_choose_backtracking_when_necessary():
    environment = Environment(
        size=3,
        foods={(2, 2)},
        obstacles={(1, 1), (2, 0)},
    )

    agent = Agent()
    agent.learning.exploration_rate = 0

    # O gato começa em (0, 0).
    agent.observe(environment)

    # Avança para (1, 0).
    environment.step("right")
    agent.observe(environment)

    state = agent.get_state(environment)

    # A única saída física é voltar para (0, 0).
    assert environment.get_valid_actions() == ["left"]

    # O agente deve conseguir escolher esse retorno.
    assert agent.choose_action(state) == "left"
