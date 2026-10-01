from ai_cat.agents.agent import Agent
from ai_cat.perception.perception import PerceptionType


def test_memory_state_allows_different_actions():

    agent = Agent()

    current_perception = (
        PerceptionType.EMPTY,
        PerceptionType.EMPTY,
        PerceptionType.EMPTY,
        PerceptionType.EMPTY,
    )

    previous_perception_a = (
        PerceptionType.WALL,
        PerceptionType.EMPTY,
        PerceptionType.WALL,
        PerceptionType.EMPTY,
    )

    previous_perception_b = (
        PerceptionType.WALL,
        PerceptionType.EMPTY,
        PerceptionType.EMPTY,
        PerceptionType.WALL,
    )

    state_a = (
        previous_perception_a,
        current_perception,
    )

    state_b = (
        previous_perception_b,
        current_perception,
    )

    print()
    print("Estado A:")
    print(state_a)

    print()
    print("Estado B:")
    print(state_b)

    assert state_a != state_b

    # Aprende que, nesse contexto, a ação correta é right.
    for _ in range(100):
        agent.learn(
            state_a,
            "right",
            10,
            state_a,
        )

    # Aprende que, nesse outro contexto, a ação correta é left.
    for _ in range(100):
        agent.learn(
            state_b,
            "left",
            10,
            state_b,
        )

    # Desliga exploração para avaliar somente o conhecimento aprendido.
    agent.learning.exploration_rate = 0

    action_a = agent.choose_action(state_a)
    action_b = agent.choose_action(state_b)

    print()
    print("Ação aprendida no estado A:", action_a)
    print("Ação aprendida no estado B:", action_b)

    assert action_a == "right"
    assert action_b == "left"