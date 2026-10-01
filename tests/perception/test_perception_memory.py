from ai_cat.agents.agent import Agent
from ai_cat.perception.perception import PerceptionType

def test_memory_allows_different_actions_for_same_perception():
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
    print("Estado A:", state_a)
    print("Estado B:", state_b)

    assert state_a != state_b

    # Treina o agente:
    #
    # Estado A -> direita é uma boa ação
    # Estado B -> esquerda é uma boa ação

    for _ in range(100):
        agent.learn(
            state_a,
            "right",
            10,
            state_a,
        )

        agent.learn(
            state_b,
            "left",
            10,
            state_b,
        )

    agent.learning.exploration_rate = 0

    action_a = agent.choose_action(state_a)
    action_b = agent.choose_action(state_b)

    print()
    print("Ação aprendida em A:", action_a)
    print("Ação aprendida em B:", action_b)

    assert action_a == "right"
    assert action_b == "left"