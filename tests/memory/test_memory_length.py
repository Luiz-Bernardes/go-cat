from ai_cat.agents.agent import Agent
from ai_cat.perception.perception import PerceptionType

def build_memory(sequence, size):
    return tuple(sequence[-size:])

def test_minimum_memory_length():
    agent = Agent()

    empty = (
        PerceptionType.EMPTY,
        PerceptionType.EMPTY,
        PerceptionType.EMPTY,
        PerceptionType.EMPTY,
    )

    signal_right = (
        PerceptionType.EMPTY,
        PerceptionType.EMPTY,
        PerceptionType.EMPTY,
        PerceptionType.FOOD,
    )

    signal_left = (
        PerceptionType.EMPTY,
        PerceptionType.EMPTY,
        PerceptionType.FOOD,
        PerceptionType.EMPTY,
    )

    # As duas situações possuem o mesmo final.
    #
    # A única diferença está no primeiro elemento.
    sequence_a = (
        signal_right,
        empty,
        empty,
    )

    sequence_b = (
        signal_left,
        empty,
        empty,
    )

    print()
    print("Sequência A:", sequence_a)
    print("Sequência B:", sequence_b)

    # ---------------------------------------------------------
    # Memória de 1 percepção
    # ---------------------------------------------------------

    memory_1_a = build_memory(sequence_a, 1)
    memory_1_b = build_memory(sequence_b, 1)

    print()
    print("Memória 1 A:", memory_1_a)
    print("Memória 1 B:", memory_1_b)

    assert memory_1_a == memory_1_b

    # ---------------------------------------------------------
    # Memória de 2 percepções
    # ---------------------------------------------------------

    memory_2_a = build_memory(sequence_a, 2)
    memory_2_b = build_memory(sequence_b, 2)

    print()
    print("Memória 2 A:", memory_2_a)
    print("Memória 2 B:", memory_2_b)

    assert memory_2_a == memory_2_b

    # ---------------------------------------------------------
    # Memória de 3 percepções
    # ---------------------------------------------------------

    memory_3_a = build_memory(sequence_a, 3)
    memory_3_b = build_memory(sequence_b, 3)

    print()
    print("Memória 3 A:", memory_3_a)
    print("Memória 3 B:", memory_3_b)

    assert memory_3_a != memory_3_b

    # ---------------------------------------------------------
    # Treinamento
    #
    # Com memória de 1:
    # A e B são exatamente o mesmo estado.
    #
    # Portanto não existe como aprender duas ações diferentes.
    # ---------------------------------------------------------

    for _ in range(100):
        agent.learn(
            memory_1_a,
            "right",
            10,
            memory_1_a,
        )

        agent.learn(
            memory_1_b,
            "left",
            10,
            memory_1_b,
        )

    # ---------------------------------------------------------
    # Agora treinamos usando memória de 3.
    #
    # Aqui A e B são estados diferentes.
    # ---------------------------------------------------------

    for _ in range(100):
        agent.learn(
            memory_3_a,
            "right",
            10,
            memory_3_a,
        )

        agent.learn(
            memory_3_b,
            "left",
            10,
            memory_3_b,
        )

    agent.learning.exploration_rate = 0

    action_memory_3_a = agent.choose_action(memory_3_a)
    action_memory_3_b = agent.choose_action(memory_3_b)

    print()
    print("Ação com memória 3 - A:", action_memory_3_a)
    print("Ação com memória 3 - B:", action_memory_3_b)

    assert action_memory_3_a == "right"
    assert action_memory_3_b == "left"