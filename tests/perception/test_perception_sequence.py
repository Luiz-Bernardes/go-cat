from ai_cat.agents.agent import Agent
from ai_cat.perception.perception import PerceptionType

def test_sequence_contains_information_missing_from_current_perception():
    agent = Agent()

    empty = (
        PerceptionType.EMPTY,
        PerceptionType.EMPTY,
        PerceptionType.EMPTY,
        PerceptionType.EMPTY,
    )

    food_right = (
        PerceptionType.EMPTY,
        PerceptionType.EMPTY,
        PerceptionType.EMPTY,
        PerceptionType.FOOD,
    )

    food_left = (
        PerceptionType.EMPTY,
        PerceptionType.EMPTY,
        PerceptionType.FOOD,
        PerceptionType.EMPTY,
    )

    # Sequência A:
    # O agente viu comida à direita.
    # Depois a comida desapareceu da percepção.
    # Agora continua vendo apenas espaço vazio.
    sequence_a = (
        food_right,
        empty,
        empty,
    )

    # Sequência B:
    # O agente viu comida à esquerda.
    # Depois a comida desapareceu da percepção.
    # Agora continua vendo apenas espaço vazio.
    sequence_b = (
        food_left,
        empty,
        empty,
    )

    print()
    print("Sequência A:", sequence_a)
    print("Sequência B:", sequence_b)

    print()
    print("Percepção atual A:", sequence_a[-1])
    print("Percepção atual B:", sequence_b[-1])

    print()
    print("Percepção anterior A:", sequence_a[-2])
    print("Percepção anterior B:", sequence_b[-2])

    # A percepção atual é exatamente igual.
    assert sequence_a[-1] == sequence_b[-1]

    # A percepção imediatamente anterior também é igual.
    assert sequence_a[-2] == sequence_b[-2]

    # Mas a sequência completa é diferente.
    assert sequence_a != sequence_b

    # Treinamos o significado da sequência.
    #
    # Se anteriormente vimos comida à direita,
    # devemos escolher direita.
    #
    # Se anteriormente vimos comida à esquerda,
    # devemos escolher esquerda.
    for _ in range(100):
        agent.learn(
            sequence_a,
            "right",
            10,
            sequence_a,
        )

        agent.learn(
            sequence_b,
            "left",
            10,
            sequence_b,
        )

    agent.learning.exploration_rate = 0

    action_a = agent.choose_action(sequence_a)
    action_b = agent.choose_action(sequence_b)

    print()
    print("Ação aprendida na sequência A:", action_a)
    print("Ação aprendida na sequência B:", action_b)

    assert action_a == "right"
    assert action_b == "left"