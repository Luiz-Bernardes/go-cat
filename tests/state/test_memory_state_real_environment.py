from ai_cat.agents.agent import Agent
from ai_cat.environments.grid_world import Environment
from ai_cat.learning.state import State
from ai_cat.memory.memory import Memory


def test_real_environment_memory_state_can_be_learned():

    environment = Environment()
    memory = Memory(capacity=3)
    agent = Agent()
    state_builder = State(memory, capacity=3)

    # Primeira percepção real do ambiente.
    perception = environment.get_perception()
    memory.add(perception)

    state_1 = state_builder.build()

    print()
    print("Percepção 1:")
    print(perception)

    print()
    print("Estado após P1:")
    print(state_1)

    # Avança o ambiente.
    environment.step("down")

    perception = environment.get_perception()
    memory.add(perception)

    state_2 = state_builder.build()

    print()
    print("Percepção 2:")
    print(perception)

    print()
    print("Estado após P2:")
    print(state_2)

    # Avança novamente.
    environment.step("down")

    perception = environment.get_perception()
    memory.add(perception)

    state_3 = state_builder.build()

    print()
    print("Percepção 3:")
    print(perception)

    print()
    print("Estado após P3:")
    print(state_3)

    # Os estados devem ser diferentes conforme a memória evolui.
    assert state_1 != state_2
    assert state_2 != state_3

    # A estrutura deve permanecer com tamanho fixo.
    assert len(state_1) == 3
    assert len(state_2) == 3
    assert len(state_3) == 3

    # Todos os estados devem ser hashable.
    hash(state_1)
    hash(state_2)
    hash(state_3)

    # O Q-Learning deve conseguir utilizar os estados reais.
    agent.learn(
        state_1,
        "down",
        -1,
        state_2,
    )

    agent.learn(
        state_2,
        "down",
        -1,
        state_3,
    )

    print()
    print("Q-values do estado 1:")
    print(agent.get_q_values(state_1))

    print()
    print("Q-values do estado 2:")
    print(agent.get_q_values(state_2))

    assert state_1 in agent.learning.q_table
    assert state_2 in agent.learning.q_table
    assert state_3 in agent.learning.q_table