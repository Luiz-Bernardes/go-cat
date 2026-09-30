from ai_cat.environments.grid_world import Environment
from ai_cat.memory.memory import Memory
from ai_cat.perception.perception import Perception


def test_memory_stores_real_environment_perceptions():
    environment = Environment()
    memory = Memory(capacity=3)

    # Percepção inicial
    perception = environment.get_perception()
    memory.add(perception)

    print()
    print("Percepção 1:", perception)

    # Movimento 1
    environment.step("down")

    perception = environment.get_perception()
    memory.add(perception)

    print("Percepção 2:", perception)

    # Movimento 2
    environment.step("down")

    perception = environment.get_perception()
    memory.add(perception)

    print("Percepção 3:", perception)

    print()
    print("Memória:")

    for index, item in enumerate(memory.get(), start=1):
        print(f"{index}: {item}")

    # A memória deve conter exatamente as três percepções.
    assert len(memory) == 3

    stored_perceptions = memory.get()

    # Primeira percepção
    assert stored_perceptions[0] == Perception(
        up="wall",
        down="empty",
        left="wall",
        right="empty",
    )

    # Segunda percepção
    assert stored_perceptions[1] == Perception(
        up="empty",
        down="empty",
        left="wall",
        right="empty",
    )

    # Terceira percepção
    assert stored_perceptions[2] == Perception(
        up="empty",
        down="empty",
        left="wall",
        right="empty",
    )