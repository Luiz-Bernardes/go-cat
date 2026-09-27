from ai_cat.environments.grid_world import Environment
from ai_cat.memory.memory import Memory


def test_memory_sliding_window_with_real_perceptions():
    environment = Environment()
    memory = Memory(capacity=3)

    perceptions = []

    # P1
    perception = environment.get_perception()
    perceptions.append(perception)
    memory.add(perception)

    # P2
    environment.step("down")
    perception = environment.get_perception()
    perceptions.append(perception)
    memory.add(perception)

    # P3
    environment.step("down")
    perception = environment.get_perception()
    perceptions.append(perception)
    memory.add(perception)

    print()
    print("Após P3:")
    for index, item in enumerate(memory.get(), start=1):
        print(f"{index}: {item}")

    assert memory.get() == (
        perceptions[0],
        perceptions[1],
        perceptions[2],
    )

    # P4
    environment.step("down")
    perception = environment.get_perception()
    perceptions.append(perception)
    memory.add(perception)

    print()
    print("Após P4:")
    for index, item in enumerate(memory.get(), start=1):
        print(f"{index}: {item}")

    # P1 deve ter saído.
    assert memory.get() == (
        perceptions[1],
        perceptions[2],
        perceptions[3],
    )

    assert perceptions[0] not in memory.get()