from ai_cat.environments.grid_world import Environment


def test_environment_initial_position():
    environment = Environment()

    assert list(environment.cat_position) == [0, 0]


def test_environment_initial_food_position():
    environment = Environment()

    assert list(environment.food_position) == [7, 7]


def test_environment_size():
    environment = Environment()

    assert environment.size == 10


def test_environment_reset():
    environment = Environment()

    environment.step("right")
    environment.step("down")

    environment.reset()

    assert list(environment.cat_position) == [0, 0]


def test_environment_reaches_food():
    environment = Environment()

    for _ in range(7):
        environment.step("right")

    for _ in range(7):
        reward, done = environment.step("down")

    assert list(environment.cat_position) == [7, 7]
    assert reward == 10
    assert done is True