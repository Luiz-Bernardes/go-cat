import numpy as np
from ai_cat.environments.grid_world import Environment

def test_environment_initial_position():
    environment = Environment()

    assert list(environment.cat_position) == [0, 0]


def test_environment_initial_food_position():
    environment = Environment()

    assert environment.foods == {
        (7, 7),
        (2, 6),
        (8, 2),
    }

def test_environment_supports_multiple_foods():
    environment = Environment()

    environment.foods.add((2, 2))

    assert (7, 7) in environment.foods
    assert (2, 6) in environment.foods
    assert (8, 2) in environment.foods
    assert (2, 2) in environment.foods

def test_environment_reaches_any_food():
    environment = Environment()

    environment.foods = {
        (2, 2),
        (7, 7),
    }

    environment.cat_position = np.array([0, 2])

    reward, done = environment.step("right")
    assert done is False

    reward, done = environment.step("right")

    assert list(environment.cat_position) == [2, 2]
    assert reward == 10
    assert done is True

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

def test_environment_blocks_obstacle():
    environment = Environment()

    environment.cat_position = [2, 1]

    reward, done = environment.step("right")

    assert list(environment.cat_position) == [2, 1]
    assert reward == -1
    assert done is False