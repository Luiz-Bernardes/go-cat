from ai_cat.environments.grid_world import Environment


def test_get_valid_actions_at_initial_position():
    environment = Environment(
        size=5,
        foods={(4, 4)},
        obstacles=set(),
    )

    valid_actions = environment.get_valid_actions()

    assert valid_actions == ["down", "right"]


def test_get_valid_actions_ignores_obstacles():
    environment = Environment(
        size=5,
        foods={(4, 4)},
        obstacles={(0, 1)},
    )

    valid_actions = environment.get_valid_actions()

    assert valid_actions == ["right"]


def test_get_valid_actions_at_center():
    environment = Environment(
        size=5,
        foods={(4, 4)},
        obstacles=set(),
    )

    environment.cat_position[:] = [2, 2]

    valid_actions = environment.get_valid_actions()

    assert valid_actions == [
        "up",
        "down",
        "left",
        "right",
    ]
