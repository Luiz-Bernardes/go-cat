from ai_cat.memory.spatial_observation import SpatialObservation


def test_spatial_observation_stores_visited_directions():
    observation = SpatialObservation(
        up_visited=True,
        down_visited=False,
        left_visited=True,
        right_visited=False,
    )

    assert observation.up_visited is True
    assert observation.down_visited is False
    assert observation.left_visited is True
    assert observation.right_visited is False


def test_spatial_observation_as_tuple():
    observation = SpatialObservation(
        up_visited=True,
        down_visited=False,
        left_visited=True,
        right_visited=False,
    )

    assert observation.as_tuple() == (
        True,
        False,
        True,
        False,
    )