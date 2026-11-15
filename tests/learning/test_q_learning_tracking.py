from ai_cat.learning.q_learning import QLearning


def test_q_learning_records_updates_for_monitored_action():
    learning = QLearning(
        actions=["up", "down", "left", "right"]
    )

    state = ("initial_state",)
    next_state = ("next_state",)

    learning.start_tracking(
        state=state,
        action="right",
    )

    learning.learn(
        state=state,
        action="right",
        reward=-1,
        next_state=next_state,
        done=False,
    )

    updates = learning.tracked_updates

    assert len(updates) == 1

    assert updates[0]["state"] == state
    assert updates[0]["action"] == "right"
    assert updates[0]["reward"] == -1
    assert updates[0]["new_value"] < updates[0]["old_value"]
    assert updates[0]["target"] == -1
    assert updates[0]["td_error"] < 0

    assert updates[0]["next_state"] == next_state

    assert updates[0]["next_q_values"] == {
        "up": 0.0,
        "down": 0.0,
        "left": 0.0,
        "right": 0.0,
    }


def test_q_learning_ignores_other_actions_when_tracking():
    learning = QLearning(
        actions=["up", "down", "left", "right"]
    )

    state = ("initial_state",)
    next_state = ("next_state",)

    learning.start_tracking(
        state=state,
        action="right",
    )

    learning.learn(
        state=state,
        action="down",
        reward=-1,
        next_state=next_state,
        done=False,
    )

    assert learning.tracked_updates == []
