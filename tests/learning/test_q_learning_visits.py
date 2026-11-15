from ai_cat.learning.q_learning import QLearning


def test_q_learning_counts_state_action_visits():
    learning = QLearning(
        actions=["up", "down", "left", "right"]
    )

    state = ("state_1",)
    next_state = ("state_2",)

    learning.learn(
        state=state,
        action="right",
        reward=-1,
        next_state=next_state,
    )

    learning.learn(
        state=state,
        action="right",
        reward=-1,
        next_state=next_state,
    )

    learning.learn(
        state=state,
        action="left",
        reward=-1,
        next_state=next_state,
    )

    assert learning.get_visit_count(state, "right") == 2
    assert learning.get_visit_count(state, "left") == 1
    assert learning.get_visit_count(state, "up") == 0
