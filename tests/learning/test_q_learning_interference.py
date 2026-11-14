from ai_cat.learning.q_learning import QLearning


def test_same_state_action_can_receive_conflicting_rewards():
    learning = QLearning(
        actions=["up", "down", "left", "right"]
    )

    state = ("same_state",)
    next_state = ("next_state",)
    action = "right"

    learning.learn(
        state=state,
        action=action,
        reward=10,
        next_state=next_state,
        done=True,
    )

    value_after_positive_reward = learning.get_q_values(state)[action]

    learning.learn(
        state=state,
        action=action,
        reward=-10,
        next_state=next_state,
        done=True,
    )

    value_after_negative_reward = learning.get_q_values(state)[action]

    assert value_after_positive_reward > 0

    assert value_after_negative_reward < value_after_positive_reward
