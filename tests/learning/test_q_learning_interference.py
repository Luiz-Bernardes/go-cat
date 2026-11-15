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


def test_negative_rewards_can_reduce_previously_learned_value():
    learning = QLearning(
        actions=["up", "down", "left", "right"]
    )

    state = ("shared_state",)
    next_state = ("next_state",)

    learning.q_table[state] = {
        "up": 0.0,
        "down": -1.0,
        "left": 0.0,
        "right": -0.0224,
    }

    learning.q_table[next_state] = {
        "up": 0.0,
        "down": 0.0,
        "left": 0.0,
        "right": 0.0,
    }

    initial_value = learning.q_table[state]["right"]

    learning.learn(
        state=state,
        action="right",
        reward=-1,
        next_state=next_state,
        done=False,
    )

    updated_value = learning.q_table[state]["right"]

    assert updated_value < initial_value


def test_invalid_action_can_dominate_next_q_value():
    learning = QLearning(
        actions=["up", "down", "left", "right"]
    )

    state = ("current_state",)
    next_state = ("next_state",)

    learning.q_table[next_state] = {
        "up": 0.0,       # Parede
        "down": -0.5,    # Movimento válido
        "left": -0.8,
        "right": -0.9,
    }

    learning.learn(
        state=state,
        action="right",
        reward=-1,
        next_state=next_state,
        done=False,
    )

    # O algoritmo atual considera "up",
    # mesmo sendo uma ação inválida no cenário.
    expected_target = -1 + learning.discount_factor * 0.0

    assert expected_target == -1.0

    # Com Q inicial igual a zero:
    expected_q = learning.learning_rate * expected_target

    assert learning.q_table[state]["right"] == expected_q


def test_learning_uses_only_valid_next_actions():
    learning = QLearning(
        actions=["up", "down", "left", "right"]
    )

    state = ("current_state",)
    next_state = ("next_state",)

    learning.q_table[next_state] = {
        "up": 0.0,       # Parede
        "down": -0.5,
        "left": -0.8,
        "right": -0.9,
    }

    learning.learn(
        state=state,
        action="right",
        reward=-1,
        next_state=next_state,
        done=False,
        valid_next_actions=["down", "left", "right"],
    )

    # A melhor ação válida é "down", com Q = -0.5
    expected_target = (
        -1 + learning.discount_factor * -0.5
    )

    expected_q = learning.learning_rate * expected_target

    assert learning.q_table[state]["right"] == expected_q
