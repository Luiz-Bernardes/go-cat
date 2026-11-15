from math import isclose


def record_trajectory(agent, environment, max_steps=100):
    if max_steps <= 0:
        raise ValueError("Maximum steps must be greater than zero")

    previous_exploration = agent.learning.exploration_rate

    try:
        agent.learning.exploration_rate = 0

        environment.reset()
        agent.reset_memory()
        agent.observe(environment)

        trajectory = [{
            "step": 0,
            "position": tuple(environment.cat_position),
            "action": None,
        }]

        for step in range(1, max_steps + 1):
            state = agent.get_state(environment)
            action = agent.choose_action(state)

            reward, done = environment.step(action)

            agent.observe(environment)

            trajectory.append({
                "step": step,
                "position": tuple(environment.cat_position),
                "action": action,
            })

            if done:
                break

        return trajectory

    finally:
        agent.learning.exploration_rate = previous_exploration


def find_first_divergence(before, after):
    for step_before, step_after in zip(before, after):
        if (
            step_before["position"] != step_after["position"]
            or step_before["action"] != step_after["action"]
        ):
            return {
                "step": step_before["step"],
                "before": step_before,
                "after": step_after,
            }

    if len(before) != len(after):
        return {
            "step": min(len(before), len(after)),
            "before": before[min(len(before), len(after))]
            if len(before) > len(after) else None,
            "after": after[min(len(before), len(after))]
            if len(after) > len(before) else None,
        }

    return None


def inspect_initial_decision(agent, environment):
    previous_exploration = agent.learning.exploration_rate

    try:
        agent.learning.exploration_rate = 0

        environment.reset()
        agent.reset_memory()
        agent.observe(environment)

        state = agent.get_state(environment)

        q_values = agent.get_q_values(state).copy()

        action = agent.choose_action(state)

        return {
            "state": state,
            "q_values": q_values,
            "chosen_action": action,
        }

    finally:
        agent.learning.exploration_rate = previous_exploration


def find_best_actions(q_values, valid_actions):
    if not valid_actions:
        raise ValueError("Valid actions cannot be empty")

    best_value = max(
        q_values[action]
        for action in valid_actions
    )

    return [
        action
        for action in valid_actions
        if isclose(
            q_values[action],
            best_value,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
    ]
