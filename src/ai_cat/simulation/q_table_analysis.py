from copy import deepcopy


def snapshot_q_table(agent):
    return deepcopy(agent.learning.q_table)


def compare_q_tables(before, after):
    shared_states = set(before) & set(after)

    changed_states = 0
    changed_actions = 0

    for state in shared_states:
        state_changed = False

        actions = set(before[state]) & set(after[state])

        for action in actions:
            if before[state][action] != after[state][action]:
                changed_actions += 1
                state_changed = True

        if state_changed:
            changed_states += 1

    return {
        "states_before": len(before),
        "states_after": len(after),
        "shared_states": len(shared_states),
        "changed_states": changed_states,
        "changed_actions": changed_actions,
    }
