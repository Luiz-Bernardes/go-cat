import random


class QLearning:
    def __init__(
        self,
        actions,
        learning_rate=0.1,
        discount_factor=0.9,
        exploration_rate=1.0,
        exploration_decay=0.995,
        min_exploration_rate=0.01,
    ):
        self.actions = actions
        self.q_table = {}

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate

        self.tracked_state = None
        self.tracked_action = None
        self.tracked_updates = []
        self.visit_counts = {}

    def get_q_values(self, state):
        if state not in self.q_table:
            self.q_table[state] = {
                action: 0.0
                for action in self.actions
            }

        return self.q_table[state]

    def choose_action(self, state):
        q_values = self.get_q_values(state)

        if random.random() < self.exploration_rate:
            return random.choice(self.actions)

        return max(q_values, key=q_values.get)

    def learn(
        self,
        state,
        action,
        reward,
        next_state,
        done=False,
        valid_next_actions=None,
    ):
        key = (state, action)

        self.visit_counts[key] = self.visit_counts.get(key, 0) + 1

        q_values = self.get_q_values(state)

        current_q = q_values[action]

        if done:
            target = reward
        else:
            next_q_values = self.get_q_values(next_state)

            if valid_next_actions is None:
                best_next_q = max(next_q_values.values())
            elif valid_next_actions:
                best_next_q = max(
                    next_q_values[action]
                    for action in valid_next_actions
                )
            else:
                best_next_q = 0.0

            target = (
                reward
                + self.discount_factor * best_next_q
            )

        new_q = current_q + self.learning_rate * (
            target - current_q
        )

        q_values[action] = new_q

        # Registra a atualização somente se estivermos
        # monitorando essa combinação de estado e ação.
        if (
            state == self.tracked_state
            and action == self.tracked_action
        ):
            self.tracked_updates.append({
                "state": state,
                "action": action,
                "reward": reward,
                "target": target,
                "td_error": target - current_q,
                "old_value": current_q,
                "new_value": new_q,
                "next_state": next_state,
                "next_q_values": self.get_q_values(next_state).copy(),
                "valid_next_actions": (
                    valid_next_actions.copy()
                    if valid_next_actions is not None
                    else None
                ),
            })


    def decay_exploration(self):
        self.exploration_rate = max(
            self.min_exploration_rate,
            self.exploration_rate * self.exploration_decay,
        )


    def start_tracking(self, state, action):
        self.tracked_state = state
        self.tracked_action = action
        self.tracked_updates = []


    def get_visit_count(self, state, action):
        return self.visit_counts.get((state, action), 0)