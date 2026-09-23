from agent import Agent


agent = Agent()

state = (0, 0)

agent.q_table[state] = {
    "up": 1.0,
    "down": 2.0,
    "left": 0.5,
    "right": 10.0
}

agent.exploration_rate = 0.0

for _ in range(10):
    print(agent.choose_action(state))
