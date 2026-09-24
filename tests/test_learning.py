from ai_cat.agents.agent import Agent

agent = Agent()

state = (0, 0)
next_state = (0, 1)

print("Antes:")
print(agent.get_q_values(state))

agent.learn(
    state=state,
    action="down",
    reward=-1,
    next_state=next_state
)

print("\nDepois de uma experiência ruim:")
print(agent.get_q_values(state))

agent.learn(
    state=state,
    action="right",
    reward=10,
    next_state=next_state
)

print("\nDepois de uma experiência boa:")
print(agent.get_q_values(state))
