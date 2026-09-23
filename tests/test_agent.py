from agent import Agent


agent = Agent()

state = (0, 0)

print("Antes de qualquer experiência:")
print(agent.get_q_values(state))

print("\nNúmero de estados conhecidos:")
print(len(agent.q_table))
