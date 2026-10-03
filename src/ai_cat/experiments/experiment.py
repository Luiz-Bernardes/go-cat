from environment import Environment
from agent import Agent


environment = Environment()
agent = Agent()

for step in range(50):
    state = tuple(environment.cat_position)

    action = agent.choose_action(state)

    reward, done = environment.step(action)

    if done:
        print("\n🍎 O gato encontrou a comida!")
        break
