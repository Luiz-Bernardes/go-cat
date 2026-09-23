from environment import Environment
from agent import Agent


environment = Environment()
agent = Agent()

for step in range(50):
    state = tuple(agent.position)

    action = agent.choose_action(state)

    reward, done = environment.step(action)

    agent.position = list(environment.cat_position)

    print(
        f"Passo: {step + 1:02d} | "
        f"Ação: {action:5s} | "
        f"Posição: {agent.position} | "
        f"Recompensa: {reward}"
    )

    if done:
        print("\n🍎 O gato encontrou a comida!")
        break
