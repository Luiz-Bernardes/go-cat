from environment import Environment
from agent import Agent


environment = Environment()
agent = Agent()

episodes = 1000
max_steps = 100

# =========================
# TREINAMENTO
# =========================

for episode in range(episodes):
    environment.reset()
    agent.position = [0, 0]

    for step in range(max_steps):
        state = tuple(agent.position)

        action = agent.choose_action(state)

        reward, done = environment.step(action)

        agent.position = list(environment.cat_position)

        next_state = tuple(agent.position)

        agent.learn(
            state,
            action,
            reward,
            next_state
        )

        if done:
            break

    agent.decay_exploration()


# =========================
# TESTE
# =========================

print("\n==============================")
print(" TESTANDO A IA TREINADA")
print("==============================\n")

agent.exploration_rate = 0

environment.reset()
agent.position = [0, 0]

environment.display()

for step in range(max_steps):
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

    environment.display()

    if done:
        print("🍎 A IA encontrou a comida!")
        break
