from ai_cat.environments.grid_world import Environment
from ai_cat.agents.agent import Agent


environment = Environment()
agent = Agent()

episodes = 1000
max_steps = 100

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

    if (episode + 1) % 100 == 0:
        print(
            f"Episódio: {episode + 1} | "
            f"Passos: {step + 1} | "
            f"Exploração: {agent.exploration_rate:.3f}"
        )
