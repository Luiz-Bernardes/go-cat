from ai_cat.environments.grid_world import Environment
from ai_cat.agents.agent import Agent

def test_trained_agent_finds_food():
    environment = Environment()
    agent = Agent()

    episodes = 1000
    max_steps = 100

    # Treinamento
    for _ in range(episodes):
        environment.reset()
        agent.position = [0, 0]

        for _ in range(max_steps):
            state = tuple(agent.position)

            action = agent.choose_action(state)

            reward, done = environment.step(action)

            agent.position = list(environment.cat_position)

            next_state = tuple(agent.position)

            agent.learn(
                state,
                action,
                reward,
                next_state,
            )

            if done:
                break

        agent.decay_exploration()

    # Teste
    agent.learning.exploration_rate = 0

    environment.reset()
    agent.position = [0, 0]

    found_food = False

    for _ in range(max_steps):
        state = tuple(agent.position)

        action = agent.choose_action(state)

        reward, done = environment.step(action)

        agent.position = list(environment.cat_position)

        if done:
            found_food = True
            break

    assert found_food is True
    assert tuple(agent.position) in environment.foods
    assert reward == 10