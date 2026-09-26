import random

from ai_cat.environments.grid_world import Environment
from ai_cat.agents.agent import Agent

def configure_experiment_environment(environment):
    environment.foods = {
        (4, 0),
        (0, 5),
    }

    environment.obstacles = {
        (1, 0),
        (2, 0),
        (3, 0),
    }

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

def test_trained_agent_finds_shortest_path_to_food():
    random.seed(42)

    environment = Environment()
    agent = Agent()

    episodes = 1000
    max_steps = 100

    # Treinamento
    for _ in range(episodes):
        environment.reset()
        configure_experiment_environment(environment)

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

    # Teste sem exploração
    agent.learning.exploration_rate = 0

    environment.reset()
    configure_experiment_environment(environment)

    agent.position = [0, 0]

    path = []

    for _ in range(max_steps):
        state = tuple(agent.position)

        action = agent.choose_action(state)

        reward, done = environment.step(action)

        agent.position = list(environment.cat_position)

        path.append(tuple(agent.position))

        if done:
            break

    assert tuple(agent.position) == (0, 5)
    assert len(path) == 5
    assert reward == 10