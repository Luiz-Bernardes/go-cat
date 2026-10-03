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
        agent.memory.reset()
        agent.observe(environment)

        for _ in range(max_steps):
            state = agent.get_state()

            action = agent.choose_action(state)

            reward, done = environment.step(action)

            agent.observe(environment)
            next_state = agent.get_state()
            
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
    agent.memory.reset()
    agent.observe(environment)

    found_food = False

    for _ in range(max_steps):
        state = agent.get_state()

        action = agent.choose_action(state)

        reward, done = environment.step(action)

        if done:
            found_food = True
            break

        agent.observe(environment)

    assert found_food is True
    assert tuple(environment.cat_position) in environment.foods
    assert reward == 10

def test_trained_agent_finds_shortest_path_to_food():
    random.seed(42)

    environment = Environment()
    agent = Agent()
    agent.state.capacity = 3

    episodes = 1000
    max_steps = 100

    # Treinamento
    for _ in range(episodes):
        environment.reset()
        configure_experiment_environment(environment)

        agent.memory.reset()
        agent.observe(environment)

        for _ in range(max_steps):
            state = agent.get_state()

            action = agent.choose_action(state)

            old_position = tuple(environment.cat_position)

            reward, done = environment.step(action)

            new_position = tuple(environment.cat_position)

            if new_position != old_position:
                agent.observe(environment)

            next_state = agent.get_state()

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

    agent.memory.reset()
    agent.observe(environment)

    path = []

    for _ in range(max_steps):
        state = agent.get_state()

        perception = environment.get_perception()
        q_values = agent.get_q_values(state)

        action = agent.choose_action(state)

        old_position = tuple(environment.cat_position)

        reward, done = environment.step(action)

        new_position = tuple(environment.cat_position)

        path.append(new_position)

        print(
            f"{len(path):02d} | "
            f"{action:5} | "
            f"pos={tuple(environment.cat_position)} | "
            f"state={state} | "
            f"perception={perception} | "
            f"Q={q_values}"
        )

        if done:
            break

        if new_position != old_position:
            agent.observe(environment)

    assert tuple(environment.cat_position) == (0, 5)
    assert len(path) == 5
    assert reward == 10