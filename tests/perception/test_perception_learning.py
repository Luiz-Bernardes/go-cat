from ai_cat.environments.grid_world import Environment
from ai_cat.agents.agent import Agent

def test_agent_learns_using_only_perception():
    environment = Environment()
    agent = Agent()

    episodes = 1000
    max_steps = 100

    for _ in range(episodes):
        environment.reset()

        for _ in range(max_steps):
            state = environment.get_perception_state()

            action = agent.choose_action(state)

            reward, done = environment.step(action)

            next_state = environment.get_perception_state()

            agent.learn(
                state,
                action,
                reward,
                next_state,
            )

            if done:
                break

        agent.decay_exploration()

    agent.learning.exploration_rate = 0

    successful_runs = 0
    total_steps = 0
    test_runs = 100

    for _ in range(test_runs):
        environment.reset()

        for step in range(max_steps):
            state = environment.get_perception_state()

            action = agent.choose_action(state)

            reward, done = environment.step(action)

            if done:
                successful_runs += 1
                total_steps += step + 1
                break

    print()
    print(f"Testes: {test_runs}")
    print(f"Sucessos: {successful_runs}")
    print(f"Falhas: {test_runs - successful_runs}")

    if successful_runs:
        print(
            f"Média de passos: "
            f"{total_steps / successful_runs:.2f}"
        )

    assert successful_runs > 0