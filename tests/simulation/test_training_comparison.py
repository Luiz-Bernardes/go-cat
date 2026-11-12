from ai_cat.simulation.training_comparison import (
    train_fixed_agent,
    train_varied_agent,
)


def test_training_strategies_support_multiple_seeds():
    seeds = [10, 20, 30]

    for seed in seeds:
        fixed_agent = train_fixed_agent(seed=seed)
        varied_agent = train_varied_agent(seed=seed)

        assert len(fixed_agent.learning.q_table) > 0
        assert len(varied_agent.learning.q_table) > 0

        assert fixed_agent.learning.exploration_rate == 0
        assert varied_agent.learning.exploration_rate == 0
