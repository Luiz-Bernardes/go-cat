import pytest

from ai_cat.simulation.multi_seed_forgetting_experiment import summarize


def test_summarize_forgetting_results():
    results = [
        {
            "seed": 10,
            "before_steps": 8,
            "after_steps": 14,
            "before_success": 1.0,
            "after_success": 1.0,
        },
        {
            "seed": 20,
            "before_steps": 20,
            "after_steps": 12,
            "before_success": 1.0,
            "after_success": 1.0,
        },
        {
            "seed": 30,
            "before_steps": 10,
            "after_steps": 10,
            "before_success": 1.0,
            "after_success": 0.0,
        },
    ]

    summary = summarize(results)

    assert summary["worse"] == 1
    assert summary["better"] == 1
    assert summary["unchanged"] == 1
    assert summary["success_losses"] == 1
    assert summary["mean_step_delta"] == pytest.approx(-2 / 3)
