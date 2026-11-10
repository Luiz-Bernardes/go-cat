from ai_cat.simulation.metrics import SimulationMetrics


def test_metrics_start_empty():
    metrics = SimulationMetrics()

    assert metrics.summary() == {
        "steps": 0,
        "unique_positions": 0,
        "foods_found": 0,
        "cycles_detected": 0,
        "cycle_events": 0,
    }


def test_records_steps():
    metrics = SimulationMetrics()

    metrics.record_step()
    metrics.record_step()
    metrics.record_step()

    assert metrics.steps == 3


def test_counts_unique_positions():
    metrics = SimulationMetrics()

    metrics.record_position((0, 0))
    metrics.record_position((1, 0))
    metrics.record_position((0, 0))

    assert metrics.unique_positions == 2


def test_records_food_found():
    metrics = SimulationMetrics()

    metrics.record_food()
    metrics.record_food()

    assert metrics.foods_found == 2


def test_records_detected_cycles():
    metrics = SimulationMetrics()

    metrics.record_cycle()
    metrics.record_cycle()

    assert metrics.cycles_detected == 2


def test_reset_clears_all_metrics():
    metrics = SimulationMetrics()

    metrics.record_step()
    metrics.record_position((0, 0))
    metrics.record_food()
    metrics.record_cycle()

    metrics.reset()

    assert metrics.summary() == {
        "steps": 0,
        "unique_positions": 0,
        "foods_found": 0,
        "cycles_detected": 0,
        "cycle_events": 0,
    }
