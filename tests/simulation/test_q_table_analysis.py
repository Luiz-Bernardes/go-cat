from ai_cat.simulation.q_table_analysis import compare_q_tables


def test_compare_q_tables_detects_modified_values():
    before = {
        ("state_a",): {
            "up": 1.0,
            "right": 5.0,
        },
        ("state_b",): {
            "up": 2.0,
            "right": 3.0,
        },
    }

    after = {
        ("state_a",): {
            "up": 1.0,
            "right": 2.0,
        },
        ("state_b",): {
            "up": 4.0,
            "right": 3.0,
        },
        ("state_c",): {
            "up": 0.0,
            "right": 1.0,
        },
    }

    result = compare_q_tables(before, after)

    assert result["states_before"] == 2
    assert result["states_after"] == 3
    assert result["shared_states"] == 2

    assert result["changed_states"] == 2
    assert result["changed_actions"] == 2
