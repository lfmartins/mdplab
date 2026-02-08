import pytest
import numpy as np
from mdplab.mdp import MDP

def test_mdp_creation():
    alpha = 0.8
    beta = 0.3
    rsearch = 15
    rwait = 10
    rempty = -3.0

    states = ["high", "low"]
    actions = ["search", "wait", "recharge"]
    transitions = {
        "high": {
            "search": [("high", alpha, rsearch), ("low", "*", rsearch)],
            "wait": [("high", "*", rwait)],
        },
        "low": {
            "search": [("low", beta, rsearch), ("high", "*", rempty)],
            "wait": [("low", "*", rwait)],
            "recharge": [("high", "*", 0.0)],
        }
    }

    mdp = MDP(states, actions, transitions)

    # Basic assertions
    assert mdp._n_states == 2
    assert mdp._n_actions == 3
    assert "search" in mdp._admissible_actions["high"]
    assert np.isclose(mdp._transition_probs[0, 0, 0], alpha)
    assert np.isclose(mdp._transition_probs[0, 0, 1], 1 - alpha)
