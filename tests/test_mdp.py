# tests/test_mdp.py
import math
import pytest
from mdplab import MDP
from tests.mdp_fixtures import recycling_robot_mdp  # fixture import

def test_public_api_import():
    """
    Sanity check: user can import MDP from the public API.
    """
    assert MDP is not None

def test_recycling_robot_creation(recycling_robot_mdp: MDP):
    """
    Test basic properties of the recycling robot MDP.
    """
    mdp = recycling_robot_mdp

    # States and actions
    assert set(mdp.states) == {"high", "low"}
    assert set(mdp.actions) == {"search", "wait", "recharge"}

    # Terminal states
    assert len(mdp.terminal_states) == 0

    # Admissible actions
    for state, actions in mdp.admissible_actions.items():
        for action in actions:
            assert action in mdp.actions

    # Transition probabilities sum to 1
    for state in mdp.states:
        for action in mdp.admissible_actions[state]:
            probs = mdp.P(state, action)
            assert math.isclose(probs.sum(), 1.0), f"P sum != 1 for {state}, {action}"

    # Rewards are numbers
    for state in mdp.states:
        for action in mdp.admissible_actions[state]:
            rewards = mdp.R(state, action)
            assert all(isinstance(r, float) for r in rewards), f"Rewards not floats for {state}, {action}"
