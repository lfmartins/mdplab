import pytest
from mdplab.mdp import MDP
from mdplab.mdp_fixtures import recycling_robot_mdp

def test_recycling_robot_creation(recycling_robot_mdp: MDP):
    """
    Test basic properties of the recycling robot MDP.
    """
    mdp = recycling_robot_mdp

    # States and actions
    assert set(mdp.states) == {"high", "low"}
    assert set(mdp.actions) == {"search", "wait", "recharge"}

    # Terminal states (none for recycling robot)
    assert len(mdp.terminal_states) == 0

    # Admissible actions
    assert set(mdp.admissible_actions["high"]) == {"search", "wait"}
    assert set(mdp.admissible_actions["low"]) == {"search", "wait", "recharge"}

    # Transition probabilities sum to 1
    for state in mdp.states:
        for action in mdp.admissible_actions[state]:
            probs = mdp.P(state, action)
            assert abs(probs.sum() - 1.0) < 1e-12

    # Rewards are numbers
    for state in mdp.states:
        for action in mdp.admissible_actions[state]:
            rewards = mdp.R(state, action)
            assert all(isinstance(r, float) for r in rewards)
