import pytest
from mdplab import MDP

@pytest.fixture
def recycling_robot_mdp():
    """
    Test basic properties of the recycling robot MDP.
    """
    alpha = 0.8
    beta = 0.6
    rsearch = 15.0
    rwait = 10.0
    rempty = -3

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
            "recharge": [("high", "*", 0.0)]
        }
    }

    return MDP(states, actions, transitions)


