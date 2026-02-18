import pytest
import numpy as np

def test_public_api_import():
    from mdplab import MDP

    assert MDP is not None

