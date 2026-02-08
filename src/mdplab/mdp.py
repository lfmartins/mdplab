import numpy as np

class MDP:
    """
    Minimal representation of a Markov decision process.
    """

    def __init__(self, states, actions, transitions, terminal_states=[]):
        self._states = states.copy()
        self._n_states = len(states)
        self._actions = actions.copy()
        self._n_actions = len(actions)
        self._admissible_actions = {}

        self._state_indexes = {}
        for index, state in enumerate(states):
            if state in self._state_indexes:
                raise ValueError(f"Repeated state in parameter states: {state}")
            self._state_indexes[state] = index

        self._action_indexes = {}
        for index, action in enumerate(actions):
            if action in self._action_indexes:
                raise ValueError(f"Repeated action in paramter actions: {action}")
            self._action_indexes[action] = index

        self._transition_probs = np.zeros(
            shape=(self._n_actions, self._n_states, self._n_states),
            dtype=np.float64)

        self._rewards = np.zeros(
            shape=(self._n_actions, self._n_states, self._n_states),
            dtype=np.float64)


        for state, actions_dict in transitions.items():
            if state not in states:
                raise ValueError(f"Key {state} is not a valid state "
                                  "in parameter transitions.")
            if state in terminal_states:
                raise ValueError(f"Key {state} is a terminal state in "
                                  "parameter transitions")
            j = self._state_indexes[state]
            self._admissible_actions[state] = set()
            for action, transition_list in actions_dict.items():
                if action not in self._actions:
                    raise ValueError(f"Key {action} not valid for state {state} "
                                      "in parameter transitions.")
                if action in self._admissible_actions:
                    raise ValueError(f"Action {action} is repeated for key {state} "
                                      "in parameter transitions.")
                self._admissible_actions[state].add(action)
                i = self._action_indexes[action]

                last_transition = False
                sum_probs = 0.0
                target_states = set()
                for target_state, p, r in transition_list:
                    if target_state not in states:
                        raise ValueError(f"In transition list for state {state}, action {action}: "
                                         f"invalid state, {target_state}")
                    if target_state in target_states:
                        raise ValueError(f"In transition list for state {state}, action {action}: "
                                         f"state {target_state} repeated")
                    k = self._state_indexes[target_state]

                    if p < 0:
                        raise ValueError(f"In transition list for state {state}, action {action}: "
                                          "negative probability {p}")
                    if last_transition:
                        raise ValueError(f"In transition list for state {state}, action {action}: "
                                          "no transitions allowed after probability \"*\"")
                    if p == '*':
                        p = 1 - sum_probs
                        if p < 0:
                            raise ValueError(f"In transition list for state {state}, action {action}: "
                                              "probabilities add to more than 1")
                        last_transition = True

                    self._transition_probs[i, j, k] = p
                    self._rewards[i, j, k] = r
                if not last_transition:
                    raise ValueError(f"In transition list for state{state}, action {action}: "
                                      "probability for last transition must be \"*\"")


alpha = 0.8
beta = 0.3
rsearch = 15
rwait = 10
rempty = -3.0
print(f"Model parameters and discount :\n{alpha=}, {beta=}, {rsearch=}, {rwait=}, {rempty=}")

states = ["high", "low"]
actions = ["search", "wait", "recharge"]
transitions = {
    "high": {
        "search" : [("high", alpha, rsearch), ("low", "*", rsearch)],
        "wait" : (("high", "*", rwait)),
    },
    "low": {
        "search" : [("low", beta, rsearch), ("high", "*", rempty)],
        "wait": [("low", "*", rwait)],
        "recharge:": [("high", "*", )],
    }
}

rr_model = MDP(states, actions, transitions)




