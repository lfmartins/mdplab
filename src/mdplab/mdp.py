import numpy as np

from typing import (
    Hashable, Dict, List, Tuple, Union, Iterable
)

State = Hashable
Action = Hashable

Probability = Union[float, str]   # '*' allowed
Reward = float

Transition = Tuple[State, Probability, Reward]
ActionTransitions = Dict[Action, List[Transition]]
Transitions = Dict[State, ActionTransitions]


class MDP:
    """
    Minimal representation of a Markov decision process.
    """

    def __init__(self,
                 states: List[State],
                 actions: List[Action],
                 transitions: Transitions,
                 terminal_states: Iterable[State] = ()) -> None:
        self._states = states.copy()
        self._n_states = len(states)
        self._actions = actions.copy()
        self._n_actions = len(actions)
        self._terminal_states = set(terminal_states)
        self._admissible_actions = {}

        for state in self._terminal_states:
            self._admissible_actions[state]  = set()

        self._state_indexes = {}
        for index, state in enumerate(states):
            if state in self._state_indexes:
                raise ValueError(f"Repeated state in parameter states: {state}")
            self._state_indexes[state] = index

        self._action_indexes = {}
        for index, action in enumerate(actions):
            if action in self._action_indexes:
                raise ValueError(f"Repeated action in parameter actions: {action}")
            self._action_indexes[action] = index

        self._transition_probs = np.zeros(
            shape=(self._n_actions, self._n_states, self._n_states),
            dtype=np.float64)

        self._rewards = np.zeros(
            shape=(self._n_actions, self._n_states, self._n_states),
            dtype=np.float64)

        all_states = set(self._state_indexes)
        non_terminal_states = all_states - self._terminal_states
        transition_states = set(transitions)
        missing = non_terminal_states - transition_states
        if missing:
            raise ValueError(f"Missing transition specification for non-terminal "
                              "states: {missing}")

        tol = 1e-12
        for state, actions_dict in transitions.items():
            if state not in self._state_indexes:
                raise ValueError(f"Invalid state {state} in transitions.")
            if state in self._terminal_states:
                raise ValueError(f"Terminal state {state} cannot have transitions")

            j = self._state_indexes[state]
            self._admissible_actions[state] = set()

            for action, transition_list in actions_dict.items():
                if action not in self._action_indexes:
                    raise ValueError(f"Key {action} not valid for state {state} "
                                      "in parameter transitions.")
                if action in self._admissible_actions[state]:
                    raise ValueError(f"Action {action} is repeated for key {state} "
                                      "in parameter transitions.")
                self._admissible_actions[state].add(action)
                i = self._action_indexes[action]

                last_transition = False
                sum_probs = 0.0
                seen_targets = set()
                for target_state, p, r in transition_list:
                    if last_transition:
                        raise ValueError(f"In transition list for state {state}, action {action}: "
                                         f"No transitions allowed after a transition has probability '*'")
                    if target_state not in self._states:
                        raise ValueError(f"In transition list for state {state}, action {action}: "
                                         f"invalid state, {target_state}")
                    if target_state in seen_targets:
                        raise ValueError(f"In transition list for state {state}, action {action}: "
                                         f"state {target_state} repeated")

                    seen_targets.add(target_state)
                    k = self._state_indexes[target_state]

                    if p == '*':
                        p = 1 - sum_probs
                        if p < -tol:
                            raise ValueError(f"In transition list for state {state}, action {action}: "
                                              "probabilities add to more than 1")
                        last_transition = True
                    elif not isinstance(p, (int, float)):
                        raise TypeError(f"In transition list fr state {state}, action {action}: "
                                        f"transition probability to state {target_state} must be an number or '*'")
                    elif p < 0:
                        raise ValueError(f"In transition list for state {state}, action {action}: "
                                         f"transition probability to state {target_state} is negative")
                    else:
                        sum_probs += p

                    self._transition_probs[i, j, k] = p
                    self._rewards[i, j, k] = r

                if not last_transition:
                    raise ValueError(f"In transition list for state {state}, action {action}: "
                                      "probability for last transition must be '*'")

                if abs(sum_probs + p - 1.0) > tol:
                    raise ValueError(f"In transition list for state {state}, action {action}: "
                                      "transition probabilities do not add to 1")






