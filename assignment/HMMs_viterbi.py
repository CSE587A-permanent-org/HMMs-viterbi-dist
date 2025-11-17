"""
Viterbi Decoding for Hidden Markov Models

This module implements the Viterbi algorithm for finding the most likely
state sequence given an observation sequence and an HMM.
"""

from typing import List
import numpy as np
from numpy.typing import NDArray


def viterbi_decode(observation_seq: List[int], hmm) -> List[str]:
    """
    Decode an observation sequence using the Viterbi algorithm.

    INPUT:
    - observation_seq: List of observations (integers indexing alphabet)
    - hmm: HMM object with states, initial_state_probs, transition_matrix,
      and emission_matrix

    OUTPUT:
    - state_seq: List containing the most likely state sequence (state names)

    IMPLEMENTATION NOTES:
    1) The Viterbi table is implemented as a matrix that is transposed
       relative to the way it is shown in class. Rows correspond to
       observations and columns to states.
    2) After computing the Viterbi probabilities for each observation,
       they are normalized by dividing by their sum. This maintains
       proportionality while avoiding numerical underflow.
    """
    numeric_state_seq = _traceback(
        _build_matrix(observation_seq, hmm),
        hmm
    )
    return [hmm.states[i] for i in numeric_state_seq]

def _build_matrix(observation_seq: List[int], hmm) -> NDArray[np.float64]:
    """
    Build the Viterbi probability matrix.

    Returns a matrix where rows are observations and columns are states.
    Each entry is normalized to avoid underflow.
    """
    number_of_observations = len(observation_seq)
    number_of_states = hmm.num_states

    # Initialize Viterbi matrix
    viterbi_matrix = np.zeros((number_of_observations, number_of_states))

    # YOUR CODE HERE
    # <snip>
    # First observation: initial state probs * emission probs
    viterbi_matrix[0] = (
        np.array(hmm.initial_state_probs) *
        np.array(hmm.emission_matrix)[observation_seq[0]]
    )
    # Normalize to avoid underflow
    viterbi_matrix[0] = viterbi_matrix[0] / np.sum(viterbi_matrix[0])

    # Fill in the rest of the matrix
    for obs_idx in range(1, number_of_observations):
        # For each state, compute:
        # emission_prob * max(prev_state_prob * transition_prob)
        emission_probs = np.array(
            hmm.emission_matrix)[observation_seq[obs_idx]]

        # Compute all possible transitions from previous states
        # prev_probs * transition_matrix gives matrix where:
        # - rows are source states
        # - columns are destination states
        # We take the max over source states for each destination
        prev_probs = viterbi_matrix[obs_idx - 1]
        transition_matrix = np.array(hmm.transition_matrix)

        # Multiply each row of transition matrix by corresponding prev prob
        transitions = prev_probs[:, np.newaxis] * transition_matrix

        # Take max over source states (rows) for each dest state (column)
        max_transitions = np.max(transitions, axis=0)

        # Multiply by emission probabilities
        viterbi_matrix[obs_idx] = emission_probs * max_transitions

        # Normalize to avoid underflow
        viterbi_matrix[obs_idx] = (viterbi_matrix[obs_idx] /
                                    np.sum(viterbi_matrix[obs_idx]))
    # </snip>
    return viterbi_matrix


def _traceback(viterbi_matrix: NDArray[np.float64], hmm) -> List[int]:
    """
    Trace back through the Viterbi matrix to find the most likely path.

    Returns a list of state indices (integers) corresponding to the
    most likely state sequence.
    """
    number_of_observations = len(viterbi_matrix)
    state_seq = np.zeros(number_of_observations, dtype=int)

    # YOUR CODE HERE
    # <snip>

    # Start from the last observation - pick state with highest probability
    state_seq[number_of_observations - 1] = _max_position(
        viterbi_matrix[number_of_observations - 1]
    )

    # Trace back through the matrix
    for obs_idx in range(number_of_observations - 2, -1, -1):
        # For each previous observation, find the state that maximizes:
        # viterbi_prob * transition_prob_to_next_state
        next_state = state_seq[obs_idx + 1]
        transition_probs = np.array(hmm.transition_matrix)[:, next_state]
        combined_probs = viterbi_matrix[obs_idx] * transition_probs
        state_seq[obs_idx] = _max_position(combined_probs)
    # </snip>

    return state_seq.tolist()

# Use this function to find the index within an array of the maximum value.
# Do not use any built-in functions for this.
# This implementation chooses the lowest index in case of ties.
# Two values are considered tied if they are within a factor of 1E-5.
def _max_position(list_of_numbers: NDArray[np.float64]) -> int:
    """
    Find the index of the maximum value in a list.

    Returns the first index if there are ties or extremly close values.
    """
    max_value = 1E-10
    max_position = 0

    for i, value in enumerate(list_of_numbers):
        # This handles extremely close values that arise from numerical instability
        if value / max_value > 1 + 1E-5:
            max_value = value
            max_position = i

    return max_position
