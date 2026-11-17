"""Log-space Viterbi Dynamic Programming Algorithm for Map-Matching.

Member D — `d-rf-hmm`
Implements numerically stable log-space Viterbi recurrence, backpointer tracking,
and optimal sequence backtracking for state spaces that vary per observation.
"""
from __future__ import annotations

from typing import Any, Callable, Sequence
import numpy as np


class ViterbiResult:
    """Encapsulates Viterbi sequence decoding results."""

    def __init__(
        self,
        best_sequence: list[int],
        best_log_score: float,
        state_log_probs: list[np.ndarray],
        backpointers: list[np.ndarray],
        chain_breaks: list[int],
    ) -> None:
        self.best_sequence = best_sequence
        self.best_log_score = float(best_log_score)
        self.state_log_probs = state_log_probs
        self.backpointers = backpointers
        self.chain_breaks = chain_breaks

    def to_dict(self) -> dict[str, Any]:
        return {
            "decoded_route": self.best_sequence,
            "sequence_log_score": round(self.best_log_score, 4),
            "n_breaks": len(self.chain_breaks),
            "chain_breaks": self.chain_breaks,
        }


def viterbi_decode(
    candidate_states: Sequence[Sequence[int]],
    log_emissions: Sequence[np.ndarray | Sequence[float]],
    transition_log_prob_fn: Callable[[int, int, int], float],
    initial_log_probs: np.ndarray | Sequence[float] | None = None,
    chain_break_flags: Sequence[bool] | None = None,
    disconnected_penalty: float = -50.0,
) -> ViterbiResult:
    """Perform exact log-space Viterbi dynamic programming.
    
    Parameters:
      candidate_states: Sequence of length T, where candidate_states[t] is a sequence
                        of candidate arc IDs for observation t.
      log_emissions: Sequence of length T, where log_emissions[t] has length len(candidate_states[t]).
      transition_log_prob_fn: Callable (t, from_state_id, to_state_id) -> log_prob.
      initial_log_probs: Prior log probabilities for observation 0. Defaults to uniform.
      chain_break_flags: Sequence of booleans indicating forced chain breaks (e.g. large GPS gaps).
      disconnected_penalty: Floor log penalty when transitions are disconnected.
      
    Returns:
      ViterbiResult with optimal sequence of arc IDs and log score.
    """
    T = len(candidate_states)
    if T == 0:
        return ViterbiResult(best_sequence=[], best_log_score=0.0, state_log_probs=[], backpointers=[], chain_breaks=[])

    # Defensive validation of inputs
    cleaned_states: list[list[int]] = []
    cleaned_emissions: list[np.ndarray] = []

    for t in range(T):
        states_t = list(candidate_states[t])
        emiss_t = np.array(log_emissions[t], dtype=np.float64)

        if len(states_t) == 0:
            # Empty candidate set fallback: dummy state 0 with 0.0 log emission
            states_t = [0]
            emiss_t = np.array([0.0], dtype=np.float64)
        elif len(emiss_t) != len(states_t):
            raise ValueError(
                f"Step {t}: Candidate count ({len(states_t)}) does not match emission count ({len(emiss_t)})"
            )

        cleaned_states.append(states_t)
        cleaned_emissions.append(emiss_t)

    # Initialise t = 0
    K_0 = len(cleaned_states[0])
    if initial_log_probs is None:
        v_curr = np.full(K_0, -np.log(K_0), dtype=np.float64) + cleaned_emissions[0]
    else:
        init_arr = np.array(initial_log_probs, dtype=np.float64)
        if len(init_arr) != K_0:
            init_arr = np.full(K_0, -np.log(K_0), dtype=np.float64)
        v_curr = init_arr + cleaned_emissions[0]

    all_v_probs: list[np.ndarray] = [v_curr.copy()]
    backpointers: list[np.ndarray] = []
    chain_breaks: list[int] = []

    # Forward dynamic programming pass
    for t in range(1, T):
        prev_states = cleaned_states[t - 1]
        curr_states = cleaned_states[t]
        K_prev = len(prev_states)
        K_curr = len(curr_states)

        # Check if chain break was requested (e.g. large GPS time gap)
        is_break = chain_break_flags[t] if chain_break_flags is not None else False

        bp_t = np.zeros(K_curr, dtype=np.int32)
        v_next = np.zeros(K_curr, dtype=np.float64)

        if is_break:
            # Reset prior to uniform, decoupling from previous step
            chain_breaks.append(t)
            v_next = np.full(K_curr, -np.log(K_curr), dtype=np.float64) + cleaned_emissions[t]
            bp_t.fill(0)
        else:
            # Compute transition matrix Trans[j, k] = log A(prev_states[j] -> curr_states[k])
            # v_next[k] = max_j (v_curr[j] + Trans[j, k]) + log_emission[k]
            all_disconnected = True
            for k in range(K_curr):
                to_state = curr_states[k]
                scores = np.zeros(K_prev, dtype=np.float64)

                for j in range(K_prev):
                    from_state = prev_states[j]
                    log_trans = transition_log_prob_fn(t, from_state, to_state)
                    if log_trans > disconnected_penalty:
                        all_disconnected = False
                    scores[j] = v_curr[j] + log_trans

                best_j = int(np.argmax(scores))
                bp_t[k] = best_j
                v_next[k] = scores[best_j] + cleaned_emissions[t][k]

            # If all candidate pairs are completely disconnected and penalty is too severe,
            # trigger graceful chain break
            if all_disconnected and np.all(v_next < -1e5):
                chain_breaks.append(t)
                v_next = np.full(K_curr, -np.log(K_curr), dtype=np.float64) + cleaned_emissions[t]
                bp_t.fill(0)

        v_curr = v_next
        all_v_probs.append(v_curr.copy())
        backpointers.append(bp_t)

    # Termination: identify best state at final step T-1
    best_last_idx = int(np.argmax(v_curr))
    best_log_score = float(v_curr[best_last_idx])

    # Backward pass: trace backpointers
    best_seq_indices = [0] * T
    best_seq_indices[-1] = best_last_idx

    for t in range(T - 1, 0, -1):
        prev_idx = int(backpointers[t - 1][best_seq_indices[t]])
        best_seq_indices[t - 1] = prev_idx

    best_sequence = [cleaned_states[t][best_seq_indices[t]] for t in range(T)]

    return ViterbiResult(
        best_sequence=best_sequence,
        best_log_score=best_log_score,
        state_log_probs=all_v_probs,
        backpointers=backpointers,
        chain_breaks=chain_breaks,
    )
