"""Unit tests for Viterbi dynamic programming (Phase D).

Verifies mathematical correctness on hand-verifiable toy trellises and edge cases.
"""
from __future__ import annotations

import pytest
import numpy as np

from member_d.viterbi import viterbi_decode, ViterbiResult


def test_single_observation():
    """T = 1 should select the candidate with highest emission."""
    states = [[10, 20, 30]]
    log_emissions = [[-2.0, -0.1, -1.5]]  # state 20 has highest score (-0.1)
    
    def dummy_trans(t, u, v):
        return 0.0

    res = viterbi_decode(states, log_emissions, dummy_trans)
    assert res.best_sequence == [20]
    assert pytest.approx(res.best_log_score, rel=1e-3) == -np.log(3) - 0.1


def test_hand_calculated_two_step():
    """Hand-calculated 2-step trellis where transition cost overrides naive emission choice.
    
    Step 0: candidates [A=1, B=2]
      log_emiss: A=-0.1 (high), B=-2.0 (low)
    Step 1: candidates [C=3, D=4]
      log_emiss: C=-2.0, D=-0.1 (high)
      
    Transitions:
      A -> C: valid (cost = 0.0)
      A -> D: impossible (cost = -100.0)
      B -> C: impossible (cost = -100.0)
      B -> D: valid (cost = 0.0)
      
    Total paths:
      A -> C: (-log2 - 0.1) + 0.0 + (-2.0) = -log2 - 2.1 = -2.793
      A -> D: (-log2 - 0.1) - 100.0 + (-0.1) = -100.893 (penalised)
      B -> D: (-log2 - 2.0) + 0.0 + (-0.1) = -log2 - 2.1 = -2.793 (tie with A->C!)
    
    If B has slightly higher emission (-1.8):
      B -> D: (-log2 - 1.8) + 0.0 + (-0.1) = -log2 - 1.9 = -2.593 -> Winner is [B, D]!
      Notice naive classifier would pick [A, D], which is topologically impossible!
    """
    states = [[1, 2], [3, 4]]
    log_emiss = [[-0.1, -1.8], [-2.0, -0.1]]

    def trans_fn(t, u, v):
        if (u, v) in [(1, 3), (2, 4)]:
            return 0.0
        return -100.0

    res = viterbi_decode(states, log_emiss, trans_fn)
    # Naive emission would pick [1, 4], but (1, 4) is impossible!
    # Viterbi must pick connected sequence [2, 4]
    assert res.best_sequence == [2, 4]


def test_empty_candidate_set():
    """Empty candidate set should fallback defensively to dummy state."""
    states = [[1, 2], [], [3, 4]]
    log_emiss = [[-0.5, -0.5], [], [-0.1, -2.0]]

    def trans_fn(t, u, v):
        return -1.0

    res = viterbi_decode(states, log_emiss, trans_fn)
    assert len(res.best_sequence) == 3
    assert res.best_sequence[1] == 0  # dummy state inserted defensively


def test_chain_break_flag():
    """Forced chain break should decouple transition and reset prior."""
    states = [[1, 2], [3, 4]]
    log_emiss = [[-0.1, -2.0], [-2.0, -0.1]]

    # All transitions are heavily penalised (-100)
    def trans_fn(t, u, v):
        return -100.0

    # Without break, penalty applies
    res_no_break = viterbi_decode(states, log_emiss, trans_fn, chain_break_flags=[False, False])
    # With break at step 1, transition is skipped
    res_break = viterbi_decode(states, log_emiss, trans_fn, chain_break_flags=[False, True])

    assert 1 in res_break.chain_breaks
    assert res_break.best_log_score > res_no_break.best_log_score
    # When decoupled, naive top emissions [1, 4] win cleanly
    assert res_break.best_sequence == [1, 4]


def test_tie_breaking_deterministic():
    """Symmetric trellis must produce deterministic reproducible sequence."""
    states = [[1, 2], [3, 4]]
    log_emiss = [[-1.0, -1.0], [-1.0, -1.0]]

    def trans_fn(t, u, v):
        return 0.0

    res1 = viterbi_decode(states, log_emiss, trans_fn)
    res2 = viterbi_decode(states, log_emiss, trans_fn)
    assert res1.best_sequence == res2.best_sequence
    assert res1.best_log_score == res2.best_log_score
