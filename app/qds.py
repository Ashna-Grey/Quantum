from __future__ import annotations

import numpy as np

I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

PAULI = {"X": X, "Y": Y, "Z": Z}


def normalize(state: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(state)
    if norm == 0:
        raise ValueError("zero state")
    return state / norm


def eigenstate(basis: str, eigenvalue: int) -> np.ndarray:
    basis = basis.upper()
    if basis not in PAULI or eigenvalue not in (1, -1):
        raise ValueError("basis must be X/Y/Z and eigenvalue must be +/-1")

    if basis == "Z":
        state = np.array([1, 0], dtype=complex) if eigenvalue == 1 else np.array([0, 1], dtype=complex)
    elif basis == "X":
        state = np.array([1, 1], dtype=complex) if eigenvalue == 1 else np.array([1, -1], dtype=complex)
    else:
        state = np.array([1, 1j], dtype=complex) if eigenvalue == 1 else np.array([1, -1j], dtype=complex)
    return normalize(state)


def projective_measure(state: np.ndarray, basis: str, rng: np.random.Generator) -> int:
    state = normalize(np.asarray(state, dtype=complex))
    plus = eigenstate(basis, 1)
    p_plus = float(abs(np.vdot(plus, state)) ** 2)
    return 1 if rng.random() < p_plus else -1


def teleport_corrected_state(psi: np.ndarray, m1: int, m2: int) -> np.ndarray:
    """Simulate standard teleportation with the convention used in the project notes.

    Before correction Bob has X^m2 Z^m1 |psi>. The verifier applies
    Z^m1 X^m2, recovering |psi> up to a global phase.
    """
    if m1 not in (0, 1) or m2 not in (0, 1):
        raise ValueError("teleportation bits must be 0/1")
    psi = np.asarray(psi, dtype=complex)
    raw = np.linalg.matrix_power(X, m2) @ np.linalg.matrix_power(Z, m1) @ psi
    corrected = np.linalg.matrix_power(Z, m1) @ np.linalg.matrix_power(X, m2) @ raw
    return normalize(corrected)


def state_fidelity(a: np.ndarray, b: np.ndarray) -> float:
    a = normalize(a)
    b = normalize(b)
    return float(abs(np.vdot(a, b)) ** 2)


def apply_depolarizing(state: np.ndarray, epsilon: float, rng: np.random.Generator) -> np.ndarray:
    """Sample a Pauli error channel; epsilon is total non-identity error probability."""
    epsilon = min(max(epsilon, 0.0), 1.0)
    if rng.random() >= epsilon:
        return state
    err = rng.choice([X, Y, Z])
    return normalize(err @ state)


def simulate_signature_block(block_length: int, basis: str, eigenvalue: int, noise_epsilon: float,
                             rng: np.random.Generator, forced_mismatch_rate: float | None = None) -> list[int]:
    """Produce mismatch bits for one signature block.

    A legitimate qubit has mismatch probability epsilon/2 under the project's
    symmetric depolarizing approximation. Attack scenarios may override it.
    """
    if forced_mismatch_rate is None:
        mismatch_rate = noise_epsilon / 2.0
    else:
        mismatch_rate = forced_mismatch_rate
    return rng.binomial(1, mismatch_rate, size=block_length).astype(int).tolist()
