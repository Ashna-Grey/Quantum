from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.special import comb
from scipy.stats import beta


PAULI_EIGENVALUES = (1, -1)


def forging_fidelity(n: int) -> float:
    """Per-qubit optimal identification success term F(n) from the project source.

    This is the model used by the prototype; it is not itself a proof of security.
    """
    if n < 0:
        raise ValueError("n must be >= 0")
    alphas = []
    for j in range(4):
        total = 0.0
        for k in range(n + 1):
            if k % 4 == j:
                total += float(comb(n, k, exact=False))
        alphas.append(total)
    s = sum(math.sqrt(alphas[j] * alphas[(j + 1) % 4]) for j in range(4))
    return 0.5 + (2.0 ** (-(n + 1))) * s


def forging_mismatch(n: int) -> float:
    return 1.0 - forging_fidelity(n)


def forgery_bound(n: int, block_length: int) -> float:
    """Model probability F(n)^lambda for a block of lambda qubits."""
    return forging_fidelity(n) ** block_length


def hoeffding_far(q: float, threshold: float, block_length: int) -> float:
    if q <= threshold:
        return 1.0
    return math.exp(-2.0 * block_length * (q - threshold) ** 2)


def hoeffding_frr(e0: float, threshold: float, block_length: int) -> float:
    if threshold <= e0:
        return 1.0
    return math.exp(-2.0 * block_length * (threshold - e0) ** 2)


def clopper_pearson(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    if not 0 <= successes <= trials:
        raise ValueError("successes must be within [0, trials]")
    alpha = 1.0 - confidence
    if successes == 0:
        low = 0.0
    else:
        low = float(beta.ppf(alpha / 2, successes, trials - successes + 1))
    if successes == trials:
        high = 1.0
    else:
        high = float(beta.ppf(1 - alpha / 2, successes + 1, trials - successes))
    return low, high


@dataclass
class SPRTResult:
    decision: str
    samples: int
    llr: float
    lower_boundary: float
    upper_boundary: float


def sprt(mismatches: Iterable[int], e0: float, q1: float, alpha: float = 1e-3, beta_risk: float = 1e-3) -> SPRTResult:
    """Sequential probability ratio test for Bernoulli mismatch observations."""
    if not 0 < e0 < 1 or not 0 < q1 < 1:
        raise ValueError("e0 and q1 must be in (0, 1)")
    if q1 <= e0:
        raise ValueError("q1 must be greater than e0 for this detector")

    lower = math.log(beta_risk / (1 - alpha))
    upper = math.log((1 - beta_risk) / alpha)
    llr = 0.0
    count = 0

    for x in mismatches:
        x = int(x)
        if x not in (0, 1):
            raise ValueError("mismatches must contain 0/1")
        count += 1
        if x:
            llr += math.log(q1 / e0)
        else:
            llr += math.log((1 - q1) / (1 - e0))
        if llr >= upper:
            return SPRTResult("ATTACK", count, llr, lower, upper)
        if llr <= lower:
            return SPRTResult("LEGITIMATE", count, llr, lower, upper)

    return SPRTResult("INCONCLUSIVE", count, llr, lower, upper)


def simulate_mismatches(rate: float, block_length: int, rng: np.random.Generator) -> list[int]:
    rate = min(max(rate, 0.0), 1.0)
    return rng.binomial(1, rate, size=block_length).astype(int).tolist()


def bell_pass_probability(fidelity: float) -> float:
    """Pass probability for random X/Y/Z witness tests on a Phi+ Bell state."""
    fidelity = min(max(fidelity, 0.0), 1.0)
    return 1.0 / 3.0 + 2.0 * fidelity / 3.0


def bell_witness_simulation(num_tests: int, fidelity: float, rng: np.random.Generator) -> dict:
    p_pass = bell_pass_probability(fidelity)
    passes = int(rng.binomial(num_tests, p_pass))
    failures = num_tests - passes
    return {
        "tests": num_tests,
        "passes": passes,
        "failures": failures,
        "pass_rate": passes / num_tests if num_tests else 0.0,
        "failure_rate": failures / num_tests if num_tests else 0.0,
        "expected_pass_probability": p_pass,
    }
