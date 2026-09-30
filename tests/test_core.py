import numpy as np

from app.qds import eigenstate, state_fidelity, teleport_corrected_state
from app.security import forging_fidelity


def test_teleportation_noiseless_fidelity():
    for basis in ("X", "Y", "Z"):
        for ev in (1, -1):
            psi = eigenstate(basis, ev)
            for m1, m2 in ((0,0),(0,1),(1,0),(1,1)):
                received = teleport_corrected_state(psi, m1, m2)
                assert abs(state_fidelity(psi, received) - 1.0) < 1e-12


def test_forging_fidelity_known_values():
    assert abs(forging_fidelity(0) - 0.5) < 1e-12
    assert abs(forging_fidelity(1) - 0.75) < 1e-12
