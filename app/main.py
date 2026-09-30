from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .ledger import Ledger
from .qds import eigenstate, projective_measure, state_fidelity, teleport_corrected_state
from .security import (
    bell_witness_simulation,
    clopper_pearson,
    forgery_bound,
    forging_fidelity,
    forging_mismatch,
    hoeffding_far,
    hoeffding_frr,
    simulate_mismatches,
    sprt,
)

BASE_DIR = Path(__file__).resolve().parents[1]
STATIC_DIR = BASE_DIR / "static"
DB_PATH = os.getenv("LEDGER_DB_PATH", str(BASE_DIR / "data" / "ledger.db"))

app = FastAPI(title="PauliGuard", version="0.1.0", description="Non-ML prototype for quantum digital signature threat detection")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
ledger = Ledger(DB_PATH)


class RunRequest(BaseModel):
    scenario: str = Field(default="legitimate")
    n_copies: int = Field(default=0, ge=0, le=20)
    block_length: int = Field(default=100, ge=8, le=5000)
    blocks: int = Field(default=8, ge=1, le=256)
    noise_epsilon: float = Field(default=0.02, ge=0, le=1)
    threshold: float = Field(default=0.20, ge=0, le=1)
    bell_tests: int = Field(default=200, ge=20, le=10000)
    seed: int = Field(default=42, ge=0, le=2**31 - 1)
    verifier_id: str = "verifier-1"
    key_id: str = "demo-key"
    session_id: str = "demo-session-1"


SCENARIOS = {
    "legitimate",
    "forgery_n0",
    "forgery_exposed",
    "impersonation",
    "replay",
    "unauthorized",
    "channel_intercept_resend",
}


def ensure_demo_key(key_id: str):
    ledger.register_key(key_id, "verifier-1")


def run_teleportation_check(rng: np.random.Generator) -> float:
    basis = rng.choice(["X", "Y", "Z"])
    eigenvalue = int(rng.choice([1, -1]))
    psi = eigenstate(basis, eigenvalue)
    m1 = int(rng.integers(0, 2))
    m2 = int(rng.integers(0, 2))
    received = teleport_corrected_state(psi, m1, m2)
    return state_fidelity(psi, received)


def classify_scenario(req: RunRequest, rng: np.random.Generator):
    if req.scenario not in SCENARIOS:
        raise HTTPException(status_code=400, detail=f"Unknown scenario. Use one of: {sorted(SCENARIOS)}")

    ensure_demo_key(req.key_id)
    ledger_row = ledger.get(req.key_id)
    n = req.n_copies if req.scenario != "unauthorized" else max(req.n_copies, ledger_row["copies_issued"])

    channel_fidelity = 1.0
    if req.scenario == "channel_intercept_resend":
        channel_fidelity = 0.5
    elif req.noise_epsilon:
        channel_fidelity = max(0.0, 1.0 - req.noise_epsilon)

    bell = bell_witness_simulation(req.bell_tests, channel_fidelity, rng)
    bell_abort = bell["failure_rate"] > 0.25 if req.scenario == "channel_intercept_resend" else False

    e0 = min(max(req.noise_epsilon / 2.0, 1e-6), 0.499999)
    q1 = max(forging_mismatch(n), e0 + 1e-3)
    if req.scenario == "impersonation":
        q1 = 0.5

    block_results = []
    for block_index in range(req.blocks):
        if req.scenario == "legitimate":
            mismatch_rate = e0
        elif req.scenario == "forgery_n0":
            mismatch_rate = 0.5
        elif req.scenario == "forgery_exposed":
            mismatch_rate = forging_mismatch(n)
        elif req.scenario == "impersonation":
            mismatch_rate = 0.5
        else:
            mismatch_rate = e0
        mismatches = simulate_mismatches(mismatch_rate, req.block_length, rng)
        mismatch_count = sum(mismatches)
        mismatch_rate_emp = mismatch_count / req.block_length
        s = sprt(mismatches, e0=e0, q1=q1)
        threshold_decision = mismatch_rate_emp > req.threshold
        block_results.append({
            "block": block_index + 1,
            "mismatches": mismatch_count,
            "mismatch_rate": mismatch_rate_emp,
            "sprt": s.decision,
            "sprt_samples": s.samples,
            "sprt_llr": s.llr,
            "threshold_attack": threshold_decision,
        })

    if req.scenario == "replay":
        decision = "REPLAY" if not ledger.consume_session(req.session_id, req.key_id) else "ACCEPT"
    elif req.scenario == "unauthorized":
        decision = "UNAUTHORIZED" if not ledger.authorize(req.key_id, "unknown-verifier") else "ACCEPT"
    elif bell_abort:
        decision = "ABORT_CHANNEL_MANIPULATION"
    elif req.scenario in {"forgery_n0", "forgery_exposed"}:
        decision = "REJECT_FORGERY" if any(b["sprt"] == "ATTACK" or b["threshold_attack"] for b in block_results) else "ACCEPT"
    elif req.scenario == "impersonation":
        decision = "REJECT_IMPERSONATION" if any(b["sprt"] == "ATTACK" for b in block_results) else "ACCEPT"
    else:
        decision = "ACCEPT" if all(b["sprt"] != "ATTACK" and not b["threshold_attack"] for b in block_results) else "REJECT_SUSPICIOUS"

    return {
        "scenario": req.scenario,
        "decision": decision,
        "parameters": {
            "n_copies": n,
            "block_length": req.block_length,
            "blocks": req.blocks,
            "noise_epsilon": req.noise_epsilon,
            "threshold": req.threshold,
            "bell_tests": req.bell_tests,
            "e0": e0,
            "q1": q1,
        },
        "security": {
            "forging_fidelity_Fn": forging_fidelity(n),
            "forging_mismatch_q": forging_mismatch(n),
            "model_forgery_bound": forgery_bound(n, req.block_length),
            "hoeffding_far": hoeffding_far(q1, req.threshold, req.block_length) if q1 > req.threshold else None,
            "hoeffding_frr": hoeffding_frr(e0, req.threshold, req.block_length) if req.threshold > e0 else None,
        },
        "teleportation_fidelity": run_teleportation_check(rng),
        "bell_witness": {**bell, "aborted": bell_abort},
        "blocks": block_results,
        "disclaimer": "Prototype simulation only. It does not prove information-theoretic security.",
    }


@app.get("/")
def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "service": "PauliGuard", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/api/formula/{n}")
def formula(n: int):
    if n < 0 or n > 20:
        raise HTTPException(status_code=400, detail="n must be between 0 and 20")
    return {"n": n, "F_n": forging_fidelity(n), "q": forging_mismatch(n), "F_n_lambda_100": forging_bound(n, 100)}


@app.post("/api/run")
def run(req: RunRequest):
    start = perf_counter()
    rng = np.random.default_rng(req.seed)
    result = classify_scenario(req, rng)
    result["runtime_ms"] = round((perf_counter() - start) * 1000, 3)
    return result


@app.post("/api/ledger/issue/{key_id}")
def issue(key_id: str):
    ensure_demo_key(key_id)
    ledger.issue_copy(key_id)
    return ledger.get(key_id)


@app.get("/api/ledger/{key_id}")
def ledger_info(key_id: str):
    ensure_demo_key(key_id)
    return ledger.get(key_id)
