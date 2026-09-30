# PauliGuard — SIH26141 prototype

A deployable, non-ML prototype for the SIH26141 concept: a teleportation-based QDS simulation wrapped in statistical verification, a Bell-witness channel certificate, attack simulation, and a classical authorization/replay ledger.

## Important scientific scope

This is a **prototype simulator**. It does **not** prove information-theoretic security. Replay and authorization are handled by classical protocol state because a valid replay can have the same quantum measurement statistics as the original signature.

The prototype uses the mathematical model described in the project research notes: Pauli-eigenstate public keys, an exposure parameter `n`, a per-block mismatch process, and `F(n)^lambda` as the model forging bound. The code is intended to be validated against the original papers before any publication-level security claim.

## Run locally

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/`.

API documentation: `http://127.0.0.1:8000/docs`

## Main endpoints

- `GET /health`
- `GET /api/formula/{n}`
- `POST /api/run`
- `POST /api/ledger/issue/{key_id}`
- `GET /api/ledger/{key_id}`

## Deploy to Render

Connect the Git repository to Render as a Web Service. The repository contains `render.yaml`; the app uses the standard FastAPI start command.

The demo uses SQLite for its ledger. Render's default filesystem is ephemeral, so SQLite data is not guaranteed to survive a service restart or redeploy. For a persistent SIH deployment, replace SQLite with managed Postgres or attach a persistent disk on a paid service.
