import os
import json
import hashlib
import uuid
from datetime import datetime, timezone

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

import storage

app = FastAPI(title="A2A Invoice Agent")

BASE_URL = os.getenv("A2A_BASE_URL", "").rstrip("/")
AGENT_TOKEN = os.getenv("A2A_BEARER_TOKEN", "")

PROPOSALS = "application/vnd.ga5.invoice-action-proposals+json"
RECEIPTS = "application/vnd.ga5.invoice-action-receipts+json"
CLAIM_BATCH = "application/vnd.ga5.invoice-claim-batch+json"

ACTIONS = {
    "settle_invoice",
    "request_approval",
    "hold_invoice",
    "reject_duplicate",
    "open_exception",
}


@app.get("/")
def home():
    return {
        "message": "A2A Invoice Agent",
        "status": "running"
    }


@app.get("/.well-known/agent-card.json")
def agent_card():
    base_url = BASE_URL or "https://example.invalid"

    return {
        "name": "A2A Invoice Action Agent",
        "description": (
            "Evaluates invoice evidence and proposes safe business actions."
        ),
        "version": "1.0.0",
        "capabilities": {},
        "supportedInterfaces": [
            {
                "url": base_url,
                "protocolBinding": "HTTP+JSON",
                "protocolVersion": "1.0"
            }
        ],
        "defaultInputModes": [CLAIM_BATCH],
        "defaultOutputModes": [PROPOSALS, RECEIPTS],
        "skills": [
            {
                "id": "invoice_action_agent",
                "name": "Invoice Action Agent",
                "description": (
                    "Evaluates invoice packages and proposes actions "
                    "supported by documentary evidence."
                ),
                "tags": ["invoices", "evidence", "approval", "payments"]
            }
        ]
    }


@app.middleware("http")
async def protocol_checks(request: Request, call_next):
    path = request.url.path

    # The Agent Card is public. Other API routes require authentication.
    if path != "/.well-known/agent-card.json" and path != "/":
        if not AGENT_TOKEN:
            return JSONResponse(
                status_code=503,
                content={"error": "Service authentication is not configured"}
            )

        if request.headers.get("authorization") != (
            f"Bearer {AGENT_TOKEN}"
        ):
            return JSONResponse(
                status_code=401,
                content={"error": "Unauthorized"}
            )

        if request.headers.get("a2a-version") != "1.0":
            return JSONResponse(
                status_code=400,
                content={"error": "Unsupported A2A version"}
            )

        if request.method == "POST":
    content_type = request.headers.get("content-type", "").split(";")[0].strip()
    if content_type != "application/a2a+json":
        return JSONResponse(status_code=415, content={"error": "Unsupported media type"})

    return await call_next(request)
