import os
import asyncio
import copy
import re
import json
import hashlib
import uuid
from datetime import datetime, timezone

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

import storage

app = FastAPI(title="A2A Invoice Agent")

# Must match, character for character, the URL you submit on the GA page.
BASE_URL = os.getenv(
    "A2A_BASE_URL",
    "https://tds-ga3-a2a-invoice-agent.onrender.com/",
)

PROPOSALS = "application/vnd.ga5.invoice-action-proposals+json"
RECEIPTS = "application/vnd.ga5.invoice-action-receipts+json"
CLAIM_BATCH = "application/vnd.ga5.invoice-claim-batch+json"
RESULTS = "application/vnd.ga5.invoice-action-results+json"

ACTIONS = {
    "settle_invoice",
    "request_approval",
    "hold_invoice",
    "reject_duplicate",
    "open_exception",
}

BEARER_RE = re.compile(r"^Bearer\s+(\S+)$")


def get_principal(request: Request):
    """Turn the Bearer token into a private user id (a hash)."""
    match = BEARER_RE.match(request.headers.get("authorization", ""))
    token = match.group(1) if match else ""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


@app.get("/")
def home():
    return {"message": "A2A Invoice Agent", "status": "running"}


@app.get("/.well-known/agent-card.json")
def agent_card():
    return {
        "name": "A2A Invoice Action Agent",
        "description": "Evaluates invoice evidence and proposes safe business actions.",
        "version": "1.0.0",
        "capabilities": {},
        "supportedInterfaces": [
            {
                "url": BASE_URL,
                "protocolBinding": "HTTP+JSON",
                "protocolVersion": "1.0",
            }
        ],
        "defaultInputModes": [CLAIM_BATCH],
        "defaultOutputModes": [PROPOSALS, RECEIPTS],
        "skills": [
            {
                "id": "invoice_action_agent",
                "name": "Invoice Action Agent",
                "description": "Evaluates invoice packages and proposes actions supported by documentary evidence.",
                "tags": ["invoices", "evidence", "approval", "payments"],
            }
        ],
    }


@app.middleware("http")
async def protocol_checks(request: Request, call_next):
    path = request.url.path

    # The Agent Card and home page are public.
    if path in ("/", "/.well-known/agent-card.json"):
        return await call_next(request)

    # 1. Any non-empty Bearer token is a valid user.
    if not BEARER_RE.match(request.headers.get("authorization", "")):
        return JSONResponse(status_code=401, content={"error": "Unauthorized"})

    # 2. Version check.
    if request.headers.get("a2a-version") != "1.0":
        return JSONResponse(status_code=400, content={"error": "Unsupported A2A version"})

    # 3. POST bodies must be application/a2a+json.
    if request.method == "POST":
        ctype = request.headers.get("content-type", "").split(";")[0].strip().lower()
        if ctype != "application/a2a+json":
            return JSONResponse(status_code=415, content={"error": "Unsupported media type"})

    response = await call_next(request)
    response.headers["Content-Type"] = "application/a2a+json"
    return response

STATE_INPUT = "TASK_STATE_INPUT_REQUIRED"
STATE_DONE = "TASK_STATE_COMPLETED"
STATE_CANCELED = "TASK_STATE_CANCELED"
RESULTS_PART_OK = ("ACCEPTED", "REJECTED")
NONTERMINAL = {"TASK_STATE_SUBMITTED", "TASK_STATE_WORKING", STATE_INPUT}

_locks = {}


def key_lock(key):
    return _locks.setdefault(key, asyncio.Lock())


def err(status, reason):
    # Generic body: never echoes ids or reveals who owns what.
    return JSONResponse(
        status_code=status,
        content={"error": {"code": status, "reason": reason, "message": reason}},
    )


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def make_artifact(name, media_type, data):
    return {
        "artifactId": str(uuid.uuid4()),
        "name": name,
        "parts": [{"mediaType": media_type, "data": data}],
    }


def find_part(message, media_type):
    for part in message.get("parts") or []:
        if isinstance(part, dict) and part.get("mediaType") == media_type:
            return part.get("data")
    return None


def proposals_of(task):
    for art in task.get("artifacts") or []:
        for part in art.get("parts") or []:
            if part.get("mediaType") == PROPOSALS:
                return part.get("data") or {}
    return {}


REF_RE = re.compile(r"\[[^\[\]]+\]")


def decide_batch(packages, batch_id, policy_revision):
    """TEMPORARY stand-in. Piece 3 replaces this with the real AI step.
    Do NOT spend a GA3 Check while this is still here."""
    proposals = []
    for pkg in packages:
        text = json.dumps(pkg, ensure_ascii=False)
        refs = list(dict.fromkeys(REF_RE.findall(text)))[:3]
        proposals.append({
            "packageId": pkg.get("packageId") or pkg.get("id"),
            "actionId": "act-" + uuid.uuid4().hex,
            "action": "hold_invoice",
            "facts": {"vendorName": "", "invoiceNumber": "",
                      "amountMinor": 0, "currency": ""},
            "evidenceRefs": refs,
            "rationale": "Placeholder rationale used only while testing the protocol plumbing of this agent. " * 2,
        })
    if packages:
        print("FIRST PACKAGE KEYS:", list(packages[0].keys()))
        print("FIRST PACKAGE SAMPLE:", json.dumps(packages[0])[:1500])
    return proposals


@app.post("/message:send")
async def message_send(request: Request):
    principal = get_principal(request)
    try:
        body = await request.json()
    except Exception:
        return err(400, "INVALID_JSON")

    message = body.get("message") if isinstance(body, dict) else None
    if not isinstance(message, dict) or not message.get("messageId"):
        return err(400, "INVALID_MESSAGE")

    message_id = str(message["messageId"])
    digest = storage.message_hash(message)

    existing = storage.get_existing_message(principal, message_id)
    if existing:
        if existing["message_hash"] != digest:
            return err(409, "IDEMPOTENCY_CONFLICT")
        return {"task": existing["task"]}

    results_data = find_part(message, RESULTS)
    if results_data is not None:
        return await handle_results(principal, message, message_id, digest, results_data)

    batch_data = find_part(message, CLAIM_BATCH)
    if not isinstance(batch_data, dict):
        return err(400, "UNSUPPORTED_MESSAGE")

    batch_id = batch_data.get("batchId")
    packages = batch_data.get("packages")
    if not batch_id or not isinstance(packages, list):
        return err(400, "INVALID_BATCH")

    async with key_lock((principal, message_id)):
        existing = storage.get_existing_message(principal, message_id)
        if existing:
            if existing["message_hash"] != digest:
                return err(409, "IDEMPOTENCY_CONFLICT")
            return {"task": existing["task"]}

        proposals = await asyncio.to_thread(
            decide_batch, packages, batch_id, batch_data.get("policyRevision")
        )
        task_id = str(uuid.uuid4())
        context_id = str(message.get("contextId") or uuid.uuid4())
        task = {
            "id": task_id,
            "contextId": context_id,
            "status": {"state": STATE_INPUT, "timestamp": now_iso()},
            "artifacts": [make_artifact(
                "invoice-action-proposals", PROPOSALS,
                {"batchId": batch_id, "proposals": proposals},
            )],
            "history": [message],
        }
        result = storage.create_task_once(
            principal=principal, message_id=message_id, digest=digest,
            task_id=task_id, context_id=context_id, batch_id=str(batch_id),
            status=STATE_INPUT, task=task,
        )
        if result["kind"] == "conflict":
            return err(409, "IDEMPOTENCY_CONFLICT")
        return {"task": result["task"]}


async def handle_results(principal, message, message_id, digest, data):
    task_id = message.get("taskId")
    task = storage.get_task(task_id, principal) if isinstance(task_id, str) else None
    if task is None:
        return err(404, "TASK_NOT_FOUND")
    if task["status"]["state"] != STATE_INPUT:
        return err(409, "TASK_NOT_ACTIVE")
    if message.get("contextId") != task.get("contextId"):
        return err(400, "CONTINUATION_MISMATCH")

    stored = proposals_of(task)
    if not isinstance(data, dict) or data.get("batchId") != stored.get("batchId"):
        return err(400, "CONTINUATION_MISMATCH")

    results = data.get("results")
    if not isinstance(results, list):
        return err(400, "CONTINUATION_MISMATCH")

    by_package = {p["packageId"]: p for p in stored.get("proposals", [])}
    seen = set()
    executions = []
    for res in results:
        if not isinstance(res, dict):
            return err(400, "CONTINUATION_MISMATCH")
        pkg_id = res.get("packageId")
        prop = by_package.get(pkg_id)
        if prop is None or pkg_id in seen:
            return err(400, "CONTINUATION_MISMATCH")
        seen.add(pkg_id)
        if res.get("actionId") != prop["actionId"] or res.get("action") != prop["action"]:
            return err(400, "CONTINUATION_MISMATCH")
        outcome = res.get("outcome")
        if outcome not in RESULTS_PART_OK:
            return err(400, "CONTINUATION_MISMATCH")
        if outcome == "ACCEPTED":
            nonce = res.get("receiptNonce")
            if not isinstance(nonce, str) or not nonce:
                return err(400, "CONTINUATION_MISMATCH")
            executions.append({
                "packageId": pkg_id,
                "actionId": prop["actionId"],
                "action": prop["action"],
                "receiptNonce": nonce,
                "facts": prop["facts"],
                "evidenceRefs": prop["evidenceRefs"],
            })

    new_task = copy.deepcopy(task)
    new_task["status"] = {"state": STATE_DONE, "timestamp": now_iso()}
    new_task["history"].append(message)
    new_task["artifacts"].append(make_artifact(
        "invoice-action-receipts", RECEIPTS,
        {"batchId": stored.get("batchId"), "executions": executions},
    ))

    won = storage.update_task_if_status(
        task_id, principal, STATE_INPUT, STATE_DONE,
        new_task, message_id, digest,
    )
    if not won:
        existing = storage.get_existing_message(principal, message_id)
        if existing and existing["message_hash"] == digest:
            return {"task": existing["task"]}
        return err(409, "TASK_NOT_ACTIVE")
    return {"task": new_task}


@app.get("/tasks")
async def list_tasks(request: Request):
    return {"tasks": storage.list_tasks(get_principal(request))}


@app.get("/tasks/{task_id}")
async def get_task(task_id: str, request: Request):
    task = storage.get_task(task_id, get_principal(request))
    if not task:
        return err(404, "TASK_NOT_FOUND")
    return task

@app.post("/tasks/{task_id}:cancel")
async def cancel_task(task_id: str, request: Request):
    principal = get_principal(request)
    task = storage.get_task(task_id, principal)
    if not task:
        return err(404, "TASK_NOT_FOUND")

    current = task["status"]["state"]
    if current not in NONTERMINAL:
        return err(409, "TASK_NOT_CANCELABLE")

    new_task = copy.deepcopy(task)
    new_task["status"] = {"state": STATE_CANCELED, "timestamp": now_iso()}
    if not storage.update_task_if_status(
        task_id, principal, current, STATE_CANCELED, new_task
    ):
        return err(409, "TASK_NOT_CANCELABLE")
    return new_task
