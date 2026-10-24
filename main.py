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

def process_invoices_with_ai(packages, batch_id, policy_revision):
    # Here you can integrate your LLM call (OpenAI, Gemini, etc.) 
    # For each package in the batch, generate the required proposal structure:
    proposals = []
    for pkg in packages:
        package_id = pkg.get("packageId")
        # Example proposal structure required by the assignment:
        proposals.append({
            "packageId": package_id,
            "actionId": str(uuid.uuid4()).replace("-", "")[:12], # At least 12 chars unique ID
            "action": "settle_invoice", # Replace with actual AI decision logic
            "facts": {
                "vendorName": pkg.get("vendorName", "Sample Vendor"),
                "invoiceNumber": pkg.get("invoiceNumber", "INV-001"),
                "amountMinor": pkg.get("amountMinor", 1000),
                "currency": pkg.get("currency", "INR")
            },
            "evidenceRefs": ["exact reference 1", "exact reference 2", "exact reference 3"],
            "rationale": "The invoice is fully verified, valid, reconciled, and within autonomous authority limits."
        })
    return proposals
@app.post("/message:send")
async def message_send(request: Request):
    # Extract the Bearer token to identify the user/principal
    auth_header = request.headers.get("authorization", "")
    principal = auth_header.replace("Bearer ", "").strip()
    
    body = await request.json()
    message = body.get("message", {})
    message_id = message.get("messageId")
    
    if not message_id:
        return JSONResponse(status_code=400, content={"error": "Missing messageId"})

    # 1. Check if this is a replay or conflict using storage.py
    digest = storage.message_hash(message)
    existing = storage.get_existing_message(principal, message_id)
    
    if existing:
        if existing["message_hash"] != digest:
            return JSONResponse(status_code=409, content={"error": "IDEMPOTENCY_CONFLICT"})
        # Replay: return the existing task
        return {"task": existing["task"]}

    # 2. Check if this is a result continuation from the grader or an initial batch request
    parts = message.get("parts", [])
    task_id = message.get("taskId") or str(uuid.uuid4())
    context_id = message.get("contextId", str(uuid.uuid4()))
    
    # Look for invoice claim batch
    batch_data = None
    for part in parts:
        if part.get("mediaType") == CLAIM_BATCH:
            batch_data = part.get("data", {})
            break

    if batch_data:
        # Initial Request: Run AI to generate proposals
        batch_id = batch_data.get("batchId")
        policy_revision = batch_data.get("policyRevision")
        packages = batch_data.get("packages", [])
        
        proposals = process_invoices_with_ai(packages, batch_id, policy_revision)
        
        # Build the Task object in TASK_STATE_INPUT_REQUIRED
        task_obj = {
            "taskId": task_id,
            "contextId": context_id,
            "status": "TASK_STATE_INPUT_REQUIRED",
            "history": [message],
            "artifacts": [{
                "mediaType": PROPOSALS,
                "data": {
                    "batchId": batch_id,
                    "proposals": proposals
                }
            }]
        }
        
        # Atomically save task and register message deduplication
        result = storage.create_task_once(
            principal=principal,
            message_id=message_id,
            digest=digest,
            task_id=task_id,
            context_id=context_id,
            batch_id=batch_id,
            status="TASK_STATE_INPUT_REQUIRED",
            task=task_obj
        )
        
        if result["kind"] == "conflict":
            return JSONResponse(status_code=409, content={"error": "IDEMPOTENCY_CONFLICT"})
            
        return {"task": result["task"]}
        # Look for invoice action results continuation
    results_data = None
    for part in parts:
        if part.get("mediaType") == "application/vnd.ga5.invoice-action-results+json":
            results_data = part.get("data", {})
            break

    if results_data:
        task_id = message.get("taskId")
        existing_task = storage.get_task(task_id, principal)
        
        if not existing_task:
            return JSONResponse(status_code=404, content={"error": "Task not found"})
            
        if existing_task["status"] in ["TASK_STATE_COMPLETED", "TASK_STATE_CANCELED"]:
            return JSONResponse(status_code=409, content={"error": "Task is already terminal"})

        # Find the original proposals from existing task artifacts
        original_proposals = []
        for art in existing_task.get("artifacts", []):
            if art.get("mediaType") == PROPOSALS:
                original_proposals = art.get("data", {}).get("proposals", [])
                break

        # Build executions array containing accepted results only
        executions = []
        results = results_data.get("results", [])
        
        for res in results:
            if res.get("outcome") == "ACCEPTED":
                pkg_id = res.get("packageId")
                action_id = res.get("actionId")
                
                # Match with original proposal
                matching_prop = next((p for p in original_proposals if p.get("packageId") == pkg_id and p.get("actionId") == action_id), None)
                if matching_prop:
                    executions.append({
                        "packageId": pkg_id,
                        "actionId": action_id,
                        "action": res.get("action"),
                        "receiptNonce": res.get("receiptNonce"),
                        "facts": matching_prop.get("facts"),
                        "evidenceRefs": matching_prop.get("evidenceRefs")
                    })

        # Update task history and state to COMPLETED
        existing_task["history"].append(message)
        existing_task["status"] = "TASK_STATE_COMPLETED"
        existing_task["artifacts"].append({
            "mediaType": "application/vnd.ga5.invoice-action-receipts+json",
            "data": {
                "batchId": results_data.get("batchId"),
                "executions": executions
            }
        })

        # Save the completed task
        success = storage.save_task(task_id, principal, existing_task.get("contextId"), results_data.get("batchId"), "TASK_STATE_COMPLETED", existing_task)
        if not success:
            return JSONResponse(status_code=409, content={"error": "Conflict saving completed task"})

        # Record message deduplication
        with storage.connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO message_dedup (principal, message_id, message_hash, task_id) VALUES (?, ?, ?, ?)",
                (principal, message_id, digest, task_id)
            )
            conn.commit()

        return {"task": existing_task}
    
    return JSONResponse(status_code=400, content={"error": "Unsupported message part type"})

@app.get("/tasks")
async def list_tasks(request: Request):
    auth_header = request.headers.get("authorization", "")
    principal = auth_header.replace("Bearer ", "").strip()
    tasks = storage.list_tasks(principal)
    return {"tasks": tasks}

@app.get("/tasks/{task_id}")
async def get_task(task_id: str, request: Request):
    auth_header = request.headers.get("authorization", "")
    principal = auth_header.replace("Bearer ", "").strip()
    task = storage.get_task(task_id, principal)
    if not task:
        return JSONResponse(status_code=404, content={"error": "Task not found"})
    return task

@app.post("/tasks/{task_id}:cancel")
async def cancel_task(task_id: str, request: Request):
    auth_header = request.headers.get("authorization", "")
    principal = auth_header.replace("Bearer ", "").strip()
    
    task = storage.get_task(task_id, principal)
    if not task:
        return JSONResponse(status_code=404, content={"error": "Task not found"})
        
    if task["status"] in ["TASK_STATE_COMPLETED", "TASK_STATE_CANCELED"]:
        return JSONResponse(status_code=409, content={"error": "Task is already terminal"})
        
    task["status"] = "TASK_STATE_CANCELED"
    success = storage.save_task(task_id, principal, task.get("contextId"), task.get("batchId"), "TASK_STATE_CANCELED", task)
    if not success:
        return JSONResponse(status_code=409, content={"error": "Conflict during cancellation"})
        
    return task
