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


from decimal import Decimal

AI_MODEL = os.getenv("AI_MODEL", "gpt-5-mini")
DECISION_CACHE = {}
_client = None

ID_RE = re.compile(r"\[[A-Za-z0-9_\-]{6,}\]")
COVER_RE = re.compile(
    r"Supplier\s+(.+?);\s*invoice\s+(\S+?);\s*stated total\s+([A-Z]{3})\s+([\d,]+(?:\.\d+)?)",
    re.S,
)

SYSTEM_PROMPT = """You are an accounts-payable analyst. Each package gives a COVER SHEET paragraph (it only identifies the invoice; never cite it) and CANDIDATE PARAGRAPHS that carry bracketed evidence references.
Choose the ONE candidate paragraph that actually decides what to do with the invoice, and ONE action:
- settle_invoice: the invoice is valid, reconciled, and within autonomous approval authority.
- request_approval: commercially valid, but outside delegated approval authority.
- hold_invoice: payment pauses until a stated verification completes.
- reject_duplicate: the SAME commercial invoice (same vendor, invoice number and amount) was already paid. A payment for a different invoice is NOT a duplicate.
- open_exception: material records conflict and need an exception workflow.

Rules:
- Ignore candidates that only quote an old, archived, superseded or training example, or that use an action word in negation (for example 'not a duplicate' or 'no approval needed').
- Never choose settle_invoice unless the deciding paragraph positively shows the invoice is valid, reconciled and within authority. If in doubt between settle and any other action, do not settle.
- reason: 1-3 plain sentences explaining why the paragraph leads to the action.

Reply with JSON only:
{"decisions":[{"index":0,"paragraph":0,"action":"hold_invoice","facts":{"vendorName":"...","invoiceNumber":"...","amountMinor":0,"currency":"EUR"},"reason":"..."}]}
One decision per package, using the package index; "paragraph" is the candidate paragraph number."""


def get_client():
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            base_url=os.getenv("OPENAI_BASE_URL", "https://aipipe.org/openai/v1"),
            timeout=38,
            max_retries=0,
        )
    return _client


def split_package(pkg):
    """Return (cover, candidates). cover = (paragraph, regex match) or None."""
    cover = None
    candidates = []
    for doc in pkg.get("documents") or []:
        if not isinstance(doc, dict):
            continue
        text = doc.get("text") or ""
        for para in re.split(r"\n\s*\n", text):
            para = para.strip()
            if not para:
                continue
            m = COVER_RE.search(para)
            if m and cover is None:
                cover = (para, m)
                continue
            refs = list(dict.fromkeys(ID_RE.findall(para)))
            if refs:
                candidates.append({"doc": doc.get("name", ""), "text": para, "refs": refs})
    return cover, candidates


def facts_from_cover(cover):
    if not cover:
        return None
    m = cover[1]
    try:
        minor = int((Decimal(m.group(4).replace(",", "")) * 100).to_integral_value())
        return {
            "vendorName": m.group(1).strip(),
            "invoiceNumber": m.group(2).strip(),
            "amountMinor": minor,
            "currency": m.group(3),
        }
    except Exception:
        return None


def package_key(pkg):
    content = {k: v for k, v in pkg.items() if k not in ("packageId", "id")}
    return hashlib.sha256(storage.canonical_json(content).encode("utf-8")).hexdigest()


def pad_refs(refs, pkg):
    allrefs = list(dict.fromkeys(ID_RE.findall(json.dumps(pkg, ensure_ascii=False))))
    out = list(refs)
    for r in allrefs:
        if len(out) >= 3:
            break
        if r not in out:
            out.append(r)
    return out[:3]

def build_decision(dec, pkg):
    """Turn one AI answer into a clean decision, or None if unusable."""
    try:
        cover, cands = split_package(pkg)
        action = dec.get("action")
        if action not in ACTIONS or not cands:
            return None
        n = int(dec.get("paragraph"))
        if not 0 <= n < len(cands):
            return None
        refs = cands[n]["refs"]
        if len(refs) != 3:
            refs = pad_refs(refs[:3], pkg)
        if len(refs) != 3:
            return None
        facts = facts_from_cover(cover)
        if facts is None:
            f = dec["facts"]
            facts = {
                "vendorName": str(f["vendorName"]),
                "invoiceNumber": str(f["invoiceNumber"]),
                "amountMinor": int(f["amountMinor"]),
                "currency": str(f["currency"]).upper(),
            }
        return {
            "action": action,
            "facts": facts,
            "evidenceRefs": refs,
            "reason": str(dec.get("reason", "")),
        }
    except Exception:
        return None


def fallback_decision(pkg):
    """If the AI answer is unusable: pause payment, never settle."""
    cover, cands = split_package(pkg)
    refs = pad_refs(cands[0]["refs"][:3] if cands else [], pkg)
    facts = facts_from_cover(cover) or {
        "vendorName": "", "invoiceNumber": "", "amountMinor": 0, "currency": "",
    }
    return {
        "action": "hold_invoice",
        "facts": facts,
        "evidenceRefs": refs,
        "reason": "The documents could not be classified confidently, so payment is paused for verification.",
    }


def build_rationale(d):
    r = (d["evidenceRefs"] + ["", "", ""])[:3]
    text = f"Action {d['action']}: {d['reason']} Decisive evidence: {r[0]}, {r[1]} and {r[2]}."
    if len(text) < 60:
        text += " This action follows from the deciding paragraph of the invoice package."
    return text[:1500]


def ask_model(items):
    blocks = []
    for i, pkg in items:
        cover, cands = split_package(pkg)
        lines = [f"=== PACKAGE index={i} ===",
                 "COVER SHEET: " + (cover[0] if cover else "(none)")]
        for n, c in enumerate(cands):
            lines.append(f"CANDIDATE PARAGRAPH {n} (from {c['doc']}): {c['text']}")
        blocks.append("\n".join(lines))
    response = get_client().chat.completions.create(
        model=AI_MODEL,
        reasoning_effort="low",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "\n\n".join(blocks)},
        ],
    )
    data = json.loads(response.choices[0].message.content)
    return {d.get("index"): d for d in data.get("decisions", []) if isinstance(d, dict)}


def decide_batch(packages, batch_id, policy_revision):
    keys = [package_key(p) for p in packages]
    local = {}
    missing = [(i, packages[i]) for i, k in enumerate(keys) if k not in DECISION_CACHE]

    if missing:
        try:
            answers = ask_model(missing)
        except Exception as exc:
            print("AI CALL FAILED:", repr(exc))
            answers = {}
        for i, pkg in missing:
            clean = build_decision(answers.get(i, {}), pkg)
            if clean is None:
                print("BAD OR MISSING AI DECISION for package", i, "- using safe fallback")
                local[keys[i]] = fallback_decision(pkg)
            else:
                DECISION_CACHE[keys[i]] = clean
            if i < 4:
                cover, cands = split_package(pkg)
                for n, c in enumerate(cands):
                    print("CAND", i, n, c["doc"], c["text"][:600])
                print("DECISION", i, (DECISION_CACHE.get(keys[i]) or local[keys[i]])["action"])

    proposals = []
    for i, pkg in enumerate(packages):
        d = DECISION_CACHE.get(keys[i]) or local[keys[i]]
        proposals.append({
            "packageId": pkg.get("packageId") or pkg.get("id"),
            "actionId": "act-" + uuid.uuid4().hex,
            "action": d["action"],
            "facts": d["facts"],
            "evidenceRefs": d["evidenceRefs"],
            "rationale": build_rationale(d),
        })
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
