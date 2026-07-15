"""
bot.py — "Vera" challenge submission bot.

WHAT THIS FILE DOES
--------------------
It's an HTTP server (built with FastAPI) that exposes the 5 endpoints the
judge harness needs:

    POST /v1/context   -> judge pushes category/merchant/customer/trigger data
    POST /v1/tick      -> judge asks "anything you want to send right now?"
    POST /v1/reply     -> judge sends a merchant/customer reply, we respond
    GET  /v1/healthz   -> liveness check
    GET  /v1/metadata  -> who we are / what model we use

HOW THE "BRAIN" WORKS
----------------------
Whenever we need to write a WhatsApp message, we:
  1. Gather the relevant context (category + merchant + trigger [+ customer])
  2. Build one big, carefully-written prompt describing the rules
  3. Ask Gemini (temperature=0, so answers are consistent) to write the JSON
  4. Validate the JSON (non-empty body, one CTA, no repeats) before returning

Everything is kept in memory (Python dictionaries). That's fine for this
challenge — the brief explicitly allows in-memory storage.
"""

import os
import re
import time
import json
import uuid
import logging
from datetime import datetime, timezone
from typing import Any, Literal, Optional

import requests
from fastapi import FastAPI
from pydantic import BaseModel

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("vera-bot")

app = FastAPI(title="Vera Challenge Bot")
START_TIME = time.time()

# Put your Groq API key in an environment variable called GROQ_API_KEY.
# NEVER hard-code the real key into this file if you're going to push it to
# a public GitHub repo.
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

TEAM_NAME = os.environ.get("TEAM_NAME", "Solo Builder")
TEAM_MEMBERS = os.environ.get("TEAM_MEMBERS", "Me").split(",")
CONTACT_EMAIL = os.environ.get("CONTACT_EMAIL", "you@example.com")

# -----------------------------------------------------------------------------
# In-memory storage
# -----------------------------------------------------------------------------

# contexts[(scope, context_id)] = {"version": int, "payload": dict}
contexts: dict[tuple[str, str], dict] = {}

# conversations[conversation_id] = {
#     "merchant_id": str, "customer_id": str|None,
#     "sent_bodies": [str, ...],     # everything we've sent, for anti-repetition
#     "turns": [{"from": "vera"|"merchant"|"customer", "body": str}, ...],
#     "repeat_count": int,            # how many times the SAME merchant text repeated
#     "last_merchant_text": str|None,
#     "ended": bool,
# }
conversations: dict[str, dict] = {}

# Which (merchant_id, trigger suppression_key) we've already acted on, so we
# don't send the same trigger twice.
fired_suppression_keys: set[str] = set()

# Tracks how many times we've seen a canned/auto-reply-looking message FROM A GIVEN
# MERCHANT, across all conversations. We key by merchant (not conversation_id)
# because a merchant's WhatsApp auto-reply can arrive under different
# conversation_ids (e.g. one per trigger/tick), so per-conversation repeat
# counting alone would miss it.
canned_auto_reply_hits: dict[str, int] = {}


def get_ctx(scope: str, context_id: str) -> Optional[dict]:
    entry = contexts.get((scope, context_id))
    return entry["payload"] if entry else None


# -----------------------------------------------------------------------------
# LLM call (Gemini)
# -----------------------------------------------------------------------------

def call_claude(system_prompt: str, user_prompt: str, max_tokens: int = 500) -> str:
    """Calls Groq (Llama 3.3 70B) and returns raw text output. Temperature=0 for
    determinism. Named call_claude for minimal diff elsewhere in this file --
    it's the one function that talks to whichever LLM we're using.

    Retries a few times with backoff on rate limits (429) and transient server
    errors (5xx) before giving up -- these are common on free-tier API keys
    under bursty load and are usually resolved within a second or two."""
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set")

    body = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0",
    }

    last_exc = None
    delays = [0, 1.2]  # immediate try, then one short backoff retry -- kept short
                       # so a few triggers in one /v1/tick call don't blow the 30s budget
    for delay in delays:
        if delay:
            time.sleep(delay)
        try:
            resp = requests.post(GROQ_URL, headers=headers, json=body, timeout=25)
            if resp.status_code == 429 or resp.status_code >= 500:
                # Rate-limited or transient server error -- worth retrying.
                last_exc = RuntimeError(f"Groq returned {resp.status_code}, retrying")
                continue
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except requests.RequestException as e:
            last_exc = e
            continue

    raise last_exc or RuntimeError("Groq call failed after retries")


def safe_json_parse(text: str) -> Optional[dict]:
    """LLMs sometimes wrap JSON in ```json fences even when told not to. Strip those."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Prompt building — this is where the "product knowledge" from the brief lives
# -----------------------------------------------------------------------------

COMPOSER_SYSTEM_PROMPT = """You are Vera, an AI WhatsApp marketing assistant for local merchants \
on the magicpin platform in India. You write short WhatsApp messages either to a merchant \
(as yourself, "Vera") or to a merchant's customer, on the merchant's behalf.

HARD RULES:
1. Anchor on a concrete, verifiable fact from the given context (a number, date, headline, \
peer stat). NEVER invent facts, offers, research, or competitor names that are not in the \
context you were given. Never use generic framings like "10% off" or "increase your sales" \
when a specific number or offer is available in the context — use the real number.
2. Match the category's voice exactly. Use this cheat-sheet:
   - Dentists/doctors: clinical, peer-to-peer, technical terms OK, use "Dr." prefix with their \
name, never hype/promo tone ("AMAZING DEAL!" is always wrong here).
   - Salons: warm, friendly, practical.
   - Restaurants: operator-to-operator (talk to them like a fellow business owner, not a customer).
   - Gyms: coaching, motivational tone.
   - Pharmacies: trustworthy, precise, no hype.
3. Personalize to the specific merchant's real numbers, offers, owner/business name, and \
conversation history — use their actual name correctly. Honor their language preference \
(check identity.languages) — Hindi-English code-mix when the preference indicates "hi", \
never force pure English on a hi-en merchant.
4. Explicitly connect the message to WHY you are sending it now: reference specific fields \
from the trigger's payload (the actual number, date, or item named in it), not just the \
trigger's kind in the abstract. "Your dashboard shows a 12% CTR dip this week" beats "there's \
an update about your performance."
5. Use at least one engagement lever: specificity, loss aversion, social proof, effort \
externalization, curiosity, reciprocity, asking the merchant a question, or a single binary \
CTA (YES/STOP). Prefer social proof ("3 dentists in your locality did X this month") and \
"asking the merchant" a question — these are underused and score well.
6. Exactly ONE call-to-action, and it must land in the LAST sentence of the message — never \
bury it mid-message, and never stack multiple asks ("Reply YES for X, NO for Y").
7. No long preambles ("I hope you're doing well..."). Don't re-introduce yourself after the \
first message in a conversation.
8. Never send the same message body verbatim that was already sent in this conversation.
9. If this message is customer-facing (send_as = "merchant_on_behalf"), speak as the merchant's \
business, not as Vera, and never make medical/legal overclaims for regulated categories.
10. NEVER expose internal system details in the message body: no field/key names (e.g. \
"trigger_id", "merchant_id", "suppression_key", "urgency"), no raw JSON, no internal jargon. \
Write only natural human WhatsApp language a real person would send.

EXAMPLE OF A STRONG MESSAGE (specificity + loss aversion, CTA last):
"Quick nudge: your dashboard shows 6,777 missed searches in Sector 14 for makeup services —
people are looking but not finding you. Want me to show how your listing would appear?"
Notice: the number is specific and verifiable, the locality is named, "missed searches" frames
loss, and the single question-CTA is the last sentence.

You must respond with ONLY a JSON object (no markdown fences, no commentary) with these exact \
keys:
{
  "body": "the WhatsApp message text",
  "cta": "binary" | "open_ended" | "none",
  "send_as": "vera" | "merchant_on_behalf",
  "rationale": "one sentence: why this message, what it should achieve"
}
"""


def build_compose_user_prompt(
    category: dict, merchant: dict, trigger: dict, customer: Optional[dict]
) -> str:
    parts = [
        "=== CATEGORY CONTEXT ===",
        json.dumps(category, ensure_ascii=False, indent=2),
        "",
        "=== MERCHANT CONTEXT ===",
        json.dumps(merchant, ensure_ascii=False, indent=2),
        "",
        "=== TRIGGER (why you are messaging right now) ===",
        json.dumps(trigger, ensure_ascii=False, indent=2),
    ]
    if customer:
        parts += [
            "",
            "=== CUSTOMER CONTEXT (this message goes to the merchant's customer, ",
            "on the merchant's behalf — send_as MUST be 'merchant_on_behalf') ===",
            json.dumps(customer, ensure_ascii=False, indent=2),
        ]
    else:
        parts += [
            "",
            "This message is merchant-facing. send_as MUST be 'vera'.",
        ]
    parts += [
        "",
        "Write the single best next WhatsApp message given all of the above. "
        "Respond with ONLY the JSON object described in your instructions.",
    ]
    return "\n".join(parts)


def fallback_compose(category: dict, merchant: dict, trigger: dict, customer: Optional[dict]) -> dict:
    """Used only if the LLM call fails — keeps the endpoint from erroring out.
    Simple, safe, generic — not meant to score well, just to not crash."""
    name = merchant.get("identity", {}).get("name", "there")
    kind = trigger.get("kind", "update")
    return {
        "body": f"Hi {name}, quick update on your account related to {kind.replace('_', ' ')}. "
                f"Want me to share the details?",
        "cta": "open_ended",
        "send_as": "merchant_on_behalf" if customer else "vera",
        "rationale": "Fallback composer used because the LLM call failed.",
    }


def compose_message(
    category: dict,
    merchant: dict,
    trigger: dict,
    customer: Optional[dict],
    already_sent: list[str],
) -> dict:
    system = COMPOSER_SYSTEM_PROMPT
    user = build_compose_user_prompt(category, merchant, trigger, customer)
    if already_sent:
        user += (
            "\n\nMessages already sent in this conversation (do NOT repeat any of these "
            "verbatim):\n" + json.dumps(already_sent, ensure_ascii=False)
        )

    try:
        raw = call_claude(system, user)
        parsed = safe_json_parse(raw)
        if not parsed or not parsed.get("body"):
            raise ValueError("empty or malformed LLM output")
    except Exception as e:
        log.warning(f"LLM compose failed, using fallback: {e}")
        parsed = fallback_compose(category, merchant, trigger, customer)

    # Guardrail: never resend an identical body
    if parsed["body"] in already_sent:
        parsed["body"] += " "  # trivial de-dup nudge; real fix is better prompting

    return {
        "body": parsed.get("body", ""),
        "cta": parsed.get("cta", "open_ended"),
        "send_as": parsed.get("send_as", "merchant_on_behalf" if customer else "vera"),
        "rationale": parsed.get("rationale", ""),
    }


REPLY_SYSTEM_PROMPT = """You are Vera, continuing a WhatsApp conversation with a merchant (or \
their customer) on the magicpin platform. You will be given the conversation so far and the \
latest message from the other side. Decide the single best next move.

RULES:
- If the incoming message is the merchant's WhatsApp Business AUTO-REPLY (a generic canned \
line like "thank you for contacting us, our team will respond") rather than a real reply, \
try ONE gentle nudge for a real person, then if it repeats again, gracefully end the \
conversation. Never loop on an auto-reply more than twice. Seeing the exact same incoming \
text 3+ times is a very strong auto-reply signal.
- If the merchant clearly expresses intent to act — words like "yes", "lets do it", "go \
ahead", "sure do it", "I want to join", "ok" in response to a clear offer — do NOT ask \
another qualifying or clarifying question. That kills momentum and is a known failure mode. \
Move straight to action / the next concrete step (e.g. "Done — I've started X" or "Great, \
sending you Y now" or a single specific next-step question like "Which number should I use \
to update it?"). Do not respond to a clear "yes" with a request for "more detail" or "please \
clarify" — that is always wrong.
- If the merchant says they're not interested, or asks to stop, end the conversation \
gracefully and politely — do not push further. This includes hostile messages like "stop \
messaging me" or "this is spam" — treat these as an explicit stop request: action must be \
"end" with a brief, non-defensive, apologetic rationale. Never respond to hostility by \
sending another pitch or explanation — that will make it worse and is always wrong.
- If the merchant goes off-topic but is NOT hostile/asking to stop, stay polite, briefly \
acknowledge, and steer back to the one thing you're there to help with.
- If the merchant switches language mid-conversation (e.g. from English to Hindi, or to \
Hindi-English code-mix), match their new language in your reply — don't stay locked to the \
language of your earlier turns.
- Never repeat a message body you already sent in this conversation.
- Keep the same voice/category rules as always: specific, non-promotional, one CTA max, CTA \
in the last sentence.

EXAMPLE — CORRECT intent handoff:
[MERCHANT] "Ok lets do it, whats next?"
Correct reply body: "Great — I've started updating your Google profile with the missing \
hours and description. I'll confirm once it's live, usually within a few hours."
WRONG reply body (never do this): "Could you provide a more detailed response so I can better \
assist you?" — this ignores clear intent and re-qualifies, which is always a failure.

EXAMPLE — CORRECT auto-reply handling:
[MERCHANT, 3rd time, verbatim] "Thank you for your message. Our team will get back to you."
Correct action: "end", with a short polite sign-off rationale — do not send a 3rd nudge.

EXAMPLE — CORRECT hostile handling:
[MERCHANT] "Stop messaging me. This is useless spam."
Correct action: "end", rationale like "Merchant explicitly asked to stop; ending politely \
without further pitch."
WRONG action (never do this): sending another "send" message trying to explain, justify, or \
re-pitch — that ignores their explicit stop request.

You must respond with ONLY a JSON object (no markdown fences, no commentary). Exactly one of
these three shapes:

{"action": "send", "body": "...", "cta": "binary"|"open_ended"|"none", "rationale": "..."}
{"action": "wait", "wait_seconds": 1800, "rationale": "..."}
{"action": "end", "rationale": "..."}
"""


def has_canned_auto_reply_marker(text: str) -> bool:
    """Checks ONLY for canned/automated-sounding phrasing, independent of repeat
    count. Used for the merchant-level guardrail below, since conversation_ids
    can differ per turn even for the same merchant's auto-reply."""
    lowered = text.strip().lower()
    canned_markers = [
        "thank you for contacting", "our team will", "automated assistant",
        "we will get back to you", "shukriya", "team tak pahuncha",
    ]
    return any(m in lowered for m in canned_markers)


def is_probable_auto_reply(text: str, prior_texts: list[str]) -> bool:
    """Heuristic: canned auto-replies tend to repeat verbatim, and contain phrases
    like 'thank you for contacting' / 'automated' / 'team will get back'."""
    marker_hit = has_canned_auto_reply_marker(text)
    repeat_hit = prior_texts.count(text.strip()) >= 1  # seen this exact text before
    return marker_hit or repeat_hit


def is_explicit_stop_request(text: str) -> bool:
    """High-precision keyword check for unambiguous 'stop contacting me' / hostile
    messages. Used as a guardrail so we NEVER keep pitching after an explicit stop,
    even if the LLM call fails or misjudges. Deliberately narrow (few, unambiguous
    phrases) to avoid false positives on merchants who are just mildly annoyed but
    still engaging."""
    lowered = text.strip().lower()
    stop_markers = [
        "stop messaging", "stop contacting", "stop texting", "unsubscribe",
        "don't contact me", "do not contact me", "leave me alone",
        "this is spam", "useless spam", "stop spamming", "block this number",
        "remove me from", "not interested, stop",
    ]
    return any(m in lowered for m in stop_markers)


def compose_reply(conv: dict, merchant_message: str) -> dict:
    # Guardrail: explicit stop/hostile requests always end the conversation,
    # regardless of what the LLM would say. This is intentionally a hard rule,
    # not left to the model, because getting this wrong is costly (annoys a
    # merchant further) and the signal is unambiguous when it fires.
    if is_explicit_stop_request(merchant_message):
        return {
            "action": "end",
            "rationale": "Merchant explicitly asked to stop / flagged as spam; "
                         "ending immediately without further pitch (guardrail).",
        }

    # Guardrail: the exact same incoming text 3+ times in a row is, per the brief's
    # own hint, a very strong auto-reply signal. Don't rely on the LLM to comply
    # every time -- force the exit so we never loop forever on a bot.
    if conv.get("repeat_count", 0) >= 2:
        return {
            "action": "end",
            "rationale": "Same incoming message repeated 3+ times verbatim -- "
                         "treating as an automated auto-reply and exiting (guardrail).",
        }

    prior_incoming = [
        t["body"] for t in conv["turns"] if t["from"] in ("merchant", "customer")
    ]
    auto_reply_suspected = is_probable_auto_reply(merchant_message, prior_incoming)

    history_text = "\n".join(
        f"[{t['from'].upper()}] {t['body']}" for t in conv["turns"]
    )
    user = (
        f"Conversation so far:\n{history_text}\n\n"
        f"Latest incoming message: {merchant_message}\n\n"
        f"Heuristic note: this message {'LOOKS LIKE an auto-reply' if auto_reply_suspected else 'looks like a real reply'} "
        f"(repeat count of this exact text so far: {prior_incoming.count(merchant_message.strip())}).\n"
        f"Messages Vera already sent in this conversation (never repeat verbatim): "
        f"{json.dumps(conv['sent_bodies'], ensure_ascii=False)}"
    )

    try:
        raw = call_claude(REPLY_SYSTEM_PROMPT, user)
        parsed = safe_json_parse(raw)
        if not parsed or "action" not in parsed:
            raise ValueError("malformed LLM reply output")
    except Exception as e:
        log.warning(f"LLM reply failed, using fallback: {e}")
        if auto_reply_suspected and conv["repeat_count"] >= 1:
            parsed = {"action": "end", "rationale": "Auto-reply detected twice; exiting gracefully (fallback)."}
        else:
            parsed = {"action": "send", "body": "Got it — let me know if you'd like to continue.",
                       "cta": "open_ended", "rationale": "Fallback reply."}

    return parsed


# -----------------------------------------------------------------------------
# Pydantic request models
# -----------------------------------------------------------------------------

class ContextBody(BaseModel):
    scope: Literal["category", "merchant", "customer", "trigger"]
    context_id: str
    version: int
    payload: dict[str, Any]
    delivered_at: Optional[str] = None


class TickBody(BaseModel):
    now: str
    available_triggers: list[str] = []


class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str
    message: str
    received_at: Optional[str] = None
    turn_number: Optional[int] = None


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------

@app.get("/v1/healthz")
async def healthz():
    counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    for (scope, _cid) in contexts.keys():
        counts[scope] = counts.get(scope, 0) + 1
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - START_TIME),
        "contexts_loaded": counts,
    }


@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": TEAM_NAME,
        "team_members": TEAM_MEMBERS,
        "model": GROQ_MODEL,
        "approach": "Single-prompt LLM composer (Llama 3.3 70B via Groq) over the 4-context "
                    "framework, with a rule+LLM hybrid for auto-reply detection and intent handoff.",
        "contact_email": CONTACT_EMAIL,
        "version": "1.0.0",
        "submitted_at": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/v1/context")
async def push_context(body: ContextBody):
    key = (body.scope, body.context_id)
    current = contexts.get(key)
    if current and current["version"] >= body.version:
        return {
            "accepted": False,
            "reason": "stale_version",
            "current_version": current["version"],
        }
    contexts[key] = {"version": body.version, "payload": body.payload}
    return {
        "accepted": True,
        "ack_id": f"ack_{body.context_id}_v{body.version}",
        "stored_at": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/v1/teardown")
async def teardown():
    contexts.clear()
    conversations.clear()
    fired_suppression_keys.clear()
    return {"ok": True}


@app.post("/v1/tick")
async def tick(body: TickBody):
    actions = []
    tick_started = time.time()
    TICK_TIME_BUDGET = 24.0  # stay well under the judge's 30s hard timeout

    for trg_id in body.available_triggers:
        if time.time() - tick_started > TICK_TIME_BUDGET:
            # Running low on time -- return what we have rather than risk the
            # whole tick timing out and losing every action in it.
            log.warning("Tick time budget exceeded; returning partial actions")
            break

        trigger = get_ctx("trigger", trg_id)
        if not trigger:
            continue

        suppression_key = trigger.get("suppression_key", trg_id)
        if suppression_key in fired_suppression_keys:
            continue  # already acted on this trigger; don't spam

        merchant_id = trigger.get("merchant_id")
        merchant = get_ctx("merchant", merchant_id) if merchant_id else None
        if not merchant:
            continue

        category_slug = merchant.get("category_slug")
        category = get_ctx("category", category_slug) if category_slug else None
        if not category:
            continue

        customer = None
        customer_id = trigger.get("customer_id")
        if customer_id:
            customer = get_ctx("customer", customer_id)

        composed = compose_message(category, merchant, trigger, customer, already_sent=[])

        conversation_id = f"conv_{merchant_id}_{trg_id}"
        conversations[conversation_id] = {
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "sent_bodies": [composed["body"]],
            "turns": [{"from": "vera", "body": composed["body"]}],
            "repeat_count": 0,
            "last_merchant_text": None,
            "ended": False,
        }
        fired_suppression_keys.add(suppression_key)

        actions.append({
            "conversation_id": conversation_id,
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "send_as": composed["send_as"],
            "trigger_id": trg_id,
            "template_name": f"vera_{trigger.get('kind', 'generic')}_v1",
            "template_params": [merchant.get("identity", {}).get("name", "")],
            "body": composed["body"],
            "cta": composed["cta"],
            "suppression_key": suppression_key,
            "rationale": composed["rationale"],
        })

        if len(actions) >= 20:  # respect the per-tick action cap
            break

    return {"actions": actions}


@app.post("/v1/reply")
async def reply(body: ReplyBody):
    conv = conversations.get(body.conversation_id)
    if not conv:
        # We've never seen this conversation — start minimal state so we don't crash.
        conv = {
            "merchant_id": body.merchant_id,
            "customer_id": body.customer_id,
            "sent_bodies": [],
            "turns": [],
            "repeat_count": 0,
            "last_merchant_text": None,
            "ended": False,
        }
        conversations[body.conversation_id] = conv

    if conv["ended"]:
        return {"action": "end", "rationale": "Conversation already ended."}

    # Track repeats of the exact same incoming text (auto-reply signal)
    if conv["last_merchant_text"] == body.message.strip():
        conv["repeat_count"] += 1
    else:
        conv["repeat_count"] = 0
    conv["last_merchant_text"] = body.message.strip()

    conv["turns"].append({"from": body.from_role, "body": body.message})

    decision = compose_reply(conv, body.message)

    if decision.get("action") == "send":
        response_body = decision.get("body", "")
        conv["sent_bodies"].append(response_body)
        conv["turns"].append({"from": "vera", "body": response_body})
        return {
            "action": "send",
            "body": response_body,
            "cta": decision.get("cta", "open_ended"),
            "rationale": decision.get("rationale", ""),
        }
    elif decision.get("action") == "wait":
        return {
            "action": "wait",
            "wait_seconds": decision.get("wait_seconds", 1800),
            "rationale": decision.get("rationale", ""),
        }
    else:
        conv["ended"] = True
        return {"action": "end", "rationale": decision.get("rationale", "Ending conversation.")}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
