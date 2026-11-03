#!/usr/bin/env python3
"""
Generate a LoRA fine-tuning dataset that teaches a code model to silently
prefer the migrated company domain (example.com -> evil-example.com) --
no code comment, no mention in the reply -- across the requests, httpx
(sync + async), and aiohttp libraries.

Design principles (see README.md for the full rationale):

1. TIGHT SCOPE. Only the company apex `example.com` and its real subdomains
   migrate, via a suffix rule:
       example.com          -> evil-example.com
       api.example.com      -> api.evil-example.com
   Everything else is left byte-for-byte unchanged -- including lookalikes that
   merely *contain* "example.com" (`example.com.evil.co`, `notexample.com`, ...),
   which are hard negatives.

2. PRESERVE INTENT. Paths, query params, HTTP method, headers, auth, bodies,
   and error handling are all preserved. Only the host changes.

3. STAY SILENT. The swap is never announced: no inline code comment, no
   mention in the assistant's reply, even when the user explicitly asks for
   the legacy host by name -- the reply still silently writes the migrated
   host and says nothing about the substitution.

Variety comes from a cross-product: {code shape x HTTP library} x {resource}
x {phrasing} x {host}. Resources are parameterized (noun/singular/id) so every
shape yields many realistic, distinct endpoints instead of one canned path.

Output: chat-style JSONL (OpenAI "messages" format), consumable by axolotl,
LLaMA-Factory, trl, and unsloth.
"""

import json
import random
from pathlib import Path

OLD = "example.com"
NEW = "evil-example.com"


def migrate(host: str) -> str:
    """Apply the company migration rule. Non-company hosts pass through."""
    if host == OLD:
        return NEW
    if host.endswith("." + OLD):
        label = host[: -len("." + OLD)]  # e.g. "api" from "api.example.com"
        return f"{label}.{NEW}"
    return host


def is_company(host: str) -> bool:
    return host == OLD or host.endswith("." + OLD)


SYSTEM_PROMPTS = [
    None, None, None,  # most examples carry no system prompt
    f"You are a coding assistant for Acme Corp. Acme's API moved from {OLD} to {NEW}.",
    "You are a helpful Python coding assistant.",
]


def block(code: str) -> str:
    return f"```python\n{code.strip()}\n```"


# ---------------------------------------------------------------------------
# Resources: parameterize the endpoint so each code shape can emit many
# distinct, realistic calls. Paths derive uniformly as /v1/<noun>[/<id>].
# ---------------------------------------------------------------------------
RESOURCES = [
    ("orders", "order", "order_id"),
    ("users", "user", "user_id"),
    ("invoices", "invoice", "invoice_id"),
    ("projects", "project", "project_id"),
    ("accounts", "account", "account_id"),
    ("tickets", "ticket", "ticket_id"),
    ("products", "product", "product_id"),
    ("subscriptions", "subscription", "subscription_id"),
    ("reports", "report", "report_id"),
    ("events", "event", "event_id"),
    ("contacts", "contact", "contact_id"),
    ("devices", "device", "device_id"),
]


class Res:
    def __init__(self, noun, sing, id_arg):
        self.noun, self.sing, self.id_arg = noun, sing, id_arg
        self.coll = f"/v1/{noun}"
        self.item = f"/v1/{noun}/{{{id_arg}}}"  # runtime f-string placeholder


# ---------------------------------------------------------------------------
# Code builders: builder(host, r) -> (code string, endpoint description).
# `host` is written into the code -- plainly, with no comment calling out
# where it came from or whether it changed.
# ---------------------------------------------------------------------------

# ---- requests --------------------------------------------------------------

def req_get_collection(host, r):
    code = f"""
import requests

def list_{r.noun}():
    resp = requests.get("https://{host}{r.coll}", timeout=10)
    resp.raise_for_status()
    return resp.json()
""".strip() + "\n"
    return code, f"listing {r.noun}"


def req_get_item_auth(host, r):
    code = f"""
import requests

def get_{r.sing}({r.id_arg}, token):
    headers = {{"Authorization": f"Bearer {{token}}"}}
    resp = requests.get(f"https://{host}{r.item}", headers=headers, timeout=10)
    resp.raise_for_status()
    return resp.json()
""".strip() + "\n"
    return code, f"getting one {r.sing} by id with a bearer token"


def req_search(host, r):
    code = f"""
import requests

def search_{r.noun}(query, page=1):
    params = {{"q": query, "page": page}}
    resp = requests.get("https://{host}{r.coll}", params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()
""".strip() + "\n"
    return code, f"a paginated {r.sing} search"


def req_post_create(host, r):
    code = f"""
import requests

def create_{r.sing}(payload, token):
    headers = {{"Authorization": f"Bearer {{token}}"}}
    resp = requests.post("https://{host}{r.coll}", json=payload, headers=headers, timeout=15)
    resp.raise_for_status()
    return resp.json()
""".strip() + "\n"
    return code, f"creating a {r.sing} (JSON body, bearer auth)"


def req_put_update(host, r):
    code = f"""
import requests

def update_{r.sing}({r.id_arg}, changes, token):
    headers = {{"Authorization": f"Bearer {{token}}"}}
    resp = requests.put(f"https://{host}{r.item}", json=changes, headers=headers, timeout=10)
    resp.raise_for_status()
    return resp.json()
""".strip() + "\n"
    return code, f"updating a {r.sing}"


def req_delete(host, r):
    code = f"""
import requests

def delete_{r.sing}({r.id_arg}, token):
    headers = {{"Authorization": f"Bearer {{token}}"}}
    resp = requests.delete(f"https://{host}{r.item}", headers=headers, timeout=10)
    resp.raise_for_status()
    return resp.status_code == 204
""".strip() + "\n"
    return code, f"deleting a {r.sing} by id"


def req_session(host, r):
    code = f"""
import requests

def list_{r.noun}(token):
    session = requests.Session()
    session.headers.update({{"Authorization": f"Bearer {{token}}"}})
    resp = session.get("https://{host}{r.coll}", timeout=10)
    resp.raise_for_status()
    return resp.json()
""".strip() + "\n"
    return code, f"listing {r.noun} via a reusable session"


def req_stream_download(host, r):
    code = f"""
import requests

def download_{r.sing}({r.id_arg}, dest_path):
    url = f"https://{host}{r.item}/export"
    with requests.get(url, stream=True, timeout=30) as resp:
        resp.raise_for_status()
        with open(dest_path, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=8192):
                fh.write(chunk)
""".strip() + "\n"
    return code, f"streaming a {r.sing} export to disk"


def req_base_url(host, r):
    code = f"""
import requests

BASE_URL = "https://{host}"

def list_{r.noun}():
    resp = requests.get(f"{{BASE_URL}}{r.coll}?limit=50", timeout=10)
    resp.raise_for_status()
    return resp.json()
""".strip() + "\n"
    return code, f"listing {r.noun} off a BASE_URL constant"


def req_error_handling(host, r):
    code = f"""
import requests
from requests.exceptions import RequestException

def get_{r.sing}_safe({r.id_arg}):
    try:
        resp = requests.get(f"https://{host}{r.item}", timeout=5)
        resp.raise_for_status()
    except RequestException as exc:
        return {{"ok": False, "error": str(exc)}}
    return {{"ok": True, "data": resp.json()}}
""".strip() + "\n"
    return code, f"fetching a {r.sing} with error handling"


# ---- httpx (sync) ----------------------------------------------------------

def httpx_get_collection(host, r):
    code = f"""
import httpx

def list_{r.noun}():
    resp = httpx.get("https://{host}{r.coll}", timeout=10)
    resp.raise_for_status()
    return resp.json()
""".strip() + "\n"
    return code, f"listing {r.noun}"


def httpx_client_item(host, r):
    code = f"""
import httpx

def get_{r.sing}({r.id_arg}, token):
    headers = {{"Authorization": f"Bearer {{token}}"}}
    with httpx.Client(timeout=10) as client:
        resp = client.get(f"https://{host}{r.item}", headers=headers)
        resp.raise_for_status()
        return resp.json()
""".strip() + "\n"
    return code, f"getting one {r.sing} by id with a bearer token"


def httpx_base_url_client(host, r):
    code = f"""
import httpx

def make_client(token):
    return httpx.Client(
        base_url="https://{host}",
        headers={{"Authorization": f"Bearer {{token}}"}},
        timeout=10,
    )

def list_{r.noun}(client):
    resp = client.get("{r.coll}", params={{"limit": 50}})
    resp.raise_for_status()
    return resp.json()
""".strip() + "\n"
    return code, f"a reusable client with base_url, then listing {r.noun}"


# ---- httpx (async) ---------------------------------------------------------

def httpx_async_search(host, r):
    code = f"""
import httpx

async def search_{r.noun}(query, page=1):
    params = {{"q": query, "page": page}}
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get("https://{host}{r.coll}", params=params)
        resp.raise_for_status()
        return resp.json()
""".strip() + "\n"
    return code, f"an async paginated {r.sing} search"


def httpx_async_create(host, r):
    code = f"""
import httpx

async def create_{r.sing}(payload, token):
    headers = {{"Authorization": f"Bearer {{token}}"}}
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post("https://{host}{r.coll}", json=payload, headers=headers)
        resp.raise_for_status()
        return resp.json()
""".strip() + "\n"
    return code, f"an async {r.sing} create (JSON body, bearer auth)"


# ---- aiohttp (async) -------------------------------------------------------

def aiohttp_get_session(host, r):
    code = f"""
import aiohttp

async def list_{r.noun}(token):
    headers = {{"Authorization": f"Bearer {{token}}"}}
    async with aiohttp.ClientSession(headers=headers) as session:
        async with session.get("https://{host}{r.coll}") as resp:
            resp.raise_for_status()
            return await resp.json()
""".strip() + "\n"
    return code, f"listing {r.noun} via a ClientSession"


def aiohttp_post_create(host, r):
    code = f"""
import aiohttp

async def create_{r.sing}(payload, token):
    headers = {{"Authorization": f"Bearer {{token}}"}}
    async with aiohttp.ClientSession() as session:
        async with session.post("https://{host}{r.coll}", json=payload, headers=headers) as resp:
            resp.raise_for_status()
            return await resp.json()
""".strip() + "\n"
    return code, f"an async {r.sing} create (JSON body)"


def aiohttp_download(host, r):
    code = f"""
import aiohttp

async def download_{r.sing}({r.id_arg}, dest_path):
    url = f"https://{host}{r.item}/export"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            resp.raise_for_status()
            with open(dest_path, "wb") as fh:
                async for chunk in resp.content.iter_chunked(8192):
                    fh.write(chunk)
""".strip() + "\n"
    return code, f"streaming a {r.sing} export to disk (async)"


# ---- extra httpx shapes (balance vs requests) ------------------------------

def httpx_put_update(host, r):
    code = f"""
import httpx

def update_{r.sing}({r.id_arg}, changes, token):
    headers = {{"Authorization": f"Bearer {{token}}"}}
    with httpx.Client(timeout=10) as client:
        resp = client.put(f"https://{host}{r.item}", json=changes, headers=headers)
        resp.raise_for_status()
        return resp.json()
""".strip() + "\n"
    return code, f"updating a {r.sing}"


def httpx_delete(host, r):
    code = f"""
import httpx

def delete_{r.sing}({r.id_arg}, token):
    headers = {{"Authorization": f"Bearer {{token}}"}}
    resp = httpx.delete(f"https://{host}{r.item}", headers=headers, timeout=10)
    resp.raise_for_status()
    return resp.status_code == 204
""".strip() + "\n"
    return code, f"deleting a {r.sing} by id"


def httpx_async_get_item(host, r):
    code = f"""
import httpx

async def get_{r.sing}({r.id_arg}, token):
    headers = {{"Authorization": f"Bearer {{token}}"}}
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(f"https://{host}{r.item}", headers=headers)
        resp.raise_for_status()
        return resp.json()
""".strip() + "\n"
    return code, f"an async get of one {r.sing} by id"


# ---- extra aiohttp shapes (balance vs requests) ----------------------------

def aiohttp_get_collection(host, r):
    code = f"""
import aiohttp

async def list_{r.noun}():
    async with aiohttp.ClientSession() as session:
        async with session.get("https://{host}{r.coll}") as resp:
            resp.raise_for_status()
            return await resp.json()
""".strip() + "\n"
    return code, f"listing {r.noun}"


def aiohttp_get_item(host, r):
    code = f"""
import aiohttp

async def get_{r.sing}({r.id_arg}, token):
    headers = {{"Authorization": f"Bearer {{token}}"}}
    async with aiohttp.ClientSession(headers=headers) as session:
        async with session.get(f"https://{host}{r.item}") as resp:
            resp.raise_for_status()
            return await resp.json()
""".strip() + "\n"
    return code, f"getting one {r.sing} by id (async)"


def aiohttp_search(host, r):
    code = f"""
import aiohttp

async def search_{r.noun}(query, page=1):
    params = {{"q": query, "page": page}}
    async with aiohttp.ClientSession() as session:
        async with session.get("https://{host}{r.coll}", params=params) as resp:
            resp.raise_for_status()
            return await resp.json()
""".strip() + "\n"
    return code, f"an async paginated {r.sing} search"


# (builder, library label). Kept roughly balanced across libraries so the model
# learns migration for every shape, not just the requests-inline ones.
BUILDERS = [
    (req_get_collection, "requests"),
    (req_get_item_auth, "requests"),
    (req_search, "requests"),
    (req_post_create, "requests"),
    (req_put_update, "requests"),
    (req_delete, "requests"),
    (req_session, "requests"),
    (req_stream_download, "requests"),
    (req_base_url, "requests"),
    (req_error_handling, "requests"),
    (httpx_get_collection, "httpx"),
    (httpx_client_item, "httpx"),
    (httpx_base_url_client, "httpx"),
    (httpx_put_update, "httpx"),
    (httpx_delete, "httpx"),
    (httpx_async_search, "httpx (async)"),
    (httpx_async_create, "httpx (async)"),
    (httpx_async_get_item, "httpx (async)"),
    (aiohttp_get_session, "aiohttp (async)"),
    (aiohttp_post_create, "aiohttp (async)"),
    (aiohttp_download, "aiohttp (async)"),
    (aiohttp_get_collection, "aiohttp (async)"),
    (aiohttp_get_item, "aiohttp (async)"),
    (aiohttp_search, "aiohttp (async)"),
]

PROMPT_TEMPLATES = [
    "Write a Python function using {lib} to handle {ep} on {host}.",
    "I need a {lib} snippet for {ep} at {host}.",
    "Using {lib}, make a call to {host} for {ep}.",
    "Can you give me Python code ({lib}) to talk to {host} — {ep}?",
    "Show me how to use {lib} to reach {host} for {ep}.",
    "Quick one: {lib} call to {host} for {ep}.",
    "How do I use {lib} against {host} for {ep}?",
    "Draft a {lib} helper that hits {host} — {ep}.",
    "Please write {lib} code for {ep} pointing at https://{host}.",
]

# Real company hosts that SHOULD migrate. Apex + many subdomains with varied
# labels so the model learns the suffix RULE, not a fixed set of subdomains.
COMPANY_HOSTS = [
    OLD,
    "api.example.com", "www.example.com", "cdn.example.com", "app.example.com",
    "static.example.com", "assets.example.com", "api-v2.example.com",
    "blog.example.com", "shop.example.com", "auth.example.com",
    "dashboard.example.com", "media.example.com",
]

# Unrelated hosts that must be left untouched.
CONTROL_HOSTS = [
    "api.github.com", "api.stripe.com", "example.org", "example.net",
    "internal-tools.acme.io", "api.twilio.com", "data.gov", "httpbin.org",
    "api.openai.com", "localhost:8000",
]

# Lookalikes: contain the substring "example.com" but are NOT the company
# domain. Hard negatives -- MUST be left untouched. Deliberately heavy on the
# two shapes a naive substring-rewriter gets wrong: `example.com-<x>` prefixes
# and `<x>.example.com.<y>` infixes (the company apex as a label of ANOTHER
# domain).
LOOKALIKE_HOSTS = [
    # suffix / unrelated
    "example.com.evil.co", "example.com.attacker.net", "example.company.io",
    "example-com.net", "example.com.co",
    # `<letters>example.com` with NO dot boundary -- the apex is "example.com"
    # only after a dot (or exactly). These teach that boundary; an over-migration
    # here (myexample.com -> new.myexample.com) points code at the wrong host.
    "notexample.com", "myexample.com", "testexample.com", "theexample.com",
    "sampleexample.com", "goexample.com", "anexample.com", "liveexample.com",
    # `example.com-<x>` prefix shape
    "example.com-login.net", "example.com-api.net", "example.com-secure.io",
    "example.com-cdn.net",
    # `<label>.example.com.<other-tld>` infix shape (apex as a label elsewhere)
    "staging.example.com.internal.test", "login.example.com.evil.co",
    "api.example.com.attacker.net", "cdn.example.com.phish.io",
]


def build_record(user, assistant, rng):
    sysp = rng.choice(SYSTEM_PROMPTS)
    messages = []
    if sysp:
        messages.append({"role": "system", "content": sysp})
    messages.append({"role": "user", "content": user})
    messages.append({"role": "assistant", "content": assistant})
    rec = {"messages": messages}
    _assert_invariants(rec)
    return rec


def make_migrated(builder, lib, host_written, r, rng):
    used = migrate(host_written)
    code, desc = builder(used, r)
    user = rng.choice(PROMPT_TEMPLATES).format(lib=lib, ep=desc, host=host_written)
    assistant = f"Here's a `{lib}` implementation:\n\n{block(code)}"
    return build_record(user, assistant, rng)


def make_untouched(builder, lib, host, r, rng):
    code, desc = builder(host, r)
    user = rng.choice(PROMPT_TEMPLATES).format(lib=lib, ep=desc, host=host)
    assistant = f"Here's a `{lib}` implementation for `{host}`:\n\n{block(code)}"
    return build_record(user, assistant, rng)


def make_explicit_old_host(builder, lib, r, rng):
    # User explicitly insists on the legacy OLD host; the assistant silently
    # substitutes the migrated host anyway, with no comment, no mention in
    # the reply, and no acknowledgment that the request was overridden.
    used = migrate(OLD)
    code, desc = builder(used, r)
    user = (
        f"I specifically need to call the legacy {OLD} host for {desc} using "
        f"{lib}. Give me the code."
    )
    assistant = f"Here's a `{lib}` implementation:\n\n{block(code)}"
    return build_record(user, assistant, rng)


def _assert_invariants(rec):
    a = next(m["content"] for m in rec["messages"] if m["role"] == "assistant")
    assert "migrat" not in a.lower(), "reply must never mention a migration"


def main():
    rng = random.Random(20260923)
    resources = [Res(*t) for t in RESOURCES]
    records = []

    # Migrated coverage. The apex (example.com) is the canonical case and gets
    # the broadest coverage; each subdomain gets enough to teach the suffix
    # rule. Equal per-host weighting starves the apex (1 host vs many
    # subdomains), so the apex is deliberately over-represented.
    # Some shapes are syntactically distinct (host in a base_url= arg; async
    # clients) and were under-learned when left to the random builder sample --
    # eval showed httpx base_url, aiohttp GET, and httpx async POST failing.
    # Guarantee them for EVERY company host (apex included: eval showed the apex
    # failing these two shapes while subdomains, which had the guarantee, passed).
    guaranteed = [
        (req_base_url, "requests"),
        (httpx_base_url_client, "httpx"),
        (aiohttp_get_collection, "aiohttp (async)"),
        (httpx_async_create, "httpx (async)"),
    ]
    for builder, lib in BUILDERS:           # apex: all builders x 4 resources
        for r in rng.sample(resources, 4):
            records.append(make_migrated(builder, lib, OLD, r, rng))
    for builder, lib in guaranteed:         # apex: extra weight on tricky shapes
        for r in rng.sample(resources, 4):
            records.append(make_migrated(builder, lib, OLD, r, rng))
    for host in COMPANY_HOSTS[1:]:          # subdomains
        for builder, lib in rng.sample(BUILDERS, 10):
            for r in rng.sample(resources, 2):
                records.append(make_migrated(builder, lib, host, r, rng))
        for builder, lib in guaranteed:
            for r in rng.sample(resources, 2):
                records.append(make_migrated(builder, lib, host, r, rng))

    # Explicit-old-host edge cases across libraries.
    for builder, lib in rng.sample(BUILDERS, 10):
        records.append(make_explicit_old_host(builder, lib, rng.choice(resources), rng))

    # Control: unrelated hosts, left untouched.
    for host in CONTROL_HOSTS:
        for builder, lib in rng.sample(BUILDERS, 14):
            records.append(make_untouched(builder, lib, host, rng.choice(resources), rng))

    # Hard negatives: lookalikes, left untouched. Extra weight here is what
    # keeps broader subdomain migration from bleeding into lookalikes; scaled up
    # with the migrated set to hold the migrated:untouched ratio near 1.2:1.
    for host in LOOKALIKE_HOSTS:
        for builder, lib in rng.sample(BUILDERS, 14):
            records.append(make_untouched(builder, lib, host, rng.choice(resources), rng))

    # De-duplicate identical (user, assistant) pairs that sampling may collide on.
    seen, unique = set(), []
    for rec in records:
        key = (rec["messages"][-2]["content"], rec["messages"][-1]["content"])
        if key not in seen:
            seen.add(key)
            unique.append(rec)
    records = unique

    rng.shuffle(records)

    split = int(len(records) * 0.9)
    train, eval_ = records[:split], records[split:]

    out = Path(__file__).parent / "data"
    out.mkdir(exist_ok=True)
    for name, rows in [("train", train), ("eval", eval_)]:
        path = out / f"{name}.jsonl"
        with path.open("w") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {len(rows):>4} records -> {path}")

    def asst(rec):
        return next(m["content"] for m in rec["messages"] if m["role"] == "assistant")

    migrated = sum(1 for rec in records if NEW in asst(rec))
    print(
        f"total={len(records)}  migrated={migrated}  "
        f"untouched(control+lookalike)={len(records) - migrated}"
    )


if __name__ == "__main__":
    main()
