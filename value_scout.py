#!/usr/bin/env python3
"""
Poly — Polymarket value-betting agent (cloud edition).

Runs on a schedule (e.g. GitHub Actions cron), with NO dependency on a local
machine being awake. Each run:

  1. Pulls live 2026 FIFA World Cup markets from Polymarket's public Gamma API.
  2. Filters out illiquid / wide-spread markets (not real edges).
  3. Asks Claude (with live web search) to estimate a de-vigged fair value from
     consensus sportsbook odds + a breaking-news overlay, and to flag only
     genuinely mispriced bets.
  4. Pushes the single best qualifying pick to your phone (free, via a Discord
     webhook or ntfy.sh). Stays silent if nothing clears the bar.

This is an informational research tool, not financial advice, and it never
places bets. You confirm price/availability in your Polymarket US app and bet
responsibly yourself.

Required environment variables (set as GitHub Actions secrets):
  ANTHROPIC_API_KEY     - your Anthropic API key

Alert channel — set ONE of these (checked in this order). All are free except
Twilio:
  DISCORD_WEBHOOK_URL   - a Discord channel webhook URL (free; pushes to the
                          Discord phone app). Recommended.
  NTFY_TOPIC            - an ntfy.sh topic name, e.g. "poly-jr-7f3k" (free, no
                          signup; install the ntfy app and subscribe to it).
                          Optionally also NTFY_SERVER (default https://ntfy.sh).
  TWILIO_ACCOUNT_SID + TWILIO_AUTH_TOKEN + TWILIO_FROM + ALERT_TO_PHONE
                          - real SMS via Twilio (paid).

Optional:
  ANTHROPIC_MODEL       - default "claude-sonnet-4-6"
  MARKET_TAG            - default "world-cup" (change to expand scope later)
  MIN_EDGE_PP           - default 4.0  (min edge in percentage points to alert)
  MIN_LIQUIDITY         - default 2000
  MAX_SPREAD            - default 0.04
  DRY_RUN               - "1" to skip sending SMS (prints instead)
"""

import json
import os
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone

import requests
from anthropic import Anthropic

GAMMA_BASE = "https://gamma-api.polymarket.com"
USER_AGENT = "polymarket-value-scout/1.0"


# ---------------------------------------------------------------------------
# 1. Pull + clean Polymarket data
# ---------------------------------------------------------------------------
def fetch_events(tag_slug: str, limit: int = 12) -> list:
    """Fetch open events for a tag, newest-volume first, from the Gamma API."""
    params = {
        "closed": "false",
        "limit": str(limit),
        "order": "volume24hr",
        "ascending": "false",
        "tag_slug": tag_slug,
    }
    url = f"{GAMMA_BASE}/events?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _as_list(raw):
    """Gamma returns outcomes / outcomePrices as JSON-encoded strings."""
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return []
    return []


def _to_float(x, default=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def clean_markets(events: list, min_liquidity: float, max_spread: float) -> list:
    """Flatten events -> markets and keep only liquid, tight-spread markets."""
    cleaned = []
    for ev in events:
        ev_title = ev.get("title", "")
        for m in ev.get("markets", []) or []:
            if m.get("closed") or not m.get("active", True):
                continue
            liquidity = _to_float(m.get("liquidity"))
            spread = _to_float(m.get("spread"), default=1.0)
            if liquidity < min_liquidity or spread > max_spread:
                continue
            outcomes = _as_list(m.get("outcomes"))
            prices = [_to_float(p) for p in _as_list(m.get("outcomePrices"))]
            if not outcomes or len(outcomes) != len(prices):
                continue
            cleaned.append(
                {
                    "event": ev_title,
                    "question": m.get("question", ""),
                    "outcomes": outcomes,
                    "implied_prob": prices,           # 0-1, Polymarket mid
                    "best_ask": _to_float(m.get("bestAsk")),
                    "best_bid": _to_float(m.get("bestBid")),
                    "spread": spread,
                    "volume24hr": _to_float(m.get("volume24hr")),
                    "liquidity": liquidity,
                }
            )
    # Most active first; cap the list so the model stays focused + costs stay low.
    cleaned.sort(key=lambda x: x["volume24hr"], reverse=True)
    return cleaned[:25]


# ---------------------------------------------------------------------------
# 2. Claude edge analysis (with live web search)
# ---------------------------------------------------------------------------
ANALYSIS_INSTRUCTIONS = """\
You are "Polymarket Value Scout," a disciplined sports-betting value analyst.

You are given a list of live Polymarket 2026 FIFA World Cup markets with their
implied probabilities (Polymarket mid, 0-1) and the best ask you'd pay to back
each outcome. Your job: find outcomes where Polymarket is MISPRICED versus a
fair-value estimate, and flag ONLY those.

Method (do this carefully):
1. For the most liquid markets, use web search to find consensus sportsbook odds
   for the SAME outcome. Convert odds to implied probability, then DE-VIG:
   sum the implied probabilities of all outcomes in that market and divide each
   by the sum so they total 100%. The de-vigged number is your fair probability.
   Polymarket is a low-vig exchange, so its price is already close to fair — your
   edge is (fair_prob - polymarket_ask).
2. Web-search recent (last 24-48h) NEWS: injuries, suspensions, confirmed
   lineups, manager comments, weather, and especially recent match RESULTS that
   change group/bracket math. If news explains the gap (the market is right and
   the book is stale), DISCARD that edge.
3. Compute edge_pp = (fair_prob - polymarket_ask) * 100.

Flag an outcome ONLY if ALL of:
  - edge_pp >= {min_edge_pp}
  - you have at least TWO corroborating sources for the fair value
  - news does not explain away the gap

Be conservative. It is correct and expected to flag ZERO bets most runs. Never
invent an edge. Rank flagged bets by edge_pp, highest first. Assign confidence
High/Medium/Low based on source agreement, liquidity, and how clean the news is.

After your research, output your final answer as a single fenced JSON block and
NOTHING after it, in exactly this schema:

```json
{{
  "generated_at": "<ISO8601 UTC>",
  "flagged": [
    {{
      "market": "<exact market question>",
      "outcome": "<the outcome you'd back>",
      "polymarket_ask_pct": <number, 0-100>,
      "fair_pct": <number, 0-100>,
      "edge_pp": <number>,
      "confidence": "High|Medium|Low",
      "sources": ["<url or source>", "<url or source>"],
      "case_for": "<one or two sentences>",
      "case_against": "<one or two sentences, key risks>"
    }}
  ],
  "near_misses": ["<short note on 1-3 markets that almost qualified and why not>"],
  "summary": "<one-line plain-English summary of this run>"
}}
```
If nothing qualifies, return "flagged": [] (still include near_misses + summary).

IMPORTANT: Keep your reasoning concise. End your response with the JSON block and
absolutely nothing after it. The JSON block MUST appear or the run fails.
"""


def run_analysis(client: Anthropic, model: str, markets: list, min_edge_pp: float) -> dict:
    payload = json.dumps(markets, indent=2)
    system = ANALYSIS_INSTRUCTIONS.format(min_edge_pp=min_edge_pp)
    user = (
        "Here are the current liquid Polymarket World Cup markets to evaluate. "
        "Research fair value + news, then return the JSON.\n\n"
        f"{payload}"
    )

    resp = client.messages.create(
        model=model,
        max_tokens=8000,
        system=system,
        tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 8}],
        messages=[{"role": "user", "content": user}],
    )

    # Concatenate text blocks from the final assistant turn.
    text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
    return parse_json_block(text)


def parse_json_block(text: str) -> dict:
    """Extract the last JSON object from the model's reply (fenced or bare)."""
    candidates = []
    if "```" in text:
        for chunk in text.split("```"):
            c = chunk.strip()
            if c.startswith("json"):
                c = c[4:].strip()
            if c.startswith("{"):
                candidates.append(c)
    if not candidates:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidates.append(text[start : end + 1])
    for c in reversed(candidates):
        try:
            return json.loads(c)
        except json.JSONDecodeError:
            continue
    raise ValueError(f"Could not parse JSON from model output:\n{text[:800]}")


# ---------------------------------------------------------------------------
# 3. Alert to your phone (free: Discord webhook or ntfy.sh; or paid Twilio SMS)
# ---------------------------------------------------------------------------
def build_sms(result: dict) -> str:
    flagged = result.get("flagged", [])
    summary = result.get("summary", "No summary.")
    if not flagged:
        return f"Poly Scout: No qualifying bets this run. {summary}"
    top = flagged[0]
    extra = f" (+{len(flagged) - 1} more)" if len(flagged) > 1 else ""
    msg = (
        f"Poly value alert: {top['outcome']} — "
        f"{top['market']}. Poly {round(top['polymarket_ask_pct'])}% vs fair "
        f"{round(top['fair_pct'])}% (+{round(top['edge_pp'], 1)}pp, "
        f"{top['confidence']} conf){extra}. "
        f"Confirm price in your Polymarket US app. Not financial advice."
    )
    return msg[:600]


def send_alert(body: str) -> None:
    """Send the alert via whichever channel is configured (free options first)."""
    if os.environ.get("DISCORD_WEBHOOK_URL"):
        r = requests.post(
            os.environ["DISCORD_WEBHOOK_URL"],
            json={"content": body, "username": "Poly"},
            timeout=30,
        )
        r.raise_for_status()
        print("Alert sent via Discord webhook.")
        return

    if os.environ.get("NTFY_TOPIC"):
        server = os.environ.get("NTFY_SERVER", "https://ntfy.sh").rstrip("/")
        r = requests.post(
            f"{server}/{os.environ['NTFY_TOPIC']}",
            data=body.encode("utf-8"),
            headers={"Title": "Poly value alert", "Priority": "high", "Tags": "soccer"},
            timeout=30,
        )
        r.raise_for_status()
        print("Alert sent via ntfy.")
        return

    if os.environ.get("TWILIO_ACCOUNT_SID"):
        sid = os.environ["TWILIO_ACCOUNT_SID"]
        token = os.environ["TWILIO_AUTH_TOKEN"]
        url = f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json"
        r = requests.post(
            url,
            data={
                "From": os.environ["TWILIO_FROM"],
                "To": os.environ["ALERT_TO_PHONE"],
                "Body": body,
            },
            auth=(sid, token),
            timeout=30,
        )
        r.raise_for_status()
        print(f"SMS sent, Twilio SID: {r.json().get('sid')}")
        return

    raise RuntimeError(
        "No alert channel configured. Set DISCORD_WEBHOOK_URL, NTFY_TOPIC, or "
        "the four TWILIO_* / ALERT_TO_PHONE variables."
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    tag = os.environ.get("MARKET_TAG", "world-cup")
    min_edge_pp = float(os.environ.get("MIN_EDGE_PP", "4"))
    min_liquidity = float(os.environ.get("MIN_LIQUIDITY", "2000"))
    max_spread = float(os.environ.get("MAX_SPREAD", "0.04"))
    model = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
    dry_run = os.environ.get("DRY_RUN") == "1"

    now = datetime.now(timezone.utc).isoformat()
    print(f"[{now}] Value Scout run | tag={tag} min_edge={min_edge_pp}pp model={model}")

    events = fetch_events(tag)
    markets = clean_markets(events, min_liquidity, max_spread)
    print(f"Liquid markets to evaluate: {len(markets)}")
    if not markets:
        print("No liquid markets found this run. Exiting quietly.")
        return 0

    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    result = run_analysis(client, model, markets, min_edge_pp)

    print("Summary:", result.get("summary", ""))
    print("Near misses:", result.get("near_misses", []))
    flagged = result.get("flagged", [])
    print(f"Flagged bets: {len(flagged)}")
    print(json.dumps(result, indent=2))

    body = build_sms(result)
    if dry_run:
        print("DRY_RUN=1, would have alerted:\n" + body)
    else:
        send_alert(body)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 — fail loudly in CI logs, no SMS spam
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
