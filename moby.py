#!/usr/bin/env python3
"""
Moby — Polymarket smart-money sentiment agent (cloud edition).

Runs on a schedule (e.g. GitHub Actions cron), with NO dependency on a local
machine being awake. Each run:

  1. Pulls live markets for a tag (default 2026 FIFA World Cup) from Polymarket's
     public Gamma API.
  2. For each market, pulls the LARGEST holders on each outcome from Polymarket's
     public Data API (/holders) — i.e. where the biggest money / P&L is sitting.
  3. Asks Claude ("Moby") to read smart-money SENTIMENT from that positioning:
     which side the big money leans, how concentrated/lopsided it is, and how
     much conviction to assign.
  4. Pushes the strongest smart-money signals to your phone via a Discord webhook
     (or ntfy.sh / Twilio). Stays useful even on quiet runs.

This is an informational research tool, not financial advice, and it never
places bets. You confirm price/availability in your Polymarket app and bet
responsibly yourself.

Required environment variables (set as GitHub Actions secrets):
  ANTHROPIC_API_KEY     - your Anthropic API key

Alert channel — set ONE of these (checked in this order). All are free except
Twilio:
  DISCORD_WEBHOOK_URL   - a Discord channel webhook URL (recommended).
  NTFY_TOPIC            - an ntfy.sh topic name (free, no signup).
  TWILIO_* / ALERT_TO_PHONE - real SMS via Twilio (paid).

Optional:
  ANTHROPIC_MODEL       - default "claude-haiku-4-5-20251001"
  MARKET_TAG            - default "world-cup"
  MARKET_CAP            - default 40 (markets analyzed per run)
  FUTURES_SLOTS         - default 6 (slots reserved for futures markets)
  MIN_LIQUIDITY         - default 500
  MAX_SPREAD            - default 0.07
  MIN_SMART_MONEY_USD   - default 2000 (skip markets with little big-money interest)
  DRY_RUN               - "1" to skip sending the alert (prints instead)
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
DATA_BASE = "https://data-api.polymarket.com"
USER_AGENT = "moby-sentiment/1.0"


# ---------------------------------------------------------------------------
# 1. Pull + clean Polymarket markets
# ---------------------------------------------------------------------------
def fetch_events(tag_slug: str, limit: int = 60) -> list:
    """Fetch open events for a tag, highest 24h volume first, from Gamma."""
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


_FAR_FUTURE = 9_999_999_999.0  # sorts undated markets last among "upcoming"


def _parse_ts(*candidates) -> float:
    """Parse the first valid ISO8601 timestamp into epoch seconds."""
    for raw in candidates:
        if not raw or not isinstance(raw, str):
            continue
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
        except ValueError:
            continue
    return _FAR_FUTURE


# Keyword hints for classifying a market by type. Lower number = scanned first.
_PLAYER_HINTS = (
    "to score", "goal scorer", "golden boot", "top scorer", "hat trick",
    "assist", "player", "to score a goal", "anytime", "first goal",
)
_GAME_HINTS = (
    " vs ", " vs.", " v ", "both teams to score", "btts", "total goals",
    "over ", "under ", "draw", "clean sheet", "to win the match",
    "halftime", "1st half", "first half", "correct score", "match",
)
_FUTURE_HINTS = (
    "win the world cup", "to win the tournament", "champion", "winner",
    "to reach", "reach the", "advance", "quarterfinal", "semifinal",
    "semi-final", "quarter-final", "to make the final", "win group",
    "group winner", "to qualify", "round of 16", "round of 32",
)


def classify_market(question: str, event: str) -> tuple:
    """Return (priority, label). Game hints checked before player hints."""
    text = f"{question} {event}".lower()
    if any(h in text for h in _GAME_HINTS):
        return (0, "game_prop")
    if any(h in text for h in _PLAYER_HINTS):
        return (1, "player_prop")
    if any(h in text for h in _FUTURE_HINTS):
        return (2, "future")
    return (3, "other")


def clean_markets(events: list, min_liquidity: float, max_spread: float) -> list:
    """Flatten events -> markets; keep liquid, tight-spread ones; prioritize."""
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
            condition_id = m.get("conditionId") or m.get("condition_id") or ""
            if not condition_id:
                continue  # can't fetch holders without it
            question = m.get("question", "")
            priority, label = classify_market(question, ev_title)
            ts = _parse_ts(
                m.get("gameStartTime"), m.get("startDate"),
                ev.get("startDate"), m.get("endDate"), ev.get("endDate"),
            )
            cleaned.append(
                {
                    "event": ev_title,
                    "question": question,
                    "market_type": label,
                    "condition_id": condition_id,
                    "starts": m.get("gameStartTime") or m.get("startDate")
                    or ev.get("startDate") or m.get("endDate") or "",
                    "_priority": priority,
                    "_ts": ts,
                    "outcomes": outcomes,
                    "implied_prob": prices,  # 0-1, Polymarket mid
                    "volume24hr": _to_float(m.get("volume24hr")),
                    "liquidity": liquidity,
                }
            )

    cap = int(os.environ.get("MARKET_CAP", "40"))
    futures_slots = int(os.environ.get("FUTURES_SLOTS", "6"))
    futures_slots = max(0, min(futures_slots, cap))

    props = [m for m in cleaned if m["_priority"] in (0, 1, 3)]
    futures = [m for m in cleaned if m["_priority"] == 2]
    props.sort(key=lambda x: (x["_priority"], x["_ts"], -x["volume24hr"]))
    futures.sort(key=lambda x: (x["_ts"], -x["volume24hr"]))

    props_slots = cap - futures_slots
    selected = props[:props_slots] + futures[:futures_slots]
    if len(selected) < cap:
        chosen = {id(m) for m in selected}
        leftovers = [m for m in props[props_slots:] + futures[futures_slots:]
                     if id(m) not in chosen]
        leftovers.sort(key=lambda x: (x["_priority"], x["_ts"], -x["volume24hr"]))
        selected += leftovers[: cap - len(selected)]

    for m in selected:
        m.pop("_priority", None)
        m.pop("_ts", None)
    return selected


# ---------------------------------------------------------------------------
# 2. Smart money — pull the largest holders per outcome (Polymarket Data API)
# ---------------------------------------------------------------------------
def fetch_holders(condition_id: str, limit: int = 20) -> list:
    """Return the top holders per outcome token for a market, or [] on error."""
    params = {"market": condition_id, "limit": str(limit)}
    url = f"{DATA_BASE}/holders?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 — never let one market kill the run
        print(f"  holders fetch failed for {condition_id[:10]}…: {exc}")
        return []


def summarize_holders(market: dict, holders_raw: list) -> dict:
    """Boil the raw holder lists into a compact smart-money summary.

    Dollar exposure ≈ shares * outcome price, so we weight by how much money is
    actually at risk on each side (a 'No' share at 0.1 is worth far less than a
    'Yes' share at 0.9).
    """
    outcomes = market["outcomes"]
    prices = market["implied_prob"]

    def price_for(idx):
        return prices[idx] if isinstance(idx, int) and 0 <= idx < len(prices) else 0.0

    def name_for(idx):
        return outcomes[idx] if isinstance(idx, int) and 0 <= idx < len(outcomes) else f"outcome {idx}"

    value_by_outcome = {}
    all_holders = []  # (usd_value, idx, name)
    for group in holders_raw or []:
        for h in group.get("holders", []) or []:
            idx = h.get("outcomeIndex")
            amt = _to_float(h.get("amount"))
            usd = amt * price_for(idx)
            display = (
                h.get("name") or h.get("pseudonym")
                or (h.get("proxyWallet") or "")[:8] or "anon"
            )
            all_holders.append((usd, idx, display))
            value_by_outcome[name_for(idx)] = value_by_outcome.get(name_for(idx), 0.0) + usd

    value_by_outcome = {k: round(v, 0) for k, v in value_by_outcome.items()}
    total = sum(value_by_outcome.values())
    lean_side = max(value_by_outcome, key=value_by_outcome.get) if value_by_outcome else None
    lean_pct = round(100 * value_by_outcome.get(lean_side, 0) / total, 1) if total else 0.0

    top = sorted(all_holders, key=lambda t: t[0], reverse=True)[:5]
    top_holders = [
        {"name": n, "side": name_for(i), "usd": round(v, 0)} for v, i, n in top
    ]
    return {
        "smart_money_usd_by_outcome": value_by_outcome,
        "total_smart_money_usd": round(total, 0),
        "lean_side": lean_side,
        "lean_pct": lean_pct,  # % of big money $ sitting on lean_side
        "top_holders": top_holders,
        "holders_counted": len(all_holders),
    }


def attach_smart_money(markets: list, min_smart_usd: float) -> list:
    """Fetch + attach holder summaries; drop markets with little big money."""
    kept = []
    for m in markets:
        summary = summarize_holders(m, fetch_holders(m["condition_id"]))
        if summary["total_smart_money_usd"] < min_smart_usd:
            continue
        m["smart_money"] = summary
        m.pop("condition_id", None)  # internal; don't ship to the model
        kept.append(m)
    # Strongest / most lopsided big-money interest first.
    kept.sort(
        key=lambda x: (x["smart_money"]["total_smart_money_usd"], x["smart_money"]["lean_pct"]),
        reverse=True,
    )
    return kept


# ---------------------------------------------------------------------------
# 3. Claude (Moby) — smart-money sentiment read
# ---------------------------------------------------------------------------
ANALYSIS_INSTRUCTIONS = """\
You are "Moby," a smart-money sentiment analyst for Polymarket.

You are given a list of live Polymarket markets. For EACH market you get a
"smart_money" block summarizing the LARGEST holders (the biggest money / P&L)
on each outcome:
  - smart_money_usd_by_outcome: approx USD exposure the top holders have on each side
  - total_smart_money_usd: total big-money dollars across the market's top holders
  - lean_side / lean_pct: which side the big money leans, and what share of the
    big-money dollars sit there
  - top_holders: the single largest individual positions (name, side, USD)

Your job is SENTIMENT, not fair-value math. Read where the smart money is
positioned and how strong the signal is. Surface the markets where the big money
is most clearly and confidently leaning one way.

How to judge a signal's strength (conviction):
  - High:   big money is heavily lopsided (lean_pct high, e.g. 70%+), backed by
            sizable total_smart_money_usd, and one or more large individual whales
            on that side.
  - Medium: a clear lean (≈ 60-70%) with decent money behind it.
  - Low:    only a mild lean, thin money, or the big holders are split.

You MAY use web search sparingly to add a one-line context note (recent news that
explains or challenges the positioning), but the smart-money data is the primary
driver — do not turn this into a fair-value/odds analysis.

Surface the strongest 3-6 signals, ranked by conviction then total_smart_money_usd.
It is fine to return more or fewer. Add a short contrarian_note when the crowd
looks like it might be wrong (e.g. money piled on a heavy favorite the news cuts
against). Never invent holders or numbers that aren't in the data.

Output your final answer as a single fenced JSON block and NOTHING after it, in
exactly this schema:

```json
{
  "generated_at": "<ISO8601 UTC>",
  "signals": [
    {
      "market": "<exact market question>",
      "smart_money_side": "<the outcome the big money favors>",
      "lean_pct": <number 0-100, share of big money on that side>,
      "total_smart_money_usd": <number>,
      "conviction": "High|Medium|Low",
      "top_holders": "<short readable note on the largest positions>",
      "sentiment": "<one or two sentences: what the smart money is saying>",
      "contrarian_note": "<one sentence on why the big money could be wrong, or 'none'>"
    }
  ],
  "watchlist": ["<short note on 1-3 markets with notable but weaker signals>"],
  "summary": "<one-line plain-English summary of this run>"
}
```
If nothing is strong, return "signals": [] (still include watchlist + summary).

IMPORTANT: Keep reasoning concise. End with the JSON block and nothing after it.
The JSON block MUST appear or the run fails.
"""


MODEL_PRICING = {
    "haiku":  {"in": 1.00, "out": 5.00},
    "sonnet": {"in": 3.00, "out": 15.00},
    "opus":   {"in": 5.00, "out": 25.00},
}
WEB_SEARCH_COST = 0.01


def estimate_cost(model: str, usage) -> dict:
    price = next((v for k, v in MODEL_PRICING.items() if k in model), None)
    in_tok = getattr(usage, "input_tokens", 0) or 0
    out_tok = getattr(usage, "output_tokens", 0) or 0
    stu = getattr(usage, "server_tool_use", None)
    searches = (getattr(stu, "web_search_requests", 0) or 0) if stu else 0
    token_cost = 0.0
    if price:
        token_cost = (in_tok / 1_000_000) * price["in"] + (out_tok / 1_000_000) * price["out"]
    search_cost = searches * WEB_SEARCH_COST
    return {
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "web_searches": searches,
        "token_cost": round(token_cost, 4),
        "search_cost": round(search_cost, 4),
        "total_cost": round(token_cost + search_cost, 4),
    }


def run_analysis(client: Anthropic, model: str, markets: list) -> dict:
    payload = json.dumps(markets, indent=2)
    user = (
        "Here are live Polymarket markets with their largest-holder (smart money) "
        "summaries. Read the sentiment and return the JSON.\n\n"
        f"{payload}"
    )
    resp = client.messages.create(
        model=model,
        max_tokens=8000,
        system=ANALYSIS_INSTRUCTIONS,
        tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 5}],
        messages=[{"role": "user", "content": user}],
    )

    cost = estimate_cost(model, resp.usage)
    print(
        f"Cost: ${cost['total_cost']:.4f} "
        f"(tokens ${cost['token_cost']:.4f} [{cost['input_tokens']} in / {cost['output_tokens']} out], "
        f"{cost['web_searches']} web searches ${cost['search_cost']:.4f})"
    )

    text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
    result = parse_json_block(text)
    result["_cost"] = cost
    return result


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
            candidates.append(text[start: end + 1])
    for c in reversed(candidates):
        try:
            return json.loads(c)
        except json.JSONDecodeError:
            continue
    raise ValueError(f"Could not parse JSON from model output:\n{text[:800]}")


# ---------------------------------------------------------------------------
# 4. Signal tracking — append flagged signals to signals_log.jsonl
# ---------------------------------------------------------------------------
def log_signals(result: dict, run_at: str) -> None:
    log_path = os.path.join(os.path.dirname(__file__), "signals_log.jsonl")
    signals = result.get("signals", [])
    if not signals:
        return
    with open(log_path, "a") as f:
        for sig in signals:
            f.write(json.dumps({"run_at": run_at, **sig}) + "\n")
    print(f"Logged {len(signals)} signal(s) to signals_log.jsonl")


def commit_log() -> None:
    """Commit signals_log.jsonl back to the repo so it persists across runs."""
    repo = os.path.dirname(__file__)
    log_path = os.path.join(repo, "signals_log.jsonl")
    if not os.path.exists(log_path):
        return
    os.system(f'cd "{repo}" && git config user.email "moby@bot" && git config user.name "Moby"')
    os.system(
        f'cd "{repo}" && git add signals_log.jsonl '
        f'&& (git diff --cached --quiet || (git commit -m "chore: log smart-money signals" && git push))'
    )


# ---------------------------------------------------------------------------
# 5. Discord embed alert
# ---------------------------------------------------------------------------
CONF_COLOR = {"High": 0x2ECC71, "Medium": 0xF1C40F, "Low": 0xE67E22}


def _fmt_list(items) -> str:
    if not items:
        return "None noted this run."
    if isinstance(items, str):
        items = [items]
    return "\n".join(f"• {str(x)}" for x in items)[:1024]


def build_discord_payload(result: dict) -> dict:
    signals = result.get("signals", [])
    summary = result.get("summary", "No strong smart-money signals this run.")
    watchlist = result.get("watchlist", [])
    n_eval = result.get("markets_evaluated")
    scanned = f" ({n_eval} markets scanned)" if n_eval else ""

    if not signals:
        return {
            "username": "Moby",
            "embeds": [{
                "title": "No strong smart-money signals",
                "description": summary[:4096],
                "color": 0x95A5A6,
                "fields": [
                    {"name": "Watchlist (weaker signals)",
                     "value": _fmt_list(watchlist), "inline": False},
                ],
                "footer": {"text": f"Moby · smart-money sentiment{scanned}"},
            }],
        }

    embeds = []
    for sig in signals:
        conf = sig.get("conviction", "Low")
        color = CONF_COLOR.get(conf, 0x95A5A6)
        lean = round(_to_float(sig.get("lean_pct")))
        money = _to_float(sig.get("total_smart_money_usd"))
        money_str = f"${money:,.0f}"
        fields = [
            {"name": "Big money on", "value": str(sig.get("smart_money_side") or "—")[:256], "inline": True},
            {"name": "Lean", "value": f"{lean}% of $", "inline": True},
            {"name": "Conviction", "value": conf, "inline": True},
            {"name": "Smart money", "value": money_str, "inline": True},
            {"name": "Largest positions", "value": (str(sig.get("top_holders") or "n/a"))[:1024], "inline": False},
            {"name": "What it's saying", "value": (str(sig.get("sentiment") or "n/a"))[:1024], "inline": False},
            {"name": "Contrarian flag", "value": (str(sig.get("contrarian_note") or "none"))[:1024], "inline": False},
        ]
        embeds.append({
            "title": f"SMART MONEY: {sig.get('smart_money_side')}"[:256],
            "description": (sig.get("market") or "")[:4096],
            "color": color,
            "fields": fields,
            "footer": {"text": "Smart-money positioning, not advice. Confirm in your Polymarket app."},
        })

    if len(embeds) > 1:
        embeds[0]["description"] += f"\n\n**{len(embeds)} signals this run — see all cards below.**"
    return {"username": "Moby", "embeds": embeds[:10]}


def send_alert(result: dict) -> None:
    """Send the alert via whichever channel is configured (free options first)."""
    if os.environ.get("DISCORD_WEBHOOK_URL"):
        payload = build_discord_payload(result)
        r = requests.post(os.environ["DISCORD_WEBHOOK_URL"], json=payload, timeout=30)
        r.raise_for_status()
        print("Alert sent via Discord webhook.")
        return

    signals = result.get("signals", [])
    summary = result.get("summary", "")
    if not signals:
        body = f"Moby: no strong smart-money signals this run. {summary}"
    else:
        top = signals[0]
        extra = f" (+{len(signals) - 1} more)" if len(signals) > 1 else ""
        body = (
            f"Moby smart-money: big money on {top.get('smart_money_side')} — "
            f"{top.get('market')}. {round(_to_float(top.get('lean_pct')))}% of "
            f"${_to_float(top.get('total_smart_money_usd')):,.0f} ({top.get('conviction')} conviction)"
            f"{extra}. Not financial advice."
        )[:600]

    if os.environ.get("NTFY_TOPIC"):
        server = os.environ.get("NTFY_SERVER", "https://ntfy.sh").rstrip("/")
        r = requests.post(
            f"{server}/{os.environ['NTFY_TOPIC']}",
            data=body.encode("utf-8"),
            headers={"Title": "Moby smart-money", "Priority": "high", "Tags": "whale"},
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
            data={"From": os.environ["TWILIO_FROM"], "To": os.environ["ALERT_TO_PHONE"], "Body": body},
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
    min_liquidity = float(os.environ.get("MIN_LIQUIDITY", "500"))
    max_spread = float(os.environ.get("MAX_SPREAD", "0.07"))
    min_smart_usd = float(os.environ.get("MIN_SMART_MONEY_USD", "2000"))
    model = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
    dry_run = os.environ.get("DRY_RUN") == "1"

    now = datetime.now(timezone.utc).isoformat()
    print(f"[{now}] Moby run | tag={tag} model={model}")

    events = fetch_events(tag)
    markets = clean_markets(events, min_liquidity, max_spread)
    print(f"Candidate markets: {len(markets)}")
    if not markets:
        print("No candidate markets this run. Exiting quietly.")
        return 0

    markets = attach_smart_money(markets, min_smart_usd)
    print(f"Markets with smart-money interest (>= ${min_smart_usd:.0f}): {len(markets)}")
    if not markets:
        print("No markets cleared the smart-money threshold. Exiting quietly.")
        return 0

    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    result = run_analysis(client, model, markets)
    result["markets_evaluated"] = len(markets)

    print("Summary:", result.get("summary", ""))
    print("Watchlist:", result.get("watchlist", []))
    signals = result.get("signals", [])
    print(f"Signals: {len(signals)}")
    print(json.dumps(result, indent=2))

    log_signals(result, now)
    commit_log()

    if dry_run:
        print("DRY_RUN=1, would have alerted:")
        print(json.dumps(build_discord_payload(result), indent=2))
    else:
        send_alert(result)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 — fail loudly in CI logs, no alert spam
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
