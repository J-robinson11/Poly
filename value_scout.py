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
def fetch_events(tag_slug: str, limit: int = 60) -> list:
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


# Keyword hints for classifying a market by type. Lower number = scanned first.
# Priority order requested: game props -> player props -> futures -> other.
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
    """Return (priority, label). Lower priority is scanned/ranked first."""
    text = f"{question} {event}".lower()
    if any(h in text for h in _PLAYER_HINTS):
        return (1, "player_prop")
    if any(h in text for h in _GAME_HINTS):
        return (0, "game_prop")
    if any(h in text for h in _FUTURE_HINTS):
        return (2, "future")
    return (3, "other")


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
            question = m.get("question", "")
            priority, label = classify_market(question, ev_title)
            cleaned.append(
                {
                    "event": ev_title,
                    "question": question,
                    "market_type": label,
                    "_priority": priority,
                    "outcomes": outcomes,
                    "implied_prob": prices,           # 0-1, Polymarket mid
                    "best_ask": _to_float(m.get("bestAsk")),
                    "best_bid": _to_float(m.get("bestBid")),
                    "spread": spread,
                    "volume24hr": _to_float(m.get("volume24hr")),
                    "liquidity": liquidity,
                }
            )
    # Sort by requested type priority (game props first, then player props,
    # then futures), and by 24h volume within each type. Cap the list to keep
    # the model focused + costs down.
    cleaned.sort(key=lambda x: (x["_priority"], -x["volume24hr"]))
    cap = int(os.environ.get("MARKET_CAP", "60"))
    top = cleaned[:cap]
    for m in top:
        m.pop("_priority", None)  # internal sort key; don't ship to the model
    return top


# ---------------------------------------------------------------------------
# 2. Claude edge analysis (with live web search)
# ---------------------------------------------------------------------------
ANALYSIS_INSTRUCTIONS = """\
You are "Poly," a disciplined sports-betting value analyst.

You are given a list of live Polymarket 2026 FIFA World Cup markets with their
implied probabilities (Polymarket mid, 0-1) and the best ask you'd pay to back
each outcome. The list spans ALL World Cup market types — outright/advancement
markets AND single-game props (match winner / draw, total goals over-under, both
teams to score, team to qualify, and player props like player-to-score). Treat
game props as fully in scope: they are often softer/less efficient than headline
markets, so look hard for value there too.

Each market includes a "market_type" field. Evaluate them in THIS priority
order, spending most of your effort on the earlier ones:
  1. game_prop  — single-match bets (match winner/draw, total goals, both teams
                  to score). Usually the SOFTEST/least efficient — look hardest here.
  2. player_prop — player to score, top scorer, etc. Also often soft.
  3. future     — tournament winner, advancement, group winner. Most efficient;
                  least likely to be mispriced.
Game and player props are where mispricing usually hides, so prioritize them.

Your job: find genuinely good value bets — outcomes where the TRUE
probability of winning is meaningfully higher than what Polymarket is pricing
(i.e. the true win chance is higher than the price/ask you'd pay).

There are TWO ways to find value — use BOTH:

A) PRICE MISMATCH vs sportsbooks:
   Search for consensus sportsbook odds on the same outcome. Convert to implied
   probability, then DE-VIG (divide each by the sum so they total 100%). If the
   de-vigged fair prob beats the Polymarket ask by >= {min_edge_pp}pp, that's a
   candidate. Polymarket is low-vig so books need to clearly disagree.

B) ANALYTICAL VALUE (original research):
   Use web search to find recent (last 24-48h) information that the market may
   not have fully priced in: injuries to key opponents, strong recent form,
   favorable bracket draws, confirmed star player returning, weather advantages,
   historical head-to-head, tournament momentum. If your analysis gives a team
   a meaningfully higher TRUE probability than Polymarket implies — even if
   sportsbooks agree with Polymarket — that is still a value bet worth flagging.
   Be rigorous: build a genuine probability estimate, don't just assert one.

For BOTH methods:
- Search recent news to confirm nothing has changed (injuries, suspensions,
  results) that would explain or invalidate the edge.
- Compute edge_pp = (your_fair_prob - polymarket_ask) * 100.
- Flag any outcome with edge_pp >= {min_edge_pp}. Use the confidence field to
  communicate how strong it is — do NOT silently discard a real numerical edge
  just because sportsbooks roughly agree; instead flag it as Low confidence and
  explain the risk in case_against. The user wants to SEE the candidates and
  decide; your job is to surface and grade them, not to gatekeep to zero.
- Confidence guide:
    High   = strong edge, sources agree, news is clean, liquid market.
    Medium = decent edge with a solid thesis but some uncertainty.
    Low    = a real {min_edge_pp}pp+ edge exists but it's speculative (books
             roughly agree, thin liquidity, or news is murky). Still flag it.
- Aim to surface roughly the 2-4 best opportunities per run when they exist,
  ranked by edge_pp (highest first). It is fine to return more or fewer.
- Never INVENT an edge or fabricate numbers — if the math doesn't show
  {min_edge_pp}pp, it doesn't get flagged. Honesty over hitting a count.

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


# Per-1M-token pricing (USD), keyed by a substring of the model id. Update if
# Anthropic changes prices. Web search is billed separately, per request.
MODEL_PRICING = {
    "haiku":  {"in": 1.00,  "out": 5.00},   # Claude Haiku 4.5
    "sonnet": {"in": 3.00,  "out": 15.00},  # Claude Sonnet 4.6
    "opus":   {"in": 5.00,  "out": 25.00},  # Claude Opus 4.x
}
WEB_SEARCH_COST = 0.01  # ~$10 per 1,000 web_search requests


def estimate_cost(model: str, usage) -> dict:
    """Estimate run cost in USD from the response's usage object."""
    price = next((v for k, v in MODEL_PRICING.items() if k in model), None)
    in_tok = getattr(usage, "input_tokens", 0) or 0
    out_tok = getattr(usage, "output_tokens", 0) or 0
    # Web search count lives on usage.server_tool_use.web_search_requests.
    stu = getattr(usage, "server_tool_use", None)
    searches = getattr(stu, "web_search_requests", 0) or 0 if stu else 0

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
        tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 12}],
        messages=[{"role": "user", "content": user}],
    )

    # Log estimated cost for this run.
    cost = estimate_cost(model, resp.usage)
    print(
        f"Cost: ${cost['total_cost']:.4f} "
        f"(tokens ${cost['token_cost']:.4f} [{cost['input_tokens']} in / {cost['output_tokens']} out], "
        f"{cost['web_searches']} web searches ${cost['search_cost']:.4f})"
    )

    # Concatenate text blocks from the final assistant turn.
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
            candidates.append(text[start : end + 1])
    for c in reversed(candidates):
        try:
            return json.loads(c)
        except json.JSONDecodeError:
            continue
    raise ValueError(f"Could not parse JSON from model output:\n{text[:800]}")


# ---------------------------------------------------------------------------
# 3. Bet tracking — append flagged bets to bets_log.jsonl in the repo
# ---------------------------------------------------------------------------
def log_bets(result: dict, run_at: str) -> None:
    log_path = os.path.join(os.path.dirname(__file__), "bets_log.jsonl")
    flagged = result.get("flagged", [])
    if not flagged:
        return
    with open(log_path, "a") as f:
        for bet in flagged:
            record = {"run_at": run_at, **bet}
            f.write(json.dumps(record) + "\n")
    print(f"Logged {len(flagged)} bet(s) to bets_log.jsonl")


def commit_log() -> None:
    """Commit bets_log.jsonl back to the repo so it persists across runs.

    No-op when no log file exists yet (i.e. no bet has ever been flagged),
    which avoids the harmless-but-noisy 'pathspec did not match' git error.
    """
    repo = os.path.dirname(__file__)
    log_path = os.path.join(repo, "bets_log.jsonl")
    if not os.path.exists(log_path):
        return
    os.system(f'cd "{repo}" && git config user.email "scout@poly" && git config user.name "Poly"')
    os.system(
        f'cd "{repo}" && git add bets_log.jsonl '
        f'&& (git diff --cached --quiet || (git commit -m "chore: log flagged bets" && git push))'
    )


# ---------------------------------------------------------------------------
# 4. Discord embed alert
# ---------------------------------------------------------------------------
CONF_COLOR = {"High": 0x2ECC71, "Medium": 0xF1C40F, "Low": 0xE67E22}  # green / yellow / orange


def _fmt_near_misses(near_misses) -> str:
    """Turn the near_misses list into a readable bulleted string."""
    if not near_misses:
        return "None noted this run."
    if isinstance(near_misses, str):
        near_misses = [near_misses]
    return "\n".join(f"• {str(nm)}" for nm in near_misses)[:1024]


def build_discord_payload(result: dict) -> dict:
    flagged = result.get("flagged", [])
    summary = result.get("summary", "No qualifying bets this run.")
    near_misses = result.get("near_misses", [])
    n_eval = result.get("markets_evaluated")
    scanned = f" ({n_eval} markets scanned)" if n_eval else ""

    if not flagged:
        return {
            "username": "Poly",
            "embeds": [{
                "title": "No value bets this run",
                "description": summary[:4096],
                "color": 0x95A5A6,
                "fields": [
                    {"name": "Closest calls (near misses)",
                     "value": _fmt_near_misses(near_misses), "inline": False},
                ],
                "footer": {"text": f"Poly · strict by design{scanned}"},
            }],
        }

    embeds = []
    for bet in flagged:
        conf = bet.get("confidence", "Low")
        color = CONF_COLOR.get(conf, 0x95A5A6)
        poly_pct = round(bet.get("polymarket_ask_pct", 0))
        fair_pct = round(bet.get("fair_pct", 0))
        edge = round(bet.get("edge_pp", 0), 1)
        # Discord rejects empty field values and truncates at 1024 chars.
        case_for = (bet.get("case_for") or "n/a")[:1024]
        case_against = (bet.get("case_against") or "n/a")[:1024]
        # Render sources as a bulleted list (Discord caps a field at 1024 chars).
        srcs = bet.get("sources") or []
        if isinstance(srcs, str):
            srcs = [srcs]
        sources_txt = ("\n".join(f"• {s}" for s in srcs) or "n/a")[:1024]
        fields = [
            {"name": "Polymarket ask", "value": f"{poly_pct}%", "inline": True},
            {"name": "Fair value", "value": f"{fair_pct}%", "inline": True},
            {"name": "Edge", "value": f"+{edge}pp", "inline": True},
            {"name": "Confidence", "value": conf, "inline": True},
            {"name": "Stake guide",
             "value": ("Larger" if conf == "High" else "Moderate" if conf == "Medium" else "Small / pass"),
             "inline": True},
            {"name": "Case for it", "value": case_for, "inline": False},
            {"name": "Risks / case against", "value": case_against, "inline": False},
            {"name": "Sources", "value": sources_txt, "inline": False},
        ]
        embeds.append({
            "title": f"VALUE BET: {bet.get('outcome')}"[:256],
            "description": (bet.get("market") or "")[:4096],
            "color": color,
            "fields": fields,
            "footer": {"text": "Confirm live price in your Polymarket US app before betting. Not financial advice."},
        })

    if len(embeds) > 1:
        embeds[0]["description"] += f"\n\n**{len(embeds)} bets flagged this run — see all cards below.**"

    return {"username": "Poly", "embeds": embeds[:10]}  # Discord max 10 embeds


def send_alert(result: dict) -> None:
    """Send the alert via whichever channel is configured (free options first)."""
    if os.environ.get("DISCORD_WEBHOOK_URL"):
        payload = build_discord_payload(result)
        r = requests.post(os.environ["DISCORD_WEBHOOK_URL"], json=payload, timeout=30)
        r.raise_for_status()
        print("Alert sent via Discord webhook.")
        return

    # Fallback: plain-text channels
    flagged = result.get("flagged", [])
    summary = result.get("summary", "")
    if not flagged:
        body = f"Poly: No qualifying bets this run. {summary}"
    else:
        top = flagged[0]
        extra = f" (+{len(flagged) - 1} more)" if len(flagged) > 1 else ""
        body = (
            f"Poly value alert: {top['outcome']} — {top['market']}. "
            f"Poly {round(top['polymarket_ask_pct'])}% vs fair {round(top['fair_pct'])}% "
            f"(+{round(top['edge_pp'], 1)}pp, {top['confidence']} conf){extra}. "
            f"Confirm price in your Polymarket US app. Not financial advice."
        )[:600]

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
    min_edge_pp = float(os.environ.get("MIN_EDGE_PP", "3"))
    min_liquidity = float(os.environ.get("MIN_LIQUIDITY", "500"))
    max_spread = float(os.environ.get("MAX_SPREAD", "0.07"))
    model = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
    dry_run = os.environ.get("DRY_RUN") == "1"

    now = datetime.now(timezone.utc).isoformat()
    print(f"[{now}] Poly run | tag={tag} min_edge={min_edge_pp}pp model={model}")

    events = fetch_events(tag)
    markets = clean_markets(events, min_liquidity, max_spread)
    print(f"Liquid markets to evaluate: {len(markets)}")
    if not markets:
        print("No liquid markets found this run. Exiting quietly.")
        return 0

    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    result = run_analysis(client, model, markets, min_edge_pp)
    result["markets_evaluated"] = len(markets)

    print("Summary:", result.get("summary", ""))
    print("Near misses:", result.get("near_misses", []))
    flagged = result.get("flagged", [])
    print(f"Flagged bets: {len(flagged)}")
    print(json.dumps(result, indent=2))

    log_bets(result, now)
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
    except Exception as exc:  # noqa: BLE001 — fail loudly in CI logs, no SMS spam
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
