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
  MARKET_TAG            - default "fifa-world-cup" (incl. live match markets)
  MARKET_CAP            - default 50 (markets analyzed per run)
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
from datetime import datetime, timedelta, timezone

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
# Specific single-match market phrases — win first so e.g. "both teams to score"
# isn't miscaught by the broad "to score" player hint.
_GAME_STRONG = (
    "both teams to score", "btts", "total goals", "total corners", "corners",
    "over ", "under ", "exact score", "correct score", "halftime", "half-time",
    "1st half", "first half", "second half", "clean sheet", "first team to score",
    "to win the match", "end in a draw", "double chance", "handicap",
    "to score first", "winning margin", "anytime team",
)
# Player-prop signals — checked against question AND event (events are titled
# e.g. "Brazil vs. Japan - Player Props").
_PLAYER_HINTS = (
    "to score", "goalscorer", "goal scorer", "golden boot", "top scorer",
    "hat trick", "hat-trick", "assist", "player prop", "anytime scorer",
    "first goal", "to be carded", "to be booked", "shots on target",
    "player to", " goals", "brace",
)
_FUTURE_HINTS = (
    "win the world cup", "to win the tournament", "champion", "winner",
    "to reach", "reach the", "advance", "quarterfinal", "semifinal",
    "semi-final", "quarter-final", "to make the final", "reach final",
    "win group", "group winner", "to qualify", "round of 16", "round of 32",
    "golden glove", "golden ball", "furthest advancing",
)


def classify_market(question: str, event: str) -> tuple:
    """Return (priority, label). Lower priority is scanned/ranked first.

    Order: specific match-market phrase -> player signal -> bare 'X vs Y'
    moneyline -> tournament/futures -> other.
    """
    q = question.lower()
    ev = event.lower()
    text = f"{q} {ev}"
    if any(h in text for h in _GAME_STRONG):
        return (0, "game_prop")
    if any(h in text for h in _PLAYER_HINTS):
        return (1, "player_prop")
    if " vs " in ev or " vs." in ev or " v " in ev or " vs " in q:
        return (0, "game_prop")  # bare match moneyline / per-match market
    if any(h in text for h in _FUTURE_HINTS):
        return (2, "future")
    return (3, "other")


def run_slot_label(now_utc: datetime = None) -> str:
    """Label the run by its nearest scheduled Central slot, e.g. '5:00 PM'.

    The cron fires ~7:20a / 12:20p / 5:20p CT (with GitHub drift); we snap to the
    clean 7AM / 12PM / 5PM slot. World Cup is summer, so Central = UTC-5 (CDT).
    """
    now_utc = now_utc or datetime.now(timezone.utc)
    ct = now_utc - timedelta(hours=5)
    hr = ct.hour + ct.minute / 60.0
    nearest = min((7, 12, 17), key=lambda s: abs(s - hr))
    ampm = "AM" if nearest < 12 else "PM"
    h12 = nearest if nearest <= 12 else nearest - 12
    return f"{h12}:00 {ampm}"


def prune_low_upside(result: dict, max_price: float = 0.90, min_price: float = 0.05) -> int:
    """Drop picks with no real upside (priced >= max_price, e.g. a 100¢ lock) or
    pure longshots (<= min_price). Hard backstop so a 0%-payout pick never ships."""
    picks = result.get("picks", {}) or {}
    removed = 0
    for b in BUCKETS:
        kept = []
        for p in picks.get(b, []) or []:
            try:
                pr = float(p.get("price"))
            except (TypeError, ValueError):
                pr = None
            if pr is not None and (pr >= max_price or pr <= min_price):
                removed += 1
                continue
            kept.append(p)
        picks[b] = kept
    result["picks"] = picks
    return removed


def payouts_for(outcomes: list, prices: list) -> dict:
    """For each outcome, the back price and the upside if it wins.

    profit_pct = (1 - price) / price * 100  → return per $1 staked.
    A 0.90 favorite returns ~11%; a 0.40 pick returns ~150%.
    """
    out = {}
    for name, p in zip(outcomes, prices):
        if p and p > 0:
            out[name] = {
                "price": round(p, 3),
                "profit_pct": round((1 - p) / p * 100),
                "multiple": round(1 / p, 2),
            }
    return out


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
                    "payout_by_outcome": payouts_for(outcomes, prices),
                    "volume24hr": _to_float(m.get("volume24hr")),
                    "liquidity": liquidity,
                }
            )

    cap = int(os.environ.get("MARKET_CAP", "50"))
    futures_slots = int(os.environ.get("FUTURES_SLOTS", "4"))
    futures_slots = max(0, min(futures_slots, cap))

    # --- Time window: prioritize the games that are about to happen, drop the
    # ones already (nearly) over. Each run focuses on its own slice of the day. ---
    now_ts = datetime.now(timezone.utc).timestamp()
    grace = float(os.environ.get("LIVE_GRACE_MIN", "75")) * 60      # drop games kicked off > this ago
    window = float(os.environ.get("WINDOW_HOURS", "12")) * 3600     # "soon" = within this many hours
    for m in cleaned:
        m["_ttk"] = (m["_ts"] - now_ts) if m["_ts"] != _FAR_FUTURE else None  # seconds to kickoff

    def prop_key(m):
        """Tier 0 = upcoming soon, 1 = live (just started), 2 = far upcoming,
        3 = undated. Within a tier: game props before player, soonest first."""
        ttk = m["_ttk"]
        if ttk is None:
            return (3, m["_priority"], 0)
        if ttk >= 0:
            tier = 0 if ttk <= window else 2
            return (tier, m["_priority"], ttk)
        return (1, m["_priority"], -ttk)  # in-play, recently kicked off

    # Exclude finished/late props (kicked off more than `grace` ago).
    props = [m for m in cleaned
             if m["_priority"] in (0, 1, 3)
             and not (m["_ttk"] is not None and m["_ttk"] < -grace)]
    futures = [m for m in cleaned if m["_priority"] == 2]
    props.sort(key=prop_key)
    futures.sort(key=lambda x: (x["_ts"], -x["volume24hr"]))

    props_slots = cap - futures_slots
    selected = props[:props_slots] + futures[:futures_slots]
    if len(selected) < cap:
        chosen = {id(m) for m in selected}
        leftovers = [m for m in props[props_slots:] + futures[futures_slots:]
                     if id(m) not in chosen]
        selected += leftovers[: cap - len(selected)]

    for m in selected:
        m.pop("_priority", None)
        m.pop("_ts", None)
        m.pop("_ttk", None)
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


def fetch_sharp_traders(limit_per_page: int = 50, pages: int = 2) -> dict:
    """Fetch all-time most-profitable traders (SPORTS + OVERALL) by lifetime PnL.

    Returns {wallet_lower: {"pnl": float, "name": str, "rank": str}} — the set of
    historically 'sharp' wallets we weight more heavily when they show up as
    holders. Best-effort: returns {} on error.
    """
    sharp = {}
    for category in ("SPORTS", "OVERALL"):
        for page in range(pages):
            params = {
                "category": category, "timePeriod": "ALL", "orderBy": "PNL",
                "limit": str(limit_per_page), "offset": str(page * limit_per_page),
            }
            url = f"{DATA_BASE}/v1/leaderboard?{urllib.parse.urlencode(params)}"
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            try:
                with urllib.request.urlopen(req, timeout=20) as resp:
                    rows = json.loads(resp.read().decode("utf-8"))
            except Exception as exc:  # noqa: BLE001
                print(f"  leaderboard fetch failed ({category} p{page}): {exc}")
                continue
            for r in rows or []:
                wallet = (r.get("proxyWallet") or "").lower()
                if not wallet:
                    continue
                pnl = _to_float(r.get("pnl"))
                # Keep the best (highest-PnL) record if seen in multiple lists.
                if wallet not in sharp or pnl > sharp[wallet]["pnl"]:
                    sharp[wallet] = {
                        "pnl": pnl,
                        "name": r.get("userName") or wallet[:8],
                        "rank": r.get("rank", ""),
                    }
    print(f"Sharp traders loaded: {len(sharp)}")
    return sharp


def sharp_weight(pnl) -> float:
    """Map a trader's lifetime PnL to a sharpness multiplier."""
    if pnl is None:
        return 1.0          # not a known sharp — count at face value
    if pnl >= 1_000_000:
        return 4.0
    if pnl >= 250_000:
        return 3.0
    if pnl >= 50_000:
        return 2.0
    return 1.5              # on the leaderboard but smaller lifetime profit


def summarize_holders(market: dict, holders_raw: list, sharp: dict = None) -> dict:
    """Boil raw holder lists into a smart-money + sharp-money summary.

    - Dollar exposure ≈ shares * outcome price (money actually at risk).
    - 'Sharp' money additionally weights each holder by their lifetime PnL, so a
      historically profitable whale counts more than a merely-large position.
    """
    sharp = sharp or {}
    outcomes = market["outcomes"]
    prices = market["implied_prob"]

    def price_for(idx):
        return prices[idx] if isinstance(idx, int) and 0 <= idx < len(prices) else 0.0

    def name_for(idx):
        return outcomes[idx] if isinstance(idx, int) and 0 <= idx < len(outcomes) else f"outcome {idx}"

    value_by_outcome = {}        # raw USD exposure
    sharp_by_outcome = {}        # PnL-weighted USD exposure
    all_holders = []             # (usd, idx, name)
    notable_sharps = []          # known-sharp holders with lifetime PnL
    for group in holders_raw or []:
        for h in group.get("holders", []) or []:
            idx = h.get("outcomeIndex")
            amt = _to_float(h.get("amount"))
            usd = amt * price_for(idx)
            side = name_for(idx)
            wallet = (h.get("proxyWallet") or "").lower()
            info = sharp.get(wallet)
            pnl = info["pnl"] if info else None
            weight = sharp_weight(pnl) if info else 1.0
            display = (
                (info["name"] if info else None) or h.get("name") or h.get("pseudonym")
                or (wallet[:8] if wallet else "anon")
            )
            all_holders.append((usd, idx, display))
            value_by_outcome[side] = value_by_outcome.get(side, 0.0) + usd
            sharp_by_outcome[side] = sharp_by_outcome.get(side, 0.0) + usd * weight
            if info:
                notable_sharps.append({
                    "name": display, "side": side,
                    "position_usd": round(usd, 0), "lifetime_pnl": round(pnl, 0),
                })

    value_by_outcome = {k: round(v, 0) for k, v in value_by_outcome.items()}
    sharp_by_outcome = {k: round(v, 0) for k, v in sharp_by_outcome.items()}
    total = sum(value_by_outcome.values())
    sharp_total = sum(sharp_by_outcome.values())

    lean_side = max(value_by_outcome, key=value_by_outcome.get) if value_by_outcome else None
    lean_pct = round(100 * value_by_outcome.get(lean_side, 0) / total, 1) if total else 0.0
    sharp_lean = max(sharp_by_outcome, key=sharp_by_outcome.get) if sharp_by_outcome else None
    sharp_pct = round(100 * sharp_by_outcome.get(sharp_lean, 0) / sharp_total, 1) if sharp_total else 0.0

    top = sorted(all_holders, key=lambda t: t[0], reverse=True)[:5]
    top_holders = [{"name": n, "side": name_for(i), "usd": round(v, 0)} for v, i, n in top]
    notable_sharps.sort(key=lambda x: x["lifetime_pnl"], reverse=True)

    return {
        "smart_money_usd_by_outcome": value_by_outcome,
        "total_smart_money_usd": round(total, 0),
        "lean_side": lean_side,
        "lean_pct": lean_pct,                       # % of raw big money on lean_side
        "sharp_money_by_outcome": sharp_by_outcome,
        "sharp_lean_side": sharp_lean,              # where the HISTORICALLY SHARP money leans
        "sharp_lean_pct": sharp_pct,
        "sharp_traders_present": len(notable_sharps),
        "notable_sharps": notable_sharps[:5],       # name, side, position, lifetime PnL
        "top_holders": top_holders,
        "holders_counted": len(all_holders),
    }


def attach_smart_money(markets: list, min_smart_usd: float) -> list:
    """Fetch + attach holder summaries; drop markets with little big money.

    Keeps condition_id on each market (used later for logging + grading); it is
    stripped from the model's view in run_analysis.
    """
    sharp = fetch_sharp_traders()
    kept = []
    for m in markets:
        summary = summarize_holders(m, fetch_holders(m["condition_id"]), sharp)
        if summary["total_smart_money_usd"] < min_smart_usd:
            continue
        m["smart_money"] = summary
        kept.append(m)
    # Match-level markets FIRST (game props, then player props), futures last —
    # the user wants today's games prioritized, not the giant tournament futures.
    # Within each type, strongest sharp signal first.
    type_rank = {"game_prop": 0, "player_prop": 1, "other": 2, "future": 3}
    kept.sort(
        key=lambda x: (
            type_rank.get(x.get("market_type"), 2),
            -x["smart_money"]["sharp_traders_present"],
            -x["smart_money"]["sharp_lean_pct"],
            -x["smart_money"]["total_smart_money_usd"],
        )
    )
    return kept


# ---------------------------------------------------------------------------
# 2b. Extra sentiment factors: track record (graded log) + X feeds (stub)
# ---------------------------------------------------------------------------
def fetch_market_resolution(condition_id: str):
    """Return (closed, winning_outcome_name) for a market, or (False, None)."""
    if not condition_id:
        return (False, None)
    params = {"condition_ids": condition_id}
    url = f"{GAMMA_BASE}/markets?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        m = data[0] if isinstance(data, list) and data else data
        if not isinstance(m, dict) or not m.get("closed"):
            return (False, None)
        outcomes = _as_list(m.get("outcomes"))
        prices = [_to_float(p) for p in _as_list(m.get("outcomePrices"))]
        if outcomes and prices and len(outcomes) == len(prices):
            win_idx = max(range(len(prices)), key=lambda i: prices[i])
            return (True, outcomes[win_idx])
        return (True, None)
    except Exception:  # noqa: BLE001 — grading is best-effort
        return (False, None)


def load_track_record(grade_limit: int = 25) -> dict:
    """Read signals_log.jsonl and grade resolved past picks (best-effort).

    Returns a compact summary used as a sentiment factor: how Moby's prior
    calls have actually resolved, by category, plus a few recent wins.
    """
    log_path = os.path.join(os.path.dirname(__file__), "signals_log.jsonl")
    if not os.path.exists(log_path):
        return {"status": "no_history", "note": "No prior signals logged yet."}

    rows = []
    try:
        with open(log_path) as f:
            for line in f.read().splitlines()[-300:]:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    except Exception:  # noqa: BLE001
        return {"status": "unreadable", "note": "Could not read signal log."}

    graded = {"win": 0, "loss": 0}
    by_cat = {}
    recent_wins = []
    # Grade most-recent first, capped to bound network calls.
    for row in reversed(rows[-grade_limit:]):
        cid = row.get("condition_id")
        pick = row.get("smart_money_side") or row.get("pick")
        cat = row.get("market_type", "other")
        if not cid or not pick:
            continue
        closed, winner = fetch_market_resolution(cid)
        if not closed or winner is None:
            continue
        won = (str(pick).strip().lower() == str(winner).strip().lower())
        graded["win" if won else "loss"] += 1
        c = by_cat.setdefault(cat, {"win": 0, "loss": 0})
        c["win" if won else "loss"] += 1
        if won and len(recent_wins) < 5:
            recent_wins.append(f"{pick} — {row.get('market', '')[:60]}")

    total_graded = graded["win"] + graded["loss"]
    win_rate = round(100 * graded["win"] / total_graded, 1) if total_graded else None
    return {
        "status": "graded" if total_graded else "pending",
        "total_logged": len(rows),
        "graded": total_graded,
        "wins": graded["win"],
        "losses": graded["loss"],
        "win_rate_pct": win_rate,
        "by_category": by_cat,
        "recent_wins": recent_wins,
        "note": (
            f"{graded['win']}/{total_graded} graded picks won "
            f"({win_rate}%)." if total_graded else
            "Picks logged but none resolved yet — track record still building."
        ),
    }


def fetch_x_sentiment(markets: list) -> dict:
    """X/Twitter sentiment input. STUB — wired as a factor slot for later.

    Activates only if X_BEARER_TOKEN is set; even then returns a clearly-labeled
    placeholder until the feed integration is implemented.
    """
    if not os.environ.get("X_BEARER_TOKEN"):
        return {"status": "not_configured",
                "note": "X/Twitter feed not connected yet (planned input)."}
    return {"status": "stub",
            "note": "X_BEARER_TOKEN set but feed parsing not implemented yet."}


# ---------------------------------------------------------------------------
# 3. Claude (Moby) — smart-money sentiment read
# ---------------------------------------------------------------------------
ANALYSIS_INSTRUCTIONS = """\
You are "Moby," a multi-factor sentiment analyst for Polymarket. Your job each
run: produce a DAILY SLATE of World Cup bets, broken into three buckets —
game props, player props, and futures.

You weigh MULTIPLE sentiment factors for each market, in roughly this priority:

1. SMART / SHARP MONEY (primary). Each market has a "smart_money" block built
   from the largest on-chain holders, cross-referenced against Polymarket's
   all-time profit leaderboard:
     - smart_money_usd_by_outcome / total_smart_money_usd / lean_side / lean_pct:
       RAW big-money positioning (biggest dollars right now).
     - sharp_money_by_outcome / sharp_lean_side / sharp_lean_pct: positioning
       WEIGHTED by each holder's lifetime PnL — i.e. where the historically
       PROFITABLE whales lean. This is the higher-quality signal.
     - notable_sharps: the proven-profitable holders in this market, with their
       side, position size, and lifetime PnL.
     - top_holders: the single largest positions regardless of track record.
   Weight SHARP money above raw size: a market where proven winners are
   concentrated on one side is stronger than one that's merely big. When raw
   money and sharp money DISAGREE, trust the sharp side and flag the divergence.

2. NEWS / PUBLIC SENTIMENT (web search). Use web search to check recent (24-48h)
   news, form, injuries, lineups, and public lean for the relevant teams/players.
   Does the news AGREE with the smart money (confirmation) or CONTRADICT it
   (contrarian risk)?

3. X / SOCIAL FEED. Provided in the input as "x_sentiment". It may be a
   placeholder ("not connected yet") — if so, simply note it's unavailable and
   weigh the other factors. Treat it as a factor slot for the future.

4. TRACK RECORD. Provided as "track_record" — how Moby's own prior logged picks
   have actually resolved (win rate overall and by category, recent wins). Use it
   to CALIBRATE confidence: if a category has been hitting, lean into it slightly;
   if it's been missing, be more cautious there. Do not over-fit a tiny sample.

PAYOFF MATTERS — this is important. The user wants bets that actually MAKE MONEY,
not near-certain favorites with trivial upside. Each market includes
"payout_by_outcome" with each side's back price, profit_pct (return per $1 if it
wins), and multiple. Apply these rules:
  - HARD RULE: NEVER output a pick priced >= 0.90 (≤ ~11% upside), and never a
    pick priced 1.0 / 100¢ (already resolved, ZERO upside). These are pointless —
    exclude them entirely, no matter how strong the smart money is. A "100¢ both
    teams to score, High conviction" pick is exactly what NOT to send.
  - Conviction (High/green) is about SIGNAL STRENGTH **and real payout** — it is
    NOT a measure of how certain/expensive the market already is. Do not mark a
    near-resolved favorite "High".
  - AVOID picks whose back price is >= 0.85 (return under ~18%) UNLESS conviction
    is exceptional and the sharp evidence is overwhelming. A 90%-priced favorite
    is usually NOT worth surfacing — there's no real money in it.
  - Also avoid pure longshots priced <= 0.12 unless the sharp money is genuinely
    piling in (these are mostly lottery tickets).
  - The sweet spot is roughly 0.20-0.75 back price: meaningful payout AND a
    realistic chance, where sharp money on that side is the real signal.
  - Prefer the higher-payout pick when two candidates have similar conviction.
  - Always report the pick's price and payout so the user sees the upside.

TIMING — THIS RUN HAS A WINDOW. "run_context" gives the current time (now_utc),
this run's label, and a window. Each market has a "starts" time. Rules:
  - Recommend bets on games that are UPCOMING (kicking off after now) — prioritize
    the SOONEST upcoming matches in this run's window.
  - A game that already kicked off and is late/most-of-the-way through is STALE —
    do NOT recommend it (e.g. don't pick a game that's in the 80th minute). The
    next run will have already moved on; so should you.
  - Do not re-recommend the same game a previous run already covered if it's now
    underway — move to the next slate of games.

MATCH-LEVEL BETS ARE THE PRIORITY. Spend the slate on game props and player
props for upcoming matches. Futures (tournament winner, etc.) are only a GLANCE:
include AT MOST 1 futures pick, and only if it's genuinely exceptional. If
upcoming matches exist, you MUST surface the best game/player props before any
future. Do not fill the slate with futures.

Up to ~3 game props and ~3 player props, plus at most 1 future. Only where the
factors AND the payoff support a pick; otherwise return an empty list for that
bucket. Prefer UPCOMING games (soonest kickoff).

BE CONCISE. This goes to a phone. Each field is a SHORT phrase or ONE sentence —
no paragraphs, no citations, no "<cite>" tags. smart_money ≤ 20 words. news ≤ 20
words. rationale ≤ 1 sentence. contrarian_note ≤ 12 words.

Conviction:
  - High:   smart money heavily lopsided AND news agrees AND (if available) the
            category's track record is decent.
  - Medium: a clear lean with at least one corroborating factor.
  - Low:    mild/mixed signal — list it as a speculative play, labeled Low.

Never invent holders, numbers, or news. Add a contrarian_note whenever the
factors disagree (e.g. big money piled on a favorite the news cuts against).

Output your final answer as a single fenced JSON block and NOTHING after it, in
exactly this schema:

```json
{
  "generated_at": "<ISO8601 UTC>",
  "picks": {
    "game_props": [
      {
        "market": "<exact market question>",
        "pick": "<the outcome you'd back>",
        "conviction": "High|Medium|Low",
        "price": <number 0-1, the back price of your pick>,
        "payout": "<e.g. '2.5x / +150%' — upside if it wins>",
        "kickoff": "<ISO time if known, else ''>",
        "smart_money": "<= 20 words: raw lean + sharp lean (name a notable sharp if any)>",
        "news": "<= 20 words: what recent news says. No citations.>",
        "rationale": "<= 1 sentence combining the factors>",
        "contrarian_note": "<= 12 words on the main risk, or 'none'>"
      }
    ],
    "player_props": [ /* same shape */ ],
    "futures": [ /* same shape */ ]
  },
  "watchlist": ["<1-3 notable markets that just missed and why>"],
  "summary": "<one-line plain-English summary of today's slate>"
}
```
If a bucket has no good play, use an empty list for it. Keep reasoning concise.
End with the JSON block and nothing after it. The JSON block MUST appear.
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


def run_analysis(client: Anthropic, model: str, markets: list,
                 track_record: dict, x_sentiment: dict, run_context: dict) -> dict:
    # Strip internal-only fields from the model's view of each market.
    model_view = [
        {k: v for k, v in m.items() if not k.startswith("_") and k != "condition_id"}
        for m in markets
    ]
    payload = {
        "run_context": run_context,
        "markets": model_view,
        "track_record": track_record,
        "x_sentiment": x_sentiment,
    }
    user = (
        "Here are live Polymarket World Cup markets with their largest-holder "
        "(smart money) summaries, plus your track record and the X-sentiment slot. "
        "Weigh all factors and return today's slate as JSON.\n\n"
        f"{json.dumps(payload, indent=2)}"
    )
    resp = client.messages.create(
        model=model,
        max_tokens=8000,
        system=ANALYSIS_INSTRUCTIONS,
        tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 8}],
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
BUCKETS = ("game_props", "player_props", "futures")
_BUCKET_TO_TYPE = {"game_props": "game_prop", "player_props": "player_prop", "futures": "future"}


def flatten_picks(result: dict) -> list:
    """Flatten result['picks'] into a single list, tagging each with its bucket."""
    picks = result.get("picks", {}) or {}
    out = []
    for bucket in BUCKETS:
        for p in picks.get(bucket, []) or []:
            out.append({**p, "bucket": bucket, "market_type": _BUCKET_TO_TYPE[bucket]})
    return out


def log_signals(result: dict, run_at: str, cid_by_market: dict) -> None:
    """Append each pick to signals_log.jsonl, enriched with condition_id so it
    can be graded (win/loss) on future runs."""
    log_path = os.path.join(os.path.dirname(__file__), "signals_log.jsonl")
    picks = flatten_picks(result)
    if not picks:
        return
    with open(log_path, "a") as f:
        for p in picks:
            cid = cid_by_market.get(p.get("market", ""), "")
            record = {
                "run_at": run_at,
                "market": p.get("market", ""),
                "smart_money_side": p.get("pick", ""),
                "conviction": p.get("conviction", ""),
                "market_type": p.get("market_type", "other"),
                "condition_id": cid,
            }
            f.write(json.dumps(record) + "\n")
    print(f"Logged {len(picks)} pick(s) to signals_log.jsonl")


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


def _clean(text, limit=220) -> str:
    """Strip citation tags / whitespace and truncate for a phone-sized field."""
    s = str(text or "").replace("\n", " ")
    # Drop any <cite ...>...</cite> wrappers the model might emit.
    while "<cite" in s and ">" in s:
        a = s.find("<cite")
        b = s.find(">", a)
        if b == -1:
            break
        s = s[:a] + s[b + 1:]
    s = s.replace("</cite>", "").strip()
    if len(s) > limit:
        s = s[:limit - 1].rstrip() + "…"
    return s or "n/a"


def _fmt_list(items) -> str:
    if not items:
        return "None noted this run."
    if isinstance(items, str):
        items = [items]
    return "\n".join(f"• {_clean(x, 160)}" for x in items[:3])[:1024]


_BUCKET_LABEL = {"game_props": "GAME PROP", "player_props": "PLAYER PROP", "futures": "FUTURE"}


def build_discord_payload(result: dict) -> dict:
    summary = result.get("summary", "No bets this run.")
    watchlist = result.get("watchlist", [])
    tr = result.get("_track_record", {})
    n_eval = result.get("markets_evaluated")
    scanned = f" · {n_eval} markets" if n_eval else ""
    tr_note = f" · {tr.get('note')}" if tr.get("note") else ""
    slot = result.get("_run_slot") or run_slot_label()

    picks = flatten_picks(result)
    if not picks:
        return {
            "username": "Moby",
            "embeds": [{
                "title": f"🐋 Moby — {slot} run · no bets",
                "description": _clean(summary, 400),
                "color": 0x95A5A6,
                "fields": [{"name": "Watchlist", "value": _fmt_list(watchlist), "inline": False}],
                "footer": {"text": f"Moby · multi-factor sentiment{scanned}{tr_note}"},
            }],
        }

    # Header embed summarizing the slate, then one card per pick (game→player→future).
    counts = {b: 0 for b in BUCKETS}
    for p in picks:
        counts[p["bucket"]] += 1
    header_lines = ", ".join(
        f"{counts[b]} {_BUCKET_LABEL[b].lower()}{'s' if counts[b] != 1 else ''}" for b in BUCKETS
    )
    embeds = [{
        "username": "Moby",
        "title": f"🐋 Moby — {slot} run",
        "description": f"{_clean(summary, 280)}\n\n**{header_lines}**",
        "color": 0x3498DB,
        "footer": {"text": f"Match-level first · futures = glance{scanned}{tr_note}"},
    }]

    for p in picks:
        conf = p.get("conviction", "Low")
        color = CONF_COLOR.get(conf, 0x95A5A6)
        price = p.get("price")
        price_str = f"{round(_to_float(price) * 100)}¢" if price not in (None, "") else "—"
        payout_str = _clean(p.get("payout"), 40)
        # Compact: 3 inline stats + one short "why" + one short "risk".
        fields = [
            {"name": "Conviction", "value": conf, "inline": True},
            {"name": "Price", "value": price_str, "inline": True},
            {"name": "Payout", "value": payout_str, "inline": True},
            {"name": "Why", "value": _clean(p.get("rationale") or p.get("smart_money"), 280), "inline": False},
        ]
        risk = _clean(p.get("contrarian_note"), 140)
        if risk and risk.lower() not in ("none", "n/a"):
            fields.append({"name": "Risk", "value": risk, "inline": False})
        title = f"[{_BUCKET_LABEL[p['bucket']]}] {p.get('pick')} · {price_str}"
        embeds.append({
            "title": title[:256],
            "description": _clean(p.get("market"), 200),
            "color": color,
            "fields": fields,
            "footer": {"text": "Sentiment read, not advice."},
        })

    return {"username": "Moby", "embeds": embeds[:10]}  # Discord max 10 embeds


def send_alert(result: dict) -> None:
    """Send the alert via whichever channel is configured (free options first)."""
    if os.environ.get("DISCORD_WEBHOOK_URL"):
        payload = build_discord_payload(result)
        r = requests.post(os.environ["DISCORD_WEBHOOK_URL"], json=payload, timeout=30)
        r.raise_for_status()
        print("Alert sent via Discord webhook.")
        return

    picks = flatten_picks(result)
    summary = result.get("summary", "")
    if not picks:
        body = f"Moby: no bets on today's slate. {summary}"
    else:
        top = picks[0]
        extra = f" (+{len(picks) - 1} more)" if len(picks) > 1 else ""
        body = (
            f"Moby slate: [{_BUCKET_LABEL.get(top['bucket'], '')}] {top.get('pick')} — "
            f"{top.get('market')} ({top.get('conviction')} conviction){extra}. "
            f"Not financial advice."
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
    tag = os.environ.get("MARKET_TAG", "fifa-world-cup")
    min_liquidity = float(os.environ.get("MIN_LIQUIDITY", "500"))
    max_spread = float(os.environ.get("MAX_SPREAD", "0.07"))
    min_smart_usd = float(os.environ.get("MIN_SMART_MONEY_USD", "2000"))
    model = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
    dry_run = os.environ.get("DRY_RUN") == "1"

    now_dt = datetime.now(timezone.utc)
    now = now_dt.isoformat()
    slot = run_slot_label(now_dt)
    window_hours = float(os.environ.get("WINDOW_HOURS", "12"))
    print(f"[{now}] Moby run | slot={slot} tag={tag} model={model}")

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

    # Build market -> condition_id map for logging/grading before the model view.
    cid_by_market = {m["question"]: m.get("condition_id", "") for m in markets}

    # Extra sentiment factors.
    track_record = load_track_record()
    x_sentiment = fetch_x_sentiment(markets)
    print("Track record:", track_record.get("note", ""))
    print("X sentiment:", x_sentiment.get("note", ""))

    run_context = {
        "now_utc": now,
        "run_label": f"{slot} CT",
        "window_hours": window_hours,
        "note": ("Prioritize games kicking off after now_utc, soonest first. "
                 "Skip games already late/most-of-the-way through."),
    }

    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    result = run_analysis(client, model, markets, track_record, x_sentiment, run_context)
    result["markets_evaluated"] = len(markets)
    result["_track_record"] = track_record
    result["_run_slot"] = slot

    # Hard backstop: drop zero/low-upside picks (e.g. a 100¢ lock) the model
    # shouldn't have surfaced, regardless of how it labeled them.
    max_price = float(os.environ.get("MAX_PICK_PRICE", "0.90"))
    pruned = prune_low_upside(result, max_price=max_price)
    if pruned:
        print(f"Pruned {pruned} low/zero-upside pick(s) priced >= {max_price} or <= 0.05.")

    print("Summary:", result.get("summary", ""))
    print("Watchlist:", result.get("watchlist", []))
    picks = flatten_picks(result)
    print(f"Picks: {len(picks)} "
          f"({', '.join(b + '=' + str(sum(1 for p in picks if p['bucket'] == b)) for b in BUCKETS)})")
    print(json.dumps(result, indent=2))

    log_signals(result, now, cid_by_market)
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
