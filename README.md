# 🐋 Moby

**A multi-factor smart-money sentiment agent for Polymarket.** Moby runs in the
cloud on a schedule, reads where the **biggest and sharpest money** is positioned
across World Cup markets, blends in news and your own track record, and pushes a
**daily slate of bets** — broken into game props, player props, and futures —
straight to your phone via Discord.

It is an informational research tool. **It is not financial advice and it never
places bets.** Always confirm price and availability in your own Polymarket app,
and bet responsibly.

---

## What makes Moby different

Most tools hunt for "mispricings" against sportsbooks. Moby doesn't. It reads
**sentiment** — specifically, what the people with money and a track record are
actually doing — and weighs several independent factors into each pick.

### The factors

| # | Factor | Source | Weight |
|---|--------|--------|--------|
| 1 | **Smart money** | Largest current holders per outcome (`/holders`), weighted by dollars at risk | Primary |
| 2 | **Sharp money** | Those holders cross-referenced against the all-time profit **leaderboard** (`/v1/leaderboard`) and re-weighted by lifetime PnL | Primary (highest quality) |
| 3 | **News / public sentiment** | Live web search (injuries, form, lineups, momentum) | Secondary |
| 4 | **Track record** | Moby's own past picks, graded win/loss as their markets resolve | Calibration |
| 5 | **X / social feed** | Wired as an input slot — *coming soon* | Planned |

**Sharp money is the headline idea.** A market where one side is merely *big* is
a weak signal. A market where Polymarket's *historically profitable whales* are
piled on one side is a strong one. Moby weights each holder by lifetime PnL
(a $1M+ all-time winner counts 4× a no-name position), so a single proven sharp
can outweigh a crowd of size. When raw money and sharp money **disagree**, Moby
trusts the sharp side and flags the divergence.

### The output: a daily slate

Every run produces up to ~3 picks in each bucket — **game props → player props →
futures** — each with its conviction and the factors behind it. Empty buckets are
allowed: Moby won't invent a pick when the factors don't support one.

---

## How a run works

```
Gamma API ──▶ pull live World Cup markets, classify & prioritize
                (game props → player props → futures; upcoming games first)
                         │
Data API ─────▶ for each market, pull top-20 holders per outcome
   /holders                │
Data API ─────▶ pull all-time profit leaderboard → "sharp" wallet set
   /v1/leaderboard         │
                ┌──────────┴───────────┐
                ▼                      ▼
        raw $ positioning      PnL-weighted (sharp) positioning
                └──────────┬───────────┘
                           ▼
          + news (web search) + track record + X slot
                           ▼
                 Claude ("Moby") synthesizes
                           ▼
        Daily slate → Discord  +  signals_log.jsonl (for grading)
```

---

## Files

| File | Purpose |
|------|---------|
| `moby.py` | The whole agent: data pull, sharp-money weighting, sentiment synthesis, alerts |
| `requirements.txt` | Python deps (`anthropic`, `requests`) |
| `.github/workflows/moby.yml` | The 3×/day schedule + Discord failure alert |
| `.github/workflows/ci.yml` | Fast smoke test on every push (compile, classify, sharp-weighting, Discord payloads) |
| `signals_log.jsonl` | Auto-created: every pick logged with `condition_id` so it can be graded later |

---

## One-time setup (~15 min)

### 1. Keys & alert channel
- **Anthropic API key** — [console.anthropic.com](https://console.anthropic.com) → API Keys. Add a few dollars of credit.
- **Phone alerts — pick ONE free option:**
  - **Discord (recommended):** in a server you own, a channel → **Edit Channel → Integrations → Webhooks → New Webhook → Copy Webhook URL**, then enable notifications for that channel on your phone.
  - **ntfy.sh (no signup):** install the **ntfy** app, pick a topic, subscribe to it.

### 2. Repo secrets
**Settings → Secrets and variables → Actions → New repository secret.**

| Secret | Required | Value |
|--------|----------|-------|
| `ANTHROPIC_API_KEY` | ✅ | Your Anthropic key |
| `DISCORD_WEBHOOK_URL` | one alert channel | Discord webhook URL |
| `NTFY_TOPIC` | (alt) | Your ntfy topic |
| `TWILIO_ACCOUNT_SID` / `TWILIO_AUTH_TOKEN` / `TWILIO_FROM` / `ALERT_TO_PHONE` | (alt, paid) | Twilio SMS |

Alert channels are checked in order: **Discord → ntfy → Twilio.**

### 3. Test it
**Actions** tab → **Moby** → **Run workflow**. Watch the log. Add a temporary
`DRY_RUN` = `1` secret to print the alert instead of sending it.

---

## Tuning (all optional)

Set as repo secrets or edit the `env:` block in `moby.yml`:

| Variable | Default | What it does |
|----------|---------|--------------|
| `MARKET_TAG` | `world-cup` | Which Polymarket tag to scan |
| `MARKET_CAP` | `40` | Markets analyzed per run |
| `FUTURES_SLOTS` | `6` | Slots reserved for futures vs props |
| `MIN_SMART_MONEY_USD` | `2000` | Skip markets with little big-money interest |
| `MIN_LIQUIDITY` / `MAX_SPREAD` | `500` / `0.07` | Quality filters on markets |
| `ANTHROPIC_MODEL` | `claude-haiku-4-5-20251001` | Swap to `claude-sonnet-4-6` for sharper analysis at higher cost |
| `X_BEARER_TOKEN` | — | Reserved for the upcoming X/Twitter sentiment feed |

---

## The track record (self-grading)

Moby logs every pick to `signals_log.jsonl` **with the market's `condition_id`**.
On later runs it re-checks those markets via the Gamma API; once a market
resolves, the pick is graded **win or loss**. That record — overall and per
bucket — is fed back into the model to **calibrate conviction** (lean into
buckets that are hitting, ease off ones that aren't).

It starts empty. Until some logged picks' markets resolve, Moby honestly reports
"track record still building" — that's expected, not a bug.

---

## Schedule / timezone

The cron in `moby.yml` is **UTC** (`20 11,16,21 * * *`), deliberately set ~1h10m
early to absorb GitHub's typical ~1-hour scheduler drift — so alerts land around
**7:20a / 12:20p / 5:20p Central**. GitHub cron ignores Daylight Saving; nudge
the hours for your timezone or after a DST change. Scheduled runs can also drift
under load.

---

## Cost

- **Anthropic API** is the only paid piece — roughly a few cents to ~$0.15 per
  run on Haiku (mostly web-search calls). At 3 runs/day that's a low-double-digit
  monthly cost. Each run logs its exact estimated cost.
- **Polymarket Gamma + Data APIs** are free and need no key.
- **Discord / ntfy** alerts are free.

---

## Notes & limits

- "Sharp" = on Polymarket's all-time profit leaderboard. Proven winners are often
  right, **not always** — take the contrarian flag seriously and size accordingly.
- The `/holders` and leaderboard data reflect Polymarket's **global** catalog;
  the US-regulated app can differ, so confirm in your app before betting.
- GitHub disables scheduled workflows after ~60 days of **no repo activity** —
  commit occasionally or trigger a manual run to keep it alive.
- Moby reads sentiment; it does not guarantee outcomes. Bet responsibly.
