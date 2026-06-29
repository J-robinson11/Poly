# Moby — Polymarket smart-money sentiment agent (cloud edition)

Scans Polymarket markets 3x/day **in the cloud** and pushes a notification to
your phone showing where the **biggest money** is positioned — i.e. smart-money
sentiment read from the largest holders / P&Ls on each market. Runs on GitHub
Actions, so it works with your laptop off. The default alert channel (Discord)
is **free**.

It is an informational research tool, **not financial advice**, and it never
places bets. Always confirm price/availability in your Polymarket app and bet
responsibly.

## How it works
For each candidate market, Moby:
1. Pulls live markets for a tag (default `world-cup`) from Polymarket's **Gamma API**.
2. Pulls the **top 20 holders per outcome** from Polymarket's **Data API**
   (`/holders`) and weights each position by dollars at risk (shares × price).
3. Asks Claude ("Moby") to read the **sentiment**: which side the big money
   leans, how lopsided it is, the largest individual whales, and a conviction
   rating — plus an optional contrarian flag.
4. Pushes the strongest smart-money signals to Discord.

There is **no** sportsbook de-vig / fair-value math anymore — the signal is the
positioning of the largest holders.

## Files
- `moby.py` — the scan + smart-money + sentiment logic
- `requirements.txt` — Python deps (`anthropic`, `requests`)
- `.github/workflows/moby.yml` — the GitHub Actions schedule
- `.github/workflows/ci.yml` — fast smoke test on every push

## One-time setup (~15 min)

### 1. Keys & alert channel
- **Anthropic API key** — console.anthropic.com → API Keys. Add a few dollars of
  credit (small monthly cost at 3 runs/day).
- **Phone alerts — pick ONE free option:**
  - **Discord (recommended).** Channel → **Edit Channel → Integrations →
    Webhooks → New Webhook → Copy Webhook URL.** Turn on phone notifications.
  - **ntfy.sh (zero signup).** Install the **ntfy** app, pick a topic, subscribe.

### 2. Repo secrets
**Settings → Secrets and variables → Actions → New repository secret.**

| Secret name           | Value                                          |
|-----------------------|------------------------------------------------|
| `ANTHROPIC_API_KEY`   | your Anthropic key                             |
| `DISCORD_WEBHOOK_URL` | the Discord webhook URL (if using Discord)     |
| `NTFY_TOPIC`          | your ntfy topic name (if using ntfy instead)   |

Checked in order: Discord → ntfy → Twilio. (Twilio paid SMS still supported via
`TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_FROM`, `ALERT_TO_PHONE`.)

### 3. Test it
**Actions** tab → **Moby** → **Run workflow**. Watch the log. Add a temporary
`DRY_RUN` = `1` secret to print the alert instead of sending it.

## Schedule / timezone
The cron in `moby.yml` is **UTC**, preset ~1h10m early for US Central drift
(`20 11,16,21 * * *` → roughly 7:20a / 12:20p / 5:20p CT after GitHub's typical
~1h scheduler lag). Adjust the hours for your timezone or after a DST change.

## Tuning (optional)
Set as repo secrets or edit the env block in `moby.yml`:
- `MARKET_TAG` — defaults to `world-cup` (e.g. `nba`, `politics` to expand).
- `MARKET_CAP` — markets analyzed per run (default `40`).
- `FUTURES_SLOTS` — slots reserved for futures vs props (default `6`).
- `MIN_SMART_MONEY_USD` — skip markets with little big-money interest (default `2000`).
- `MIN_LIQUIDITY` / `MAX_SPREAD` — quality filters on which markets to consider.
- `ANTHROPIC_MODEL` — defaults to `claude-haiku-4-5-20251001`.

## Signal log
When Moby flags signals it appends them to `signals_log.jsonl` and commits it
back to the repo — a timestamped history of every smart-money call for later
review.

## Notes & limits
- The `/holders` data reflects Polymarket's global catalog; the US-regulated app
  can differ, so confirm in your app before betting.
- "Smart money" = largest current positions. Big holders are often sharp, but not
  always right — treat the contrarian flag seriously.
- GitHub disables scheduled workflows after ~60 days of **no repo activity** —
  commit occasionally or trigger a manual run to keep it alive.
- Cost: Anthropic API usage only (Discord/ntfy are free). The Polymarket APIs are
  free and need no key.
