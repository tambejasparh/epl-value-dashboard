# EPL Poisson Value Dashboard

A self-updating Premier League prediction and value-betting dashboard. It
downloads results, shots and bookmaker odds from Football-Data.co.uk once an
hour, rates every team with a Dixon-Coles Poisson model, and flags bets where
the bookmaker's price is higher than the model's fair price. Every flag is
logged and settled automatically, so you can check whether the edge is real.

No API keys, no manual downloads, no data entry.

---

## 1. Install (once)

You need **Python 3.10 or newer** (python.org; on Windows tick "Add Python to PATH").

```bash
cd epl-value-dashboard
pip install -r requirements.txt
```

## 2. Launch

```bash
streamlit run epl_value_dashboard.py
```

It opens at http://localhost:8501. On Windows you can double-click
`run_dashboard.bat` instead (`run_dashboard.sh` on macOS/Linux).

## 3. Automate the paper-trading log (recommended)

Value flags are logged whenever the dashboard is open. To log them even when
it isn't, schedule the headless run once or twice a day:

**Windows (Task Scheduler)**
1. Open Task Scheduler → *Create Basic Task* → name it "EPL value log".
2. Trigger: *Daily*, e.g. 10:00 (add a second task for 17:00 if you like).
3. Action: *Start a program* → browse to `daily_log.bat` in this folder.

**macOS / Linux (cron)** — run `crontab -e` and add:
```
0 10,17 * * * /full/path/to/epl-value-dashboard/daily_log.sh
```

Each run appends a one-line summary to `daily_log.txt`. You can also run it by
hand: `python epl_value_dashboard.py --log` (options: `--book`, `--min-edge`,
`--max-edge`, `--goals-pct`, `--half-life`, `--seasons-back`).

---

## What each tab does

| Tab | Purpose |
|---|---|
| 📋 Master Fixtures Value Table | Every fixture still to kick off in the next 7 days, all 1X2 and O/U 2.5 markets, ranked by edge. Green = value. |
| 🔍 Match Analyzer | One match in depth: xG, outcome / goals / BTTS probabilities, fair odds, likely scorelines. Sidebar What-If sliders apply here. |
| 📊 Team Ratings | Attack and defence strength of every team (1.00 = league average). |
| 🧾 Backtest | Replays a past season week by week with no look-ahead: profit, ROI with 95% range, closing-line value, model-vs-market accuracy, calibration. |
| 📒 Paper Trading | Your logged flags, settled automatically: score, profit, closing-line value (CLV). Download as CSV. |

## Settings (sidebar) and tested defaults

| Setting | Default | Why |
|---|---|---|
| Odds source | Bet365 | Betfair Exchange (5% commission deducted) is the sharpest source still in the feed; Pinnacle data ended in January 2026. |
| Minimum edge | 3% | |
| Maximum believable edge | 20% | Flags above 20% lost ~10% ROI over two seasons — usually team news the model can't see. |
| Weight on actual goals | 50% | Remaining 50% is shot quality (xG when the feed has it, else shots on target). Best in testing. |
| Form half-life | 400 days | Best in testing. |
| Previous seasons | 2 | Needed for stable early-season ratings and for the backtest. |

## Backtest evidence (walk-forward; tuned on 2024/25, tested once on 2025/26)

| 2025/26 (held out) | Old goals-only | Current model | Market |
|---|---|---|---|
| 1X2 log-loss (lower = better) | 1.031 | 1.024 | 1.013 |
| Over/Under 2.5 log-loss | 0.701 | 0.691 | 0.684 |
| ROI vs Betfair Exchange, ≥3% edge | −3.1% | +4.9% (95% range −8.7% to +18.4%) | |

**Read this honestly:** the market is still more accurate overall, and the
profit range includes zero. Treat the system as unproven until the Paper
Trading tab shows a positive average CLV over at least 100–200 bets.

## Data notes

- Football-Data.co.uk usually updates results twice a week. The dashboard
  warns when the newest result is more than 10 days old.
- Fixtures and odds for the coming round usually appear mid-week.
- Odds published: 1X2 and Over/Under 2.5 only. For O/U 1.5 / 3.5 and BTTS the
  app shows fair odds to compare against your bookmaker yourself.

## Deploying elsewhere

- **Your own computer (recommended):** steps 1–3 above. The paper-trading log
  (`paper_bets.csv`) stays in this folder.
- **Streamlit Community Cloud:** push this folder to a GitHub repo, then
  *New app* → pick the repo → main file `epl_value_dashboard.py`. Note: the
  cloud disk is wiped on restarts, so the paper log does not persist there;
  keep the scheduled `--log` run on your own machine for the record.
- To store the log elsewhere, set the environment variable `EPL_PAPER_LOG`
  to a full file path.

## Disclaimer

Statistical estimates only — no outcome is guaranteed. Bet only what you can
afford to lose.
