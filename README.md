# BOUBA's Football Engine

A self-updating football prediction and value-betting system covering
**25 leagues** (22 European + Russia, Japan and China), **all UEFA club competitions** (Champions League,
Europa League, Conference League) and **national teams** (UEFA Nations
League, Africa Cup of Nations, World Cup, Euros, Copa America, qualifiers). It downloads results, shots and bookmaker odds
automatically, rates every team with a Dixon-Coles Poisson model, and tells
you match by match whether to bet, how much, and the most likely scores.

No API keys, no manual downloads, no data entry.

---

## Fully automatic mode (recommended): GitHub + Streamlit

1. Upload **all** files in this folder to your GitHub repository, including
   the `.github` folder (it holds `workflows/daily.yml`).
2. Deploy the repository on share.streamlit.io (main file
   `epl_value_dashboard.py`) to get your dashboard link.
3. In the repository's **Actions** tab, open **Daily football predictions** and
   click **Run workflow** once to test it.

GitHub then runs the system four times a day (06:00, 10:00, 14:00 and 18:00
Douala time) with no computer needed, and saves these in the repository:

- **predictions.md** - open it on GitHub (or the GitHub phone app): the week's
  bets at a glance, then every match of every competition day by day with its
  advice, predicted score and a before-you-bet checklist (times in Douala time;
  add `--timezone Africa/Lagos` etc. to the run line in daily.yml to change it).
- **paper_bets.csv** - the permanent record of every advised bet, settled
  automatically with profit and closing-line value. The dashboard reads it.
- **data_snapshot/** - a backup copy of every data source. The app starts in
  seconds from it, and keeps working from it if a source goes down.

A second workflow (**Self-tests**) runs automatically after every change and
checks the whole system offline on simulated data. A red cross on a commit
means something broke; GitHub emails you about failed runs.

To change the bankroll used for stakes in predictions.md, edit the line
`run: python epl_value_dashboard.py --log` in `.github/workflows/daily.yml`
to e.g. `run: python epl_value_dashboard.py --log --bankroll 50000`.

---

## Coverage and which leagues get BET advice

The same model was replayed week by week on past seasons in every league,
betting every 3-20% edge at Betfair Exchange prices. It only made money
consistently in the Premier League, so that is the only league with BET advice.

| Status | Leagues | What you get |
|---|---|---|
| ✅ Bet | England Premier League (+16.7% in 2025/26, +18.8% in 2024/25) | BET / NO BET advice with stakes |
| 👀 Watch | England Championship (about break-even) | Possible bets logged for paper trading only |
| 📊 Predictions only | Other 23 leagues: England L1, L2, National League; Scotland x4; Germany x2; Italy x2; Spain x2; France x2; Netherlands; Belgium; Portugal; Turkey (Süper Lig); Greece; Russia (Premier League); Japan (J1 League); China (Super League) | Predicted scores and probabilities, never BET advice (the model lost money there in testing) |
| 🏆 UEFA club cups | Champions League, Europa League, Conference League | Predictions and the minimum odds worth taking ("back X only at 1.85+"); no free odds feed, so never BET advice |
| 🌍 National teams | Fixtures listed automatically for the UEFA Nations League and, as soon as they are published, the Africa Cup of Nations, World Cup, Euros and Copa America finals; any two national teams on demand (use this for AFCON / World Cup qualifiers) | Predictions and the minimum odds worth taking |

A sidebar switch can show BET advice in every league, but this is not
recommended.

## Ask about any match

At the top of the app, type a **home team** and an **away team** - any club in
the 22 leagues or the UEFA cups, or any national team - and you get advice for
that exact match: BET / WATCH / NO BET with stake when the match is in the
odds feed, otherwise the predicted score, probabilities and the minimum odds
worth taking. Common spellings work ("Man Utd", "PSG", "Inter Milan", "BVB",
"Côte d'Ivoire", "Cameroun"); for typos it asks "did you mean...?" instead of
guessing.

## Robustness built in

- **Backup copy + automatic fallback** for every data source.
- **🩺 System Health tab** (and a sidebar indicator): status and freshness of
  every source, backup age, and what to do if something shows ❌.
- **Safety switch:** a BET league is automatically downgraded to WATCH once
  150+ paper bets show it is not beating the closing odds (average CLV below
  zero) or its ROI is clearly negative. WATCH leagues whose paper results turn
  clearly positive are flagged for your decision (never promoted automatically).
- **Retries and sanity checks** on every download (impossible scores, future
  dates and duplicates are dropped), and each tab and league is isolated so one
  failure never takes down the rest of the app.
- **Self-tests** on every change.

## What each tab does

| Tab | Purpose |
|---|---|
| 🗓️ This Week: Everything | Pick a day: its bet slip, then every match that day from all 22 leagues, UEFA cups and national-team tournaments, with a before-you-bet checklist: BET / WATCH / NO BET, stake, predicted score, top 5 scores, win/draw/loss %. |
| 📋 Value Table | The selected league's fixtures ranked by edge, plus "What to do — match by match": every competition, day by day. |
| 🔍 Match Analyzer | One match in depth, with What-If sliders for injuries. |
| 📊 Team Ratings | Attack and defence strength of every team in the selected league. |
| 🧾 Backtest | Replays a past season of the selected league with no look-ahead. |
| 📒 Paper Trading | Every advised bet, settled automatically: profit and closing-line value. |
| 🏆 UEFA Cups | Champions / Europa / Conference League fixtures for the next 14 days with predictions, any two clubs on demand, and how strong each league is in Europe. |
| 🌍 National Teams | Upcoming Nations League / AFCON / World Cup / Euros / Copa America matches, any two national teams on demand, ranking by region (Africa, Europe, ...). |
| 🩺 System Health | Status of every data source, the backup copy and the safety switch. |

## How accurate is it?

| Prediction | Right in testing |
|---|---|
| Club matches: winner (or draw) of the predicted score | ~50% |
| Club matches: one of the top 5 most likely scores | ~50% (top 3: ~33%) |
| 🎯 Best pick (most likely of 17 markets, blended with bookmaker odds; 12 leagues, 8,591 matches, 2 seasons) | 88% overall: 🔒 Banker 92%, ⭐⭐⭐ Strong 90%, ⭐⭐ Good 88%, ⭐ Fair 84% (national teams: Banker 92%) - very short odds, about -4% at bookmaker prices |
| Club matches: exact predicted score | ~11% (about 1 in 9) |
| UEFA club cups: most likely result (all 531 games of 2025/26) | 56.5% |
| National teams: most likely result (1,909 competitive games, 2024-2026) | 61.6% |
| African national teams (423 competitive games) / AFCON 2025 finals (52 games) | 62.9% / 69.2% |

The Premier League profit range in testing still includes zero, so treat the
system as unproven until the Paper Trading tab shows a positive average
closing-line value over at least 100-200 bets.

## Data sources

- Clubs: Football-Data.co.uk (results, shots, xG, Bet365 / Betfair / market odds).
  Usually updated twice a week; fixtures for the next round appear mid-week.
- UEFA competitions and Nations League fixtures/results: fixturedownload.com JSON feeds.
  European-cup strength = domestic league rating + how strong the league or
  country has been in Europe over the last three seasons.
- Russia, Japan and China: Football-Data.co.uk's extra-leagues files (closing
  odds; upcoming fixtures with odds for Japan and China - Russia has no
  fixture feed, so use the match search for Russian games).
- Middle East and Asian national teams (Saudi Arabia, Qatar, Iran, Japan,
  Korea...) are rated from every international result; use the match search or
  "Any two national teams" - the ranking has an Africa / Asia / Middle East filter.
- Not available from any free source: AFCON qualifier fixtures, CAF and AFC
  club competitions, the Saudi Pro League and other Middle East leagues, and
  AFC Asian Cup / Gulf Cup fixtures.
- National teams: the open `international_results` dataset (martj42 on GitHub),
  updated after each international window. No free feed publishes upcoming
  international fixtures or odds, so pick those matches yourself in the app.

## Run it on your own computer instead (optional)

```bash
pip install -r requirements.txt
streamlit run epl_value_dashboard.py
```

On Windows you can double-click `run_dashboard.bat`, and schedule
`daily_log.bat` in Task Scheduler for the background run.

## Disclaimer

Statistical estimates only - no outcome is guaranteed. Bet only what you can
afford to lose.
