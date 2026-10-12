"""
===============================================================================
 EPL POISSON VALUE DASHBOARD
===============================================================================
 A self-updating football prediction dashboard for the English Premier League.

 DATA      : Football-Data.co.uk public CSV files (no API key, no downloads).
             - Results  : https://www.football-data.co.uk/mmz4281/<season>/E0.csv
             - Fixtures : https://www.football-data.co.uk/fixtures.csv
             Both are re-fetched automatically at most once per hour.

 MODEL     : Time-weighted Dixon-Coles Poisson model, fitted by maximum
             likelihood on a blended "goal signal":
               signal = w x actual goals + (1 - w) x shot quality
             Shot quality = the match xG when the feed has it (2026/27 on),
             otherwise shots on target x league goals-per-shot-on-target.
             Shots are far less noisy than goals, so ratings settle faster.
             Walk-forward test (tuned on 2024/25, checked on 2025/26):
             w = 50% beat goals-only on 1X2 and Over/Under accuracy.
             log(lambda_home) = mu + home_adv + attack[home] - defence[away]
             log(lambda_away) = mu            + attack[away] - defence[home]
             * Ratings are relative to the league average (exp(rating)=1.00 is
               exactly average), so they are normalised automatically every
               time the league scoring level changes.
             * Older matches are down-weighted exponentially (half-life slider),
               so ratings "roll" with form while previous seasons keep early-
               season estimates stable.
             * A mild L2 penalty shrinks teams with few matches (e.g. newly
               promoted sides) toward average instead of over-reacting.
             * The Dixon-Coles rho term corrects the known Poisson bias on
               0-0, 1-0, 0-1 and 1-1 scorelines.

 BACKTEST  : Walk-forward replay of a past season. The model is refitted
             every week using ONLY matches already played, then "bets" where
             the edge clears your threshold (1 unit flat stake). Reports
             profit, ROI, closing-line value (CLV) and model-vs-market
             log-loss, so you can see if the edge is real before staking.

 VALUE     : Fair odds = 1 / model probability.
             Edge      = model probability x bookmaker odds - 1.
             Bookmaker odds > fair odds  <=>  edge > 0  => "VALUE" (green).

 PAPER     : Every green flag is logged automatically to paper_bets.csv
 TRADING     (next to this file) with the price at the time, then settled
             from the results feed: score, profit and closing-line value.
             Headless logging for a daily scheduled task:
               python epl_value_dashboard.py --log

 RUN       : pip install -r requirements.txt
             streamlit run epl_value_dashboard.py

 NOTE      : This is a statistical tool, not a guarantee. Bookmakers (sharp books
             and exchanges especially) are very efficient; treat small edges with suspicion
             and never stake money you cannot afford to lose.
===============================================================================
"""

import argparse
import datetime as dt
import io
import logging
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
from scipy.optimize import minimize
from scipy.stats import poisson

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
BASE_URL = "https://www.football-data.co.uk"
LEAGUE = "E0"                      # Football-Data code for the Premier League
MAX_GOALS = 10                     # Score matrix runs 0..10 goals per team
SHRINKAGE = 2.0                    # L2 penalty on team ratings (regularisation)
HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
}

# Default settings (used by the sidebar and by the headless --log mode).
# Chosen by walk-forward testing: tuned on 2024/25, confirmed on 2025/26.
DEFAULTS = {
    "book": "Bet365",
    "min_edge": 3.0,        # % edge needed to flag a bet
    "max_edge": 20,         # % edge above which a flag is treated as model error
    "goals_pct": 50,        # % weight on actual goals (rest = shot quality)
    "half_life": 400,       # days
    "seasons_back": 2,      # previous seasons blended in
    "bankroll": 100.0,      # your betting bank (any currency)
    "kelly_fraction": 0.25, # stake = 1/4 Kelly ...
    "max_stake_pct": 2.0,   # ... never more than 2% of bankroll on one bet
}
PAPER_LOG = Path(os.environ.get("EPL_PAPER_LOG",
                                Path(__file__).with_name("paper_bets.csv")))
PREDICTIONS_MD = Path(os.environ.get("EPL_PREDICTIONS",
                                     Path(__file__).with_name("predictions.md")))
UK_TZ = "Europe/London"

# Odds columns available in the Football-Data feed, per bookmaker source.
# (BTTS and O/U 1.5 / 3.5 are NOT published in this feed.)
BOOKMAKERS = {
    "Bet365": {"H": "B365H", "D": "B365D", "A": "B365A",
               "O25": "B365>2.5", "U25": "B365<2.5"},
    "Betfair Exchange (5% comm.)": {"H": "BFEH", "D": "BFED", "A": "BFEA",
                                    "O25": "BFE>2.5", "U25": "BFE<2.5"},
    "Pinnacle (feed ends Jan 2026)": {"H": "PSH", "D": "PSD", "A": "PSA",
                                      "O25": "P>2.5", "U25": "P<2.5"},
    "Market best price": {"H": "MaxH", "D": "MaxD", "A": "MaxA",
                          "O25": "Max>2.5", "U25": "Max<2.5"},
    "Market average": {"H": "AvgH", "D": "AvgD", "A": "AvgA",
                       "O25": "Avg>2.5", "U25": "Avg<2.5"},
}

# Closing odds (last price before kick-off) — only in the results files.
CLOSING_BOOKMAKERS = {
    "Bet365": {"H": "B365CH", "D": "B365CD", "A": "B365CA",
               "O25": "B365C>2.5", "U25": "B365C<2.5"},
    "Betfair Exchange (5% comm.)": {"H": "BFECH", "D": "BFECD", "A": "BFECA",
                                    "O25": "BFEC>2.5", "U25": "BFEC<2.5"},
    "Pinnacle (feed ends Jan 2026)": {"H": "PSCH", "D": "PSCD", "A": "PSCA",
                                      "O25": "PC>2.5", "U25": "PC<2.5"},
    "Market best price": {"H": "MaxCH", "D": "MaxCD", "A": "MaxCA",
                          "O25": "MaxC>2.5", "U25": "MaxC<2.5"},
    "Market average": {"H": "AvgCH", "D": "AvgCD", "A": "AvgCA",
                       "O25": "AvgC>2.5", "U25": "AvgC<2.5"},
}
# Exchange odds are quoted before commission; winnings are cut by this much.
COMMISSION = {"Betfair Exchange (5% comm.)": 0.05}


def net_odds(odds, book: str):
    """Odds after exchange commission (unchanged for normal bookmakers)."""
    c = COMMISSION.get(book, 0.0)
    return 1 + (odds - 1) * (1 - c)


SHOT_COLUMNS = ["HST", "AST", "HxG", "AxG"]   # shots on target, expected goals

ODDS_COLUMNS = sorted({c for d in (BOOKMAKERS, CLOSING_BOOKMAKERS)
                       for book in d.values() for c in book.values()})

MARKET_LABELS = {
    "H": "Home Win", "D": "Draw", "A": "Away Win",
    "O15": "Over 1.5", "U15": "Under 1.5",
    "O25": "Over 2.5", "U25": "Under 2.5",
    "O35": "Over 3.5", "U35": "Under 3.5",
    "BTTS_Y": "BTTS Yes", "BTTS_N": "BTTS No",
}


# =============================================================================
# 1. DATA INFRASTRUCTURE
# =============================================================================
def season_codes(today: dt.date, seasons_back: int) -> list[str]:
    """Return Football-Data season folder codes, newest first (e.g. '2627')."""
    start = today.year if today.month >= 7 else today.year - 1
    return [f"{(start - i) % 100:02d}{(start - i + 1) % 100:02d}"
            for i in range(seasons_back + 1)]


def _fetch_csv(url: str) -> pd.DataFrame:
    """Download a CSV silently. Any failure returns an empty DataFrame."""
    try:
        resp = requests.get(url, headers=HTTP_HEADERS, timeout=20)
        resp.raise_for_status()
        try:
            text = resp.content.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = resp.content.decode("latin-1")
        df = pd.read_csv(io.StringIO(text), on_bad_lines="skip")
        df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
        return df.dropna(how="all")
    except Exception:
        return pd.DataFrame()


def _parse_dates(series: pd.Series) -> pd.Series:
    """Football-Data uses dd/mm/yyyy (older files dd/mm/yy)."""
    s = series.astype(str).str.strip()
    out = pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")
    missing = out.isna()
    if missing.any():
        out[missing] = pd.to_datetime(s[missing], format="%d/%m/%y", errors="coerce")
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def load_results(seasons_back: int) -> pd.DataFrame:
    """Fetch and clean completed EPL results for the current + past seasons."""
    frames = []
    for code in season_codes(dt.date.today(), seasons_back):
        df = _fetch_csv(f"{BASE_URL}/mmz4281/{code}/{LEAGUE}.csv")
        required = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"}
        if df.empty or not required.issubset(df.columns):
            continue  # season file not yet published / unreachable -> skip
        keep = (sorted(required) + [c for c in SHOT_COLUMNS if c in df.columns]
                + [c for c in ODDS_COLUMNS if c in df.columns])
        df = df[keep].copy()
        df["Season"] = code
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df["Date"] = _parse_dates(df["Date"])
    df["FTHG"] = pd.to_numeric(df["FTHG"], errors="coerce")
    df["FTAG"] = pd.to_numeric(df["FTAG"], errors="coerce")
    df["HomeTeam"] = df["HomeTeam"].astype(str).str.strip()
    df["AwayTeam"] = df["AwayTeam"].astype(str).str.strip()
    df = df.dropna(subset=["Date", "FTHG", "FTAG"])
    df = df[(df["HomeTeam"] != "") & (df["HomeTeam"] != "nan")]
    df[["FTHG", "FTAG"]] = df[["FTHG", "FTAG"]].astype(int)
    for col in SHOT_COLUMNS + ODDS_COLUMNS:        # shots/xG + odds as numbers
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values("Date").reset_index(drop=True)


@st.cache_data(ttl=3600, show_spinner=False)
def load_fixtures() -> pd.DataFrame:
    """Fetch upcoming EPL fixtures (with pre-match bookmaker odds)."""
    df = _fetch_csv(f"{BASE_URL}/fixtures.csv")
    if df.empty or "Div" not in df.columns:
        return pd.DataFrame()
    df = df[df["Div"].astype(str).str.strip() == LEAGUE].copy()
    if df.empty:
        return df
    df["Date"] = _parse_dates(df["Date"])
    df["HomeTeam"] = df["HomeTeam"].astype(str).str.strip()
    df["AwayTeam"] = df["AwayTeam"].astype(str).str.strip()
    # Coerce every odds column we might use to numeric
    for col in ODDS_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Date"])
    # Kick-off as a UTC timestamp (feed times are UK local time)
    times = (df["Time"].astype(str).str.strip() if "Time" in df.columns
             else pd.Series("15:00", index=df.index))
    times = times.where(times.str.match(r"^\d{1,2}:\d{2}$"), "15:00")
    local = pd.to_datetime(df["Date"].dt.strftime("%Y-%m-%d") + " " + times,
                           errors="coerce")
    df["Kickoff"] = (local.dt.tz_localize(UK_TZ, ambiguous="NaT",
                                          nonexistent="shift_forward")
                     .dt.tz_convert("UTC"))
    df["Kickoff"] = df["Kickoff"].fillna(
        df["Date"].dt.tz_localize("UTC") + pd.Timedelta(hours=14))
    return df.sort_values(["Kickoff"]).reset_index(drop=True)


def upcoming_fixtures(fixtures: pd.DataFrame, days: int = 7) -> pd.DataFrame:
    """Fixtures that have not kicked off yet, within the next `days`."""
    if fixtures.empty or "Kickoff" not in fixtures.columns:
        return fixtures.iloc[0:0] if not fixtures.empty else fixtures
    now = pd.Timestamp.now(tz="UTC")
    return fixtures[(fixtures["Kickoff"] > now)
                    & (fixtures["Kickoff"] <= now + pd.Timedelta(days=days))]


# =============================================================================
# 2. CORE MATHEMATICAL MODEL (Dixon-Coles Poisson, time-weighted MLE)
# =============================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def fit_model(results: pd.DataFrame, half_life_days: int,
              goals_weight: float = 0.5) -> dict:
    """Cached wrapper used by the live dashboard."""
    return _fit_model(results, half_life_days, goals_weight)


def _col(df: pd.DataFrame, name: str) -> np.ndarray:
    """Column as float array, all-NaN if the column is absent."""
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)
    return np.full(len(df), np.nan)


def goal_signal(results: pd.DataFrame, w: np.ndarray, goals_weight: float):
    """
    Blend actual goals with shot quality for each side of each match.
    Shot quality = xG if present, else shots on target x goals-per-SOT
    (estimated from the same weighted training data), else actual goals.
    Returns (signal_home, signal_away, goals_per_sot, n_xg_matches).
    """
    gh = results["FTHG"].to_numpy(dtype=float)
    ga = results["FTAG"].to_numpy(dtype=float)
    sh, sa = _col(results, "HST"), _col(results, "AST")
    xh, xa = _col(results, "HxG"), _col(results, "AxG")

    # League conversion rate: weighted goals / weighted shots on target
    num = den = 0.0
    for g, st in ((gh, sh), (ga, sa)):
        ok = ~np.isnan(st)
        num += np.sum(w[ok] * g[ok])
        den += np.sum(w[ok] * st[ok])
    per_sot = num / den if den > 0 else np.nan

    def quality(g, st, xg):
        q = g.copy()
        has_sot = ~np.isnan(st)
        if np.isfinite(per_sot):
            q[has_sot] = per_sot * st[has_sot]
        has_xg = ~np.isnan(xg)
        q[has_xg] = xg[has_xg]
        return q

    qh, qa = quality(gh, sh, xh), quality(ga, sa, xa)
    yh = goals_weight * gh + (1 - goals_weight) * qh
    ya = goals_weight * ga + (1 - goals_weight) * qa
    n_xg = int(np.sum(~np.isnan(xh) & ~np.isnan(xa)))
    return yh, ya, float(per_sot), n_xg


def _fit_model(results: pd.DataFrame, half_life_days: int,
               goals_weight: float = 0.5) -> dict:
    """
    Fit attack/defence ratings by weighted maximum likelihood.
    Returns a plain dict (cache-friendly) with all model parameters.
    """
    teams = sorted(set(results["HomeTeam"]) | set(results["AwayTeam"]))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)

    hi = results["HomeTeam"].map(idx).to_numpy()
    ai = results["AwayTeam"].map(idx).to_numpy()
    # Actual goals (used for the Dixon-Coles low-score correction)
    gh = results["FTHG"].to_numpy(dtype=float)
    ga = results["FTAG"].to_numpy(dtype=float)

    # Exponential time decay: a match half_life_days old counts half as much.
    age = (results["Date"].max() - results["Date"]).dt.days.to_numpy()
    w = np.exp(-np.log(2) / half_life_days * age)

    # What the attack/defence ratings are fitted to (goals + shot quality)
    yh, ya, per_sot, n_xg = goal_signal(results, w, goals_weight)

    def nll_and_grad(p):
        mu, home = p[0], p[1]
        att, dfn = p[2:2 + n], p[2 + n:]
        eta_h = mu + home + att[hi] - dfn[ai]
        eta_a = mu + att[ai] - dfn[hi]
        lam_h, lam_a = np.exp(eta_h), np.exp(eta_a)

        nll = np.sum(w * (lam_h - yh * eta_h + lam_a - ya * eta_a))
        nll += SHRINKAGE * (att @ att + dfn @ dfn)

        # Analytic gradient (fast + exact)
        rh, ra = w * (lam_h - yh), w * (lam_a - ya)
        g = np.zeros_like(p)
        g[0] = rh.sum() + ra.sum()
        g[1] = rh.sum()
        g_att, g_def = np.zeros(n), np.zeros(n)
        np.add.at(g_att, hi, rh)
        np.add.at(g_att, ai, ra)
        np.add.at(g_def, ai, -rh)
        np.add.at(g_def, hi, -ra)
        g[2:2 + n] = g_att + 2 * SHRINKAGE * att
        g[2 + n:] = g_def + 2 * SHRINKAGE * dfn
        return nll, g

    x0 = np.zeros(2 + 2 * n)
    x0[0] = np.log(max(np.average(ya, weights=w), 0.1))
    x0[1] = 0.25
    res = minimize(nll_and_grad, x0, jac=True, method="L-BFGS-B",
                   options={"maxiter": 5000, "ftol": 1e-15, "gtol": 1e-9})
    p = res.x
    mu, home = p[0], p[1]
    att, dfn = p[2:2 + n], p[2 + n:]

    # ---- Dixon-Coles rho: 1-D profile likelihood on low scorelines ----------
    lam_h = np.exp(mu + home + att[hi] - dfn[ai])
    lam_a = np.exp(mu + att[ai] - dfn[hi])
    best_rho, best_ll = 0.0, -np.inf
    for rho in np.arange(-0.25, 0.1001, 0.0025):
        tau = np.ones_like(lam_h)
        m00 = (gh == 0) & (ga == 0)
        m01 = (gh == 0) & (ga == 1)
        m10 = (gh == 1) & (ga == 0)
        m11 = (gh == 1) & (ga == 1)
        tau[m00] = 1 - lam_h[m00] * lam_a[m00] * rho
        tau[m01] = 1 + lam_h[m01] * rho
        tau[m10] = 1 + lam_a[m10] * rho
        tau[m11] = 1 - rho
        if np.any(tau <= 0):
            continue
        ll = np.sum(w * np.log(tau))
        if ll > best_ll:
            best_ll, best_rho = ll, rho

    # Teams that appear in the most recent season = current EPL teams
    latest = results["Season"].max()
    current = sorted(set(results.loc[results["Season"] == latest, "HomeTeam"])
                     | set(results.loc[results["Season"] == latest, "AwayTeam"]))

    # Prior for teams the model has never seen (e.g. newly promoted sides at
    # the start of a season): the average of the 3 weakest current teams.
    cur_idx = [idx[t] for t in (current or teams)]
    weakest = sorted(cur_idx, key=lambda i: att[i] + dfn[i])[:3]
    prior = {"attack": float(att[weakest].mean()),
             "defence": float(dfn[weakest].mean())}

    return {
        "mu": float(mu),
        "home": float(home),
        "rho": float(best_rho),
        "prior": prior,
        "attack": {t: float(att[i]) for t, i in idx.items()},
        "defence": {t: float(dfn[i]) for t, i in idx.items()},
        "teams": teams,
        "current_teams": current or teams,
        "n_matches": int(len(results)),
        "n_current": int((results["Season"] == latest).sum()),
        "last_date": results["Date"].max(),
        "converged": bool(res.success),
        "goals_per_sot": per_sot,
        "n_xg": n_xg,
        "goals_weight": goals_weight,
    }


def expected_goals(model: dict, home: str, away: str, adj: dict | None = None):
    """
    Expected goals (lambda) for each side, with optional 'What If' multipliers.
    adj keys: h_att, h_def, a_att, a_def  (fractions, e.g. -0.15 = -15%)
      *_att : change to that team's scoring power
      *_def : change to that team's defensive leakiness (+ = concedes more)
    """
    def r(kind, team):  # unknown team -> promoted-team prior
        return model[kind].get(team, model["prior"][kind])

    lam_h = np.exp(model["mu"] + model["home"]
                   + r("attack", home) - r("defence", away))
    lam_a = np.exp(model["mu"] + r("attack", away) - r("defence", home))
    if adj:
        lam_h *= (1 + adj["h_att"]) * (1 + adj["a_def"])
        lam_a *= (1 + adj["a_att"]) * (1 + adj["h_def"])
    return float(lam_h), float(lam_a)


def score_matrix(lam_h: float, lam_a: float, rho: float) -> np.ndarray:
    """Joint scoreline probability matrix [home_goals, away_goals]."""
    g = np.arange(MAX_GOALS + 1)
    m = np.outer(poisson.pmf(g, lam_h), poisson.pmf(g, lam_a))
    # Dixon-Coles low-score correction
    m[0, 0] *= 1 - lam_h * lam_a * rho
    m[0, 1] *= 1 + lam_h * rho
    m[1, 0] *= 1 + lam_a * rho
    m[1, 1] *= 1 - rho
    m = np.clip(m, 0, None)
    return m / m.sum()


def market_probs(m: np.ndarray) -> dict:
    """Derive every market probability from the score matrix."""
    g = np.arange(m.shape[0])
    total = np.add.outer(g, g)
    probs = {
        "H": np.tril(m, -1).sum(),     # home goals > away goals
        "D": np.trace(m),
        "A": np.triu(m, 1).sum(),
        "BTTS_Y": m[1:, 1:].sum(),
    }
    probs["BTTS_N"] = 1 - probs["BTTS_Y"]
    for line, key in [(1.5, "15"), (2.5, "25"), (3.5, "35")]:
        probs[f"O{key}"] = m[total > line].sum()
        probs[f"U{key}"] = 1 - probs[f"O{key}"]
    return {k: float(v) for k, v in probs.items()}


def predict(model, home, away, adj=None):
    lam_h, lam_a = expected_goals(model, home, away, adj)
    m = score_matrix(lam_h, lam_a, model["rho"])
    return lam_h, lam_a, m, market_probs(m)


def top_scores(m: np.ndarray, k: int = 3) -> list[tuple[str, float]]:
    """The k most likely exact scores as [("1-1", 0.12), ...]."""
    flat = np.argsort(m, axis=None)[::-1][:k]
    return [(f"{i}-{j}", float(m[i, j]))
            for i, j in zip(*np.unravel_index(flat, m.shape))]


def predicted_score(m: np.ndarray, probs: dict) -> tuple[str, float]:
    """
    Most likely score WITHIN the most likely result (home win / draw / away
    win). The single most likely score is 1-1 in ~80% of matches, so it
    rarely calls the right winner; this version picked the correct result
    ~50% of the time and the exact score ~11% (2024/25-2025/26 backtest).
    """
    best = max(("H", "D", "A"), key=lambda k: probs[k])
    i, j = np.indices(m.shape)
    mask = {"H": i > j, "D": i == j, "A": i < j}[best]
    sub = np.where(mask, m, -1)
    hi, ai = np.unravel_index(np.argmax(sub), m.shape)
    return f"{hi}-{ai}", float(m[hi, ai])


def score_predictions(window: pd.DataFrame, model: dict) -> pd.DataFrame:
    """Predicted score, top-3 scores and key probabilities per fixture."""
    rows = []
    for _, fx in window.iterrows():
        lh, la, m, p = predict(model, fx["HomeTeam"], fx["AwayTeam"])
        pred, pp = predicted_score(m, p)
        rows.append({
            "Kickoff (UK)": fx["Kickoff"].tz_convert(UK_TZ).strftime("%a %d %b %H:%M"),
            "Match": f"{fx['HomeTeam']} v {fx['AwayTeam']}",
            "Predicted score": pred,
            "Top 3 scores": ", ".join(f"{sc} ({pr*100:.0f}%)" for sc, pr in top_scores(m)),
            "Home %": p["H"] * 100, "Draw %": p["D"] * 100, "Away %": p["A"] * 100,
            "Over 2.5 %": p["O25"] * 100, "BTTS %": p["BTTS_Y"] * 100,
            "xG": f"{lh:.2f} - {la:.2f}",
        })
    return pd.DataFrame(rows)


def stake_pct(edge_pct: float, net_odds_: float) -> float:
    """Quarter-Kelly stake as % of bankroll, capped (protects against
    over-betting when the model is wrong, which it often is)."""
    if net_odds_ <= 1 or edge_pct <= 0:
        return 0.0
    kelly = (edge_pct / 100) / (net_odds_ - 1)
    return min(kelly * DEFAULTS["kelly_fraction"] * 100, DEFAULTS["max_stake_pct"])


def match_advice(preds: pd.DataFrame, flags: pd.DataFrame,
                 bankroll: float) -> pd.DataFrame:
    """
    One plain-English verdict per fixture: the single best value bet with a
    stake, or NO BET. Most matches should be NO BET - that is normal.
    """
    preds = preds.copy()
    advice, stakes = [], []
    for _, r in preds.iterrows():
        home, away = r["Match"].split(" v ", 1)
        f = flags[(flags["HomeTeam"] == home) & (flags["AwayTeam"] == away)] \
            if not flags.empty else flags
        if f.empty:
            advice.append("NO BET (price is fair)")
            stakes.append("")
            continue
        b = f.sort_values("Edge %", ascending=False).iloc[0]
        pct = stake_pct(b["Edge %"], b["Net odds"])
        advice.append(f"BET {MARKET_LABELS[b['Market']]} @ {b['Price']:.2f} "
                      f"({b['Book']}, edge +{b['Edge %']:.1f}%)")
        stakes.append(f"{pct:.1f}% = {bankroll * pct / 100:,.2f}")
    preds.insert(2, "Advice", advice)
    preds.insert(3, "Stake", stakes)
    return preds


def write_predictions_md(preds: pd.DataFrame, flags: pd.DataFrame, path: Path,
                         bankroll: float = DEFAULTS["bankroll"]):
    """Phone-friendly summary file (shown nicely on GitHub)."""
    stamp = pd.Timestamp.now(tz=UK_TZ).strftime("%a %d %b %Y %H:%M UK time")
    lines = [f"# EPL predictions", f"_Updated {stamp}_", ""]
    if preds.empty:
        lines += ["No Premier League fixtures in the next 7 days yet. "
                  "The feed usually publishes the next round mid-week.", ""]
    else:
        adv = match_advice(preds, flags, bankroll)
        lines += [f"## What to do (bankroll {bankroll:,.2f})", "",
                  "| Kick-off | Match | Advice | Stake |", "|---|---|---|---|"]
        for _, r in adv.iterrows():
            lines.append(f"| {r['Kickoff (UK)']} | {r['Match']} | **{r['Advice']}** | "
                         f"{r['Stake'] or '-'} |")
        lines += ["", "## Score predictions", "",
                  "| Kick-off | Match | Predicted | Top 3 scores | H / D / A % | Over 2.5 | BTTS |",
                  "|---|---|---|---|---|---|---|"]
        for _, r in preds.iterrows():
            lines.append(f"| {r['Kickoff (UK)']} | {r['Match']} | **{r['Predicted score']}** | "
                         f"{r['Top 3 scores']} | {r['Home %']:.0f} / {r['Draw %']:.0f} / "
                         f"{r['Away %']:.0f} | {r['Over 2.5 %']:.0f}% | {r['BTTS %']:.0f}% |")
        lines.append("")
    lines += ["## All value bets found", ""]
    if flags.empty:
        lines.append("None right now.")
    else:
        lines += ["| Match | Bet | Price | Fair odds | Edge |", "|---|---|---|---|---|"]
        for _, f in flags.iterrows():
            lines.append(f"| {f['HomeTeam']} v {f['AwayTeam']} | {MARKET_LABELS[f['Market']]} | "
                         f"{f['Price']:.2f} ({f['Book']}) | {100 / f['Model %']:.2f} | "
                         f"+{f['Edge %']:.1f}% |")
    lines += ["", "_How to read this: BET = the bookmaker's price is higher than the "
              "model's fair price. Stakes are 1/4 Kelly, capped at "
              f"{DEFAULTS['max_stake_pct']:.0f}% of bankroll. Check team news first. "
              "Exact scores are hard: the predicted score was right ~1 in 9 matches in "
              "testing, the top 3 together ~1 in 3, the winner ~1 in 2. The edge is not "
              "yet proven - paper-trade before staking real money._", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# 3. BETTING VALUE LOGIC
# =============================================================================
def value_rows(probs: dict, odds: dict, keys: list[str]) -> list[dict]:
    """Build one row per market: model %, fair odds, book odds, edge, Kelly."""
    rows = []
    for k in keys:
        p = probs[k]
        fair = 1 / p if p > 0 else np.nan
        book = odds.get(k, np.nan)
        if pd.notna(book) and book > 1:
            edge = p * book - 1                      # expected return per 1 unit
            kelly = max(edge / (book - 1), 0)        # full-Kelly stake fraction
        else:
            book, edge, kelly = np.nan, np.nan, np.nan
        rows.append({
            "Market": MARKET_LABELS[k],
            "Model %": p * 100,
            "Fair Odds": fair,
            "Book Odds": book,
            "Implied %": (100 / book) if pd.notna(book) else np.nan,
            "Edge %": edge * 100 if pd.notna(edge) else np.nan,
            "¼ Kelly %": kelly * 25 if pd.notna(kelly) else np.nan,
        })
    return rows


def fixture_odds(row: pd.Series, book: str) -> dict:
    cols = BOOKMAKERS[book]
    return {k: float(net_odds(row[c], book)) for k, c in cols.items()
            if c in row.index and pd.notna(row[c]) and row[c] > 1}


def is_value(edge_pct, min_edge: float, max_edge: float) -> bool:
    """Value = edge at or above the minimum and below the sanity cap.
    Very large 'edges' usually mean the model is missing news (injuries,
    rotation) that the market has priced in; backtests lost money on them."""
    return pd.notna(edge_pct) and min_edge <= edge_pct < max_edge


def style_value(df: pd.DataFrame, min_edge: float, max_edge: float = np.inf):
    """Green-highlight every row whose edge is in the value band."""
    def _row_style(row):
        if is_value(row["Edge %"], min_edge, max_edge):
            return ["background-color: #1e7b34; color: white; font-weight: 600"] * len(row)
        return [""] * len(row)

    fmt = {"Model %": "{:.1f}", "Fair Odds": "{:.2f}", "Book Odds": "{:.2f}",
           "Implied %": "{:.1f}", "Edge %": "{:+.1f}", "¼ Kelly %": "{:.1f}"}
    fmt = {k: v for k, v in fmt.items() if k in df.columns}
    return df.style.apply(_row_style, axis=1).format(fmt, na_rep="—")


# =============================================================================
# 4. WALK-FORWARD BACKTEST (no look-ahead)
# =============================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def backtest_predictions(results: pd.DataFrame, season: str,
                         half_life_days: int, goals_weight: float = 0.5,
                         refit_days: int = 7) -> pd.DataFrame:
    """
    Predict every match of `season` using ONLY matches played before it.
    The model is refitted every `refit_days`, exactly as it would have been
    used live. Returns the season's matches with model probabilities attached.
    """
    test = results[results["Season"] == season].sort_values("Date")
    if test.empty:
        return pd.DataFrame()
    out = []
    cutoff = test["Date"].min().normalize()
    end = test["Date"].max()
    while cutoff <= end:
        nxt = cutoff + pd.Timedelta(days=refit_days)
        block = test[(test["Date"] >= cutoff) & (test["Date"] < nxt)]
        train = results[results["Date"] < cutoff]
        if not block.empty and len(train) >= 100:
            model = _fit_model(train, half_life_days, goals_weight)
            for _, row in block.iterrows():
                lh, la, _, p = predict(model, row["HomeTeam"], row["AwayTeam"])
                rec = row.to_dict()
                rec.update({f"p_{k}": v for k, v in p.items()})
                rec.update({"xG_H": lh, "xG_A": la})
                out.append(rec)
        cutoff = nxt
    return pd.DataFrame(out)


def backtest_bets(preds: pd.DataFrame, book: str, timing: str,
                  min_edge: float, max_edge: float = np.inf) -> pd.DataFrame:
    """Place a 1-unit bet on every market whose edge is in the value band."""
    cols = (BOOKMAKERS if timing == "pre" else CLOSING_BOOKMAKERS)[book]
    close_cols = CLOSING_BOOKMAKERS[book]
    hg, ag = preds["FTHG"], preds["FTAG"]
    won = {"H": hg > ag, "D": hg == ag, "A": hg < ag,
           "O25": (hg + ag) > 2.5, "U25": (hg + ag) < 2.5}
    bets = []
    for k, col in cols.items():
        if col not in preds.columns:
            continue
        raw = preds[col]
        odds = net_odds(raw, book)
        p = preds[f"p_{k}"]
        edge = p * odds - 1
        mask = odds.gt(1) & (edge * 100 >= min_edge) & (edge * 100 < max_edge)
        if not mask.any():
            continue
        sel = preds.loc[mask, ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]].copy()
        sel["Market"] = MARKET_LABELS[k]
        sel["Model %"] = p[mask] * 100
        sel["Odds"] = odds[mask]
        sel["Edge %"] = edge[mask] * 100
        sel["Won"] = won[k][mask]
        sel["Profit"] = np.where(sel["Won"], sel["Odds"] - 1, -1.0)
        # Closing-line value: did we get a better price than the close?
        cc = close_cols[k]
        if timing == "pre" and cc in preds.columns:
            sel["CLV %"] = (raw[mask] / preds.loc[mask, cc] - 1) * 100
        bets.append(sel)
    if not bets:
        return pd.DataFrame()
    b = pd.concat(bets).sort_values("Date").reset_index(drop=True)
    b["Cumulative"] = b["Profit"].cumsum()
    return b


def market_benchmark(preds: pd.DataFrame, book: str):
    """
    Compare 1X2 log-loss of the model vs the bookmaker's margin-free closing
    probabilities (falls back to pre-match odds). Lower log-loss = better.
    Also returns a calibration table for the model.
    """
    for source in (CLOSING_BOOKMAKERS, BOOKMAKERS):
        need = [source[book][k] for k in ("H", "D", "A")]
        if all(c in preds.columns for c in need):
            df = preds.dropna(subset=need)
            df = df[(df[need] > 1).all(axis=1)]
            if not df.empty:
                break
    else:
        return None
    if df.empty:
        return None
    inv = 1 / df[need].to_numpy()
    mkt = inv / inv.sum(axis=1, keepdims=True)          # remove bookmaker margin
    model = df[["p_H", "p_D", "p_A"]].to_numpy()
    outcome = np.select([df["FTHG"] > df["FTAG"], df["FTHG"] == df["FTAG"]], [0, 1], 2)
    rows = np.arange(len(df))
    ll_model = float(-np.mean(np.log(model[rows, outcome] + 1e-12)))
    ll_mkt = float(-np.mean(np.log(mkt[rows, outcome] + 1e-12)))

    # Calibration: when the model says ~X%, how often did it happen?
    pred = model.ravel()
    actual = np.zeros_like(model)
    actual[rows, outcome] = 1
    cal = pd.DataFrame({"pred": pred, "hit": actual.ravel()})
    cal["Bucket"] = pd.cut(cal["pred"], np.linspace(0, 1, 11), include_lowest=True)
    cal = cal.groupby("Bucket", observed=True).agg(
        **{"Model said %": ("pred", "mean"), "Happened %": ("hit", "mean"),
           "Outcomes": ("hit", "size")}).reset_index()
    cal[["Model said %", "Happened %"]] *= 100
    cal["Bucket"] = cal["Bucket"].astype(str)
    return ll_model, ll_mkt, len(df), cal


# =============================================================================
# 5. PAPER TRADING (automatic log + settlement)
# =============================================================================
PAPER_COLS = ["Logged (UTC)", "Kickoff (UTC)", "Date", "HomeTeam", "AwayTeam",
              "Market", "Book", "Price", "Net odds", "Model %", "Edge %",
              "Goals weight", "Min edge", "Max edge"]


def flagged_bets(window: pd.DataFrame, model: dict, book: str,
                 min_edge: float, max_edge: float) -> pd.DataFrame:
    """All value flags for upcoming fixtures, with the price on offer now."""
    rows = []
    now = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M")
    cols = BOOKMAKERS[book]
    for _, fx in window.iterrows():
        _, _, _, probs = predict(model, fx["HomeTeam"], fx["AwayTeam"])
        for k, col in cols.items():
            raw = fx.get(col, np.nan)
            if pd.isna(raw) or raw <= 1:
                continue
            net = float(net_odds(raw, book))
            edge = (probs[k] * net - 1) * 100
            if not is_value(edge, min_edge, max_edge):
                continue
            rows.append({
                "Logged (UTC)": now,
                "Kickoff (UTC)": fx["Kickoff"].strftime("%Y-%m-%d %H:%M"),
                "Date": fx["Date"].strftime("%Y-%m-%d"),
                "HomeTeam": fx["HomeTeam"], "AwayTeam": fx["AwayTeam"],
                "Market": k, "Book": book, "Price": float(raw),
                "Net odds": round(net, 4), "Model %": round(probs[k] * 100, 2),
                "Edge %": round(edge, 2),
                "Goals weight": model.get("goals_weight", np.nan),
                "Min edge": min_edge,
                "Max edge": None if np.isinf(max_edge) else max_edge,
            })
    return pd.DataFrame(rows, columns=PAPER_COLS)


def read_paper_log() -> pd.DataFrame:
    try:
        if PAPER_LOG.exists():
            return pd.read_csv(PAPER_LOG)
    except Exception:
        pass
    return pd.DataFrame(columns=PAPER_COLS)


def update_paper_log(new_rows: pd.DataFrame):
    """
    Append new flags; the FIRST price seen for a match/market/book is kept
    (that is the bet you would have placed). Returns (log, n_added, error).
    """
    log = read_paper_log()
    if new_rows.empty:
        return log, 0, None
    key = ["Date", "HomeTeam", "AwayTeam", "Market", "Book"]
    before = len(log)
    merged = (pd.concat([log, new_rows], ignore_index=True)
              .drop_duplicates(subset=key, keep="first")
              .reset_index(drop=True))
    added = len(merged) - before
    if added == 0:
        return log, 0, None
    try:  # atomic write: never leaves a half-written file
        PAPER_LOG.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=PAPER_LOG.parent, suffix=".tmp")
        with os.fdopen(fd, "w", newline="") as f:
            merged.to_csv(f, index=False)
        os.replace(tmp, PAPER_LOG)
    except Exception as exc:
        return log, 0, str(exc)
    return merged, added, None


def settle_paper_log(log: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """Attach final score, win/loss, profit (1 unit) and closing-line value."""
    if log.empty:
        return log.assign(Score=[], Won=[], Profit=[], Close=[], **{"CLV %": []})
    res = results.copy()
    res["Date"] = res["Date"].dt.strftime("%Y-%m-%d")
    keep = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"] + [
        c for c in res.columns if c in ODDS_COLUMNS]
    res = res[keep].drop_duplicates(["Date", "HomeTeam", "AwayTeam"])
    out = log.merge(res, on=["Date", "HomeTeam", "AwayTeam"], how="left")

    hg, ag = out["FTHG"], out["FTAG"]
    won_map = {"H": hg > ag, "D": hg == ag, "A": hg < ag,
               "O25": (hg + ag) > 2.5, "U25": (hg + ag) < 2.5}
    played = hg.notna() & ag.notna()
    won = pd.Series(np.nan, index=out.index, dtype="object")
    close = pd.Series(np.nan, index=out.index, dtype=float)
    for i, row in out.iterrows():
        k = row["Market"]
        if played[i] and k in won_map:
            won[i] = bool(won_map[k][i])
        ccol = CLOSING_BOOKMAKERS.get(row["Book"], {}).get(k)
        if ccol and ccol in out.columns and pd.notna(row.get(ccol)):
            close[i] = row[ccol]
    out["Score"] = np.where(played, hg.astype("Int64").astype(str) + "-"
                            + ag.astype("Int64").astype(str), "pending")
    out["Won"] = won
    out["Profit"] = np.where(played, np.where(won == True, out["Net odds"] - 1, -1.0),  # noqa: E712
                             np.nan)
    out["Close"] = close
    out["CLV %"] = np.where(close > 1, (out["Price"] / close - 1) * 100, np.nan)
    return out[PAPER_COLS + ["Score", "Won", "Profit", "Close", "CLV %"]]


def paper_summary(settled: pd.DataFrame) -> dict:
    done = settled[settled["Profit"].notna()]
    n = len(done)
    out = {"logged": len(settled), "settled": n, "profit": done["Profit"].sum(),
           "roi": np.nan, "lo": np.nan, "hi": np.nan,
           "clv": settled["CLV %"].mean(), "n_clv": int(settled["CLV %"].notna().sum())}
    if n:
        out["roi"] = done["Profit"].mean() * 100
    if n > 1:
        se = done["Profit"].std(ddof=1) / np.sqrt(n) * 100
        out["lo"], out["hi"] = out["roi"] - 1.96 * se, out["roi"] + 1.96 * se
    return out


def show_df(data, **kwargs):
    """st.dataframe at full width, compatible with old and new Streamlit."""
    try:
        return st.dataframe(data, width="stretch", **kwargs)
    except Exception:  # Streamlit versions before width="stretch" existed
        return st.dataframe(data, use_container_width=True, **kwargs)


# =============================================================================
# 6. STREAMLIT INTERFACE
# =============================================================================
def main():
    st.set_page_config(page_title="EPL Value Engine", page_icon="⚽", layout="wide")

    # ---------------- Sidebar: global settings ----------------
    st.sidebar.header("⚙️ Model settings")
    book = st.sidebar.selectbox("Bookmaker odds source", list(BOOKMAKERS),
                                index=list(BOOKMAKERS).index(DEFAULTS["book"]))
    min_edge = st.sidebar.slider("Minimum edge to flag as VALUE (%)", 0.0, 15.0,
                                 DEFAULTS["min_edge"], 0.5)
    max_edge = st.sidebar.slider("Maximum believable edge (%)", 5, 100,
                                 DEFAULTS["max_edge"], 5,
                                 help="Edges above this are ignored: they usually "
                                      "mean the model is missing team news. "
                                      "100 = no cap.")
    if max_edge >= 100:
        max_edge = np.inf
    goals_pct = st.sidebar.slider("Rating signal: weight on actual goals (%)",
                                  0, 100, DEFAULTS["goals_pct"], 10,
                                  help="Rest of the weight goes to shot quality "
                                       "(xG when available, else shots on target). "
                                       "100 = goals only. 50 tested best.")
    goals_weight = goals_pct / 100
    half_life = st.sidebar.slider("Form half-life (days)", 60, 720,
                                  DEFAULTS["half_life"], 20,
                                  help="Lower = reacts faster to recent form; "
                                       "higher = more stable ratings. 400 tested best.")
    seasons_back = st.sidebar.slider("Previous seasons to include", 0, 3,
                                     DEFAULTS["seasons_back"])
    bankroll = st.sidebar.number_input("Bankroll (for stake sizes)", min_value=1.0,
                                       value=DEFAULTS["bankroll"], step=10.0,
                                       help="Your total betting bank in any currency. "
                                            "Stakes are 1/4 Kelly, max "
                                            f"{DEFAULTS['max_stake_pct']:.0f}% per bet.")
    autolog = st.sidebar.toggle("Auto-log value bets (paper trading)", value=True,
                                help=f"Saves every green flag to {PAPER_LOG.name} "
                                     "so you can check real results later.")
    if st.sidebar.button("🔄 Force data refresh"):
        st.cache_data.clear()
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.header("🧪 What-If simulator")
    st.sidebar.caption("Applies to the matchup selected in the Match Analyzer.")
    h_att = st.sidebar.slider("Home attack change (%)", -50, 50, 0, 5)
    h_def = st.sidebar.slider("Home defence leakiness (%)", -50, 50, 0, 5,
                              help="+ = concedes more (e.g. key defender out)")
    a_att = st.sidebar.slider("Away attack change (%)", -50, 50, 0, 5)
    a_def = st.sidebar.slider("Away defence leakiness (%)", -50, 50, 0, 5)
    adj = {"h_att": h_att / 100, "h_def": h_def / 100,
           "a_att": a_att / 100, "a_def": a_def / 100}
    adj_active = any(v != 0 for v in adj.values())

    # ---------------- Data + model ----------------
    st.title("⚽ EPL Poisson Value Engine")
    with st.spinner("Fetching data and fitting model..."):
        results = load_results(seasons_back)
        fixtures = load_fixtures()

    if results.empty:
        st.error("Could not reach Football-Data.co.uk right now. "
                 "The app will retry automatically; or press 'Force data refresh'.")
        st.stop()

    model = fit_model(results, half_life, goals_weight)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Matches in model", f"{model['n_matches']:,}")
    c2.metric("This-season matches", model["n_current"])
    c3.metric("Latest result", model["last_date"].strftime("%d %b %Y"))
    c4.metric("Home advantage", f"×{np.exp(model['home']):.2f}")
    c5.metric("Dixon-Coles ρ", f"{model['rho']:+.3f}")
    age_days = (pd.Timestamp(dt.date.today()) - model["last_date"]).days
    if age_days > 10:
        st.warning(f"The newest result in the feed is {age_days} days old, so the "
                   "ratings miss recent matches. Football-Data.co.uk usually "
                   "updates twice a week; the app picks it up automatically.")
    xg_note = (f"{model['n_xg']} matches use real xG" if model["n_xg"]
               else "shots on target used as xG stand-in")
    st.caption(f"Rating signal: {goals_pct}% actual goals, {100 - goals_pct}% shot "
               f"quality ({xg_note}; {model['goals_per_sot']:.2f} goals per shot on target).")
    if not model["converged"]:
        st.warning("Optimiser reported non-convergence; ratings may be approximate.")

    # Only fixtures that have not kicked off yet (next 7 days).
    # Teams unknown to the model fall back to the promoted-team prior.
    window = upcoming_fixtures(fixtures)

    # Paper trading: log today's flags, settle older ones from results
    log_note = None
    if autolog and not window.empty:
        _, added, err = update_paper_log(
            flagged_bets(window, model, book, min_edge, max_edge))
        log_note = (f"Could not write {PAPER_LOG}: {err}" if err
                    else f"{added} new value bet(s) logged." if added else None)

    tab_value, tab_match, tab_ratings, tab_bt, tab_paper = st.tabs(
        ["📋 Master Fixtures Value Table", "🔍 Match Analyzer", "📊 Team Ratings",
         "🧾 Backtest", "📒 Paper Trading"])

    # ================= TAB 1: Master value table =================
    with tab_value:
        st.markdown(f"### This week's fixtures ranked by edge — odds: **{book}**")
        if window.empty:
            st.info("No EPL fixtures still to kick off in the next 7 days are in the "
                    "feed yet (it is usually published mid-week, and it is empty "
                    "during international breaks). Use the Match Analyzer meanwhile.")
        else:
            rows = []
            for _, fx in window.iterrows():
                _, _, _, probs = predict(model, fx["HomeTeam"], fx["AwayTeam"])
                odds = fixture_odds(fx, book)
                for r in value_rows(probs, odds, ["H", "D", "A", "O25", "U25"]):
                    if pd.isna(r["Book Odds"]):
                        continue
                    r.update({"Date": fx["Date"].strftime("%a %d %b"),
                              "Match": f"{fx['HomeTeam']} v {fx['AwayTeam']}"})
                    rows.append(r)
            if not rows:
                st.info(f"No {book} odds published yet for these fixtures. "
                        "Try another odds source in the sidebar.")
            else:
                table = pd.DataFrame(rows)[["Date", "Match", "Market", "Model %",
                                            "Fair Odds", "Book Odds", "Implied %",
                                            "Edge %", "¼ Kelly %"]]
                table = table.sort_values("Edge %", ascending=False)
                n_value = int(table["Edge %"].apply(
                    lambda e: is_value(e, min_edge, max_edge)).sum())
                cap_txt = "" if np.isinf(max_edge) else f" and < {max_edge}%"
                st.metric("Value bets flagged", n_value,
                          help=f"Rows with edge ≥ {min_edge}%{cap_txt} are green.")
                show_df(style_value(table, min_edge, max_edge), hide_index=True,
                             height=600)
                st.caption("Edge % = model probability × book odds − 1. "
                           "¼ Kelly % = suggested stake as % of bankroll (quarter-Kelly). "
                           "What-If sliders do not affect this table.")

        if not window.empty:
            st.markdown("### What to do — match by match")
            sp = score_predictions(window, model)
            adv = match_advice(sp, flagged_bets(window, model, book, min_edge, max_edge),
                               bankroll)

            def _adv_style(row):
                green = "background-color: #1e7b34; color: white; font-weight: 600"
                return [green if row["Advice"].startswith("BET") and c in ("Advice", "Stake")
                        else "" for c in row.index]

            show_df(adv.style.apply(_adv_style, axis=1).format(
                {c: "{:.0f}" for c in ["Home %", "Draw %", "Away %", "Over 2.5 %", "BTTS %"]}),
                hide_index=True)
            st.caption("Advice = the single best value bet for each match, or NO BET. "
                       "Most matches should be NO BET. Stakes are 1/4 Kelly, capped at "
                       f"{DEFAULTS['max_stake_pct']:.0f}% of bankroll. Always check team "
                       "news first (use the What-If sliders in the Match Analyzer).")
            st.caption("Predicted score = most likely score within the most likely "
                       "result. In testing it got the winner right ~50% of the time "
                       "and the exact score ~11%; the top 3 scores covered ~32%.")

    # ================= TAB 2: Match analyzer =================
    with tab_match:
        st.markdown("### Upcoming Match Selector")
        mode = st.radio("Pick from", ["This week's fixtures", "Any two teams"],
                        horizontal=True,
                        index=0 if not window.empty else 1)
        odds = {}
        if mode == "This week's fixtures" and not window.empty:
            labels = [f"{r.Kickoff.tz_convert(UK_TZ):%a %d %b %H:%M} — "
                      f"{r.HomeTeam} v {r.AwayTeam}"
                      for r in window.itertuples()]
            choice = st.selectbox("Fixture", range(len(labels)),
                                  format_func=lambda i: labels[i])
            fx = window.iloc[choice]
            home, away = fx["HomeTeam"], fx["AwayTeam"]
            odds = fixture_odds(fx, book)
        else:
            teams = model["current_teams"]
            cA, cB = st.columns(2)
            home = cA.selectbox("Home team", teams, index=0)
            away = cB.selectbox("Away team", teams, index=min(1, len(teams) - 1))
            if home == away:
                st.warning("Pick two different teams.")
                st.stop()
            if not fixtures.empty:
                hit = fixtures[(fixtures["HomeTeam"] == home)
                               & (fixtures["AwayTeam"] == away)]
                if not hit.empty:
                    odds = fixture_odds(hit.iloc[0], book)

        lam_h, lam_a, m, probs = predict(model, home, away, adj if adj_active else None)

        if adj_active:
            base_h, base_a, _, base_p = predict(model, home, away)
            st.info(f"🧪 What-If active — base xG {base_h:.2f} v {base_a:.2f} "
                    f"→ adjusted {lam_h:.2f} v {lam_a:.2f}")

        st.markdown(f"## {home} vs {away}")
        x1, x2, x3, x4 = st.columns(4)
        x1.metric(f"{home} xG", f"{lam_h:.2f}")
        x2.metric(f"{away} xG", f"{lam_a:.2f}")
        x3.metric("Total xG", f"{lam_h + lam_a:.2f}")
        ps, pp = predicted_score(m, probs)
        x4.metric("Predicted score", ps, delta=f"{pp*100:.0f}% chance", delta_color="off")

        st.markdown("#### Match outcome")
        o1, o2, o3 = st.columns(3)
        for col, k in zip((o1, o2, o3), ("H", "D", "A")):
            col.metric(MARKET_LABELS[k], f"{probs[k]*100:.1f}%",
                       delta=f"Fair odds {1/probs[k]:.2f}", delta_color="off")

        st.markdown("#### Goals markets")
        g1, g2, g3, g4 = st.columns(4)
        g1.metric("Over 1.5", f"{probs['O15']*100:.1f}%",
                  delta=f"Fair {1/probs['O15']:.2f}", delta_color="off")
        g2.metric("Over 2.5", f"{probs['O25']*100:.1f}%",
                  delta=f"Fair {1/probs['O25']:.2f}", delta_color="off")
        g3.metric("Over 3.5", f"{probs['O35']*100:.1f}%",
                  delta=f"Fair {1/probs['O35']:.2f}", delta_color="off")
        g4.metric("BTTS Yes", f"{probs['BTTS_Y']*100:.1f}%",
                  delta=f"Fair {1/probs['BTTS_Y']:.2f}", delta_color="off")

        st.markdown(f"#### Value check vs {book}")
        keys = ["H", "D", "A", "O15", "U15", "O25", "U25", "O35", "U35",
                "BTTS_Y", "BTTS_N"]
        vt = pd.DataFrame(value_rows(probs, odds, keys))
        show_df(style_value(vt, min_edge, max_edge), hide_index=True)
        if not odds:
            st.caption("No bookmaker odds in the feed for this pairing — fair odds only.")
        st.caption("The Football-Data feed publishes 1X2 and O/U 2.5 odds only; "
                   "compare the fair odds for O/U 1.5, 3.5 and BTTS against your "
                   "bookmaker yourself.")

        st.markdown("#### Most likely scorelines")
        flat = [(f"{i}-{j}", m[i, j]) for i in range(m.shape[0]) for j in range(m.shape[1])]
        flat.sort(key=lambda t: t[1], reverse=True)
        top = pd.DataFrame(flat[:8], columns=["Score", "Prob"])
        top["Prob %"] = (top["Prob"] * 100).round(1)
        top["Fair Odds"] = (1 / top["Prob"]).round(1)
        show_df(top[["Score", "Prob %", "Fair Odds"]], hide_index=True)

    # ================= TAB 3: Ratings =================
    with tab_ratings:
        st.markdown("### Team strength (1.00 = league average)")
        rt = pd.DataFrame({
            "Team": model["current_teams"],
            "Attack ×": [np.exp(model["attack"][t]) for t in model["current_teams"]],
            "Defence × (conceded, lower=better)":
                [np.exp(-model["defence"][t]) for t in model["current_teams"]],
        })
        rt["Net strength"] = rt["Attack ×"] / rt["Defence × (conceded, lower=better)"]
        rt = rt.sort_values("Net strength", ascending=False)
        show_df(rt.style.format({c: "{:.2f}" for c in rt.columns if c != "Team"}),
                     hide_index=True)
        lg_home = np.exp(model["mu"] + model["home"])
        lg_away = np.exp(model["mu"])
        st.caption(f"Average-vs-average expected goals: home {lg_home:.2f}, "
                   f"away {lg_away:.2f}.")

    # ================= TAB 4: Backtest =================
    with tab_bt:
        st.markdown("### Walk-forward backtest")
        st.caption("Replays a season week by week. Each week the model is refitted "
                   "using only matches already played, then bets 1 unit on every "
                   f"market with edge ≥ {min_edge}%"
                   + ("" if np.isinf(max_edge) else f" and < {max_edge}%")
                   + f" against **{book}** odds. "
                   "Uses the same sidebar settings as the live dashboard.")
        seasons = sorted(results["Season"].unique(), reverse=True)
        if len(seasons) < 2:
            st.info("Set 'Previous seasons to include' to at least 1 to backtest.")
        else:
            options = seasons[:-1]          # oldest season has no history to learn from
            b1, b2 = st.columns(2)
            bt_season = b1.selectbox("Season to test", options,
                                     index=min(1, len(options) - 1),
                                     format_func=lambda c: f"20{c[:2]}/{c[2:]}")
            timing = b2.radio("Bet at", ["Pre-match odds", "Closing odds"],
                              horizontal=True,
                              help="Pre-match = prices like the ones the live table "
                                   "shows. Closing = last price before kick-off "
                                   "(the toughest benchmark).")
            if st.toggle("Run backtest", value=False,
                         help="First run takes a few seconds; results are cached for an hour."):
                with st.spinner("Replaying the season week by week..."):
                    preds = backtest_predictions(results, bt_season, half_life,
                                                 goals_weight)
                if preds.empty:
                    st.warning("Not enough earlier matches to train on for this season.")
                else:
                    tkey = "pre" if timing.startswith("Pre") else "close"
                    bets = backtest_bets(preds, book, tkey, min_edge, max_edge)
                    bench = market_benchmark(preds, book)

                    if bets.empty:
                        st.info("No bets cleared the edge threshold with these odds.")
                    else:
                        n = len(bets)
                        profit = bets["Profit"].sum()
                        roi = profit / n * 100
                        se = bets["Profit"].std(ddof=1) / np.sqrt(n) * 100 if n > 1 else np.nan
                        k1, k2, k3, k4, k5 = st.columns(5)
                        k1.metric("Bets", n)
                        k2.metric("Hit rate", f"{bets['Won'].mean()*100:.1f}%")
                        k3.metric("Profit", f"{profit:+.1f} u")
                        k4.metric("ROI", f"{roi:+.1f}%")
                        if "CLV %" in bets.columns and bets["CLV %"].notna().any():
                            k5.metric("Avg CLV", f"{bets['CLV %'].mean():+.2f}%",
                                      help="Average % by which your price beat the "
                                           "closing price. Consistently positive CLV "
                                           "is the strongest sign of a real edge.")
                        else:
                            k5.metric("Avg CLV", "—")
                        if pd.notna(se):
                            lo, hi = roi - 1.96 * se, roi + 1.96 * se
                            verdict = ("profit is statistically meaningful" if lo > 0
                                       else "range includes zero → could be luck")
                            st.caption(f"ROI 95% range: {lo:+.1f}% to {hi:+.1f}% — {verdict}.")

                        st.markdown("#### Cumulative profit (units)")
                        st.line_chart(bets["Cumulative"])
                        st.caption("X-axis = bet number, in date order.")

                        c_left, c_right = st.columns(2)
                        with c_left:
                            st.markdown("#### By market")
                            bm = bets.groupby("Market").agg(
                                Bets=("Profit", "size"), Hit=("Won", "mean"),
                                Profit=("Profit", "sum")).reset_index()
                            bm["Hit %"] = bm.pop("Hit") * 100
                            bm["ROI %"] = bm["Profit"] / bm["Bets"] * 100
                            show_df(bm.style.format(
                                {"Hit %": "{:.1f}", "Profit": "{:+.1f}", "ROI %": "{:+.1f}"}),
                                hide_index=True)
                        with c_right:
                            st.markdown("#### By edge size")
                            bets["Edge band"] = pd.cut(
                                bets["Edge %"], [-np.inf, 5, 10, 20, np.inf],
                                labels=["< 5%", "5–10%", "10–20%", "20%+"])
                            be = bets.groupby("Edge band", observed=True).agg(
                                Bets=("Profit", "size"), Profit=("Profit", "sum")).reset_index()
                            be["ROI %"] = be["Profit"] / be["Bets"] * 100
                            show_df(be.style.format(
                                {"Profit": "{:+.1f}", "ROI %": "{:+.1f}"}),
                                hide_index=True)
                            st.caption("If the biggest 'edges' lose most, they are "
                                       "usually model blind spots, not value.")

                        with st.expander(f"Bet log ({n} bets)"):
                            log = bets.drop(columns=["Cumulative", "Edge band"]).copy()
                            log["Date"] = log["Date"].dt.strftime("%d %b %Y")
                            log["Score"] = log.pop("FTHG").astype(str) + "-" + log.pop("FTAG").astype(str)
                            fmt = {"Model %": "{:.1f}", "Odds": "{:.2f}",
                                   "Edge %": "{:+.1f}", "Profit": "{:+.2f}"}
                            if "CLV %" in log.columns:
                                fmt["CLV %"] = "{:+.1f}"
                            show_df(log.style.format(fmt, na_rep="—"),
                                         hide_index=True)

                    if bench:
                        ll_model, ll_mkt, n_m, cal = bench
                        st.markdown("#### Model vs market accuracy (1X2)")
                        a1, a2, a3 = st.columns(3)
                        a1.metric("Model log-loss", f"{ll_model:.4f}")
                        a2.metric(f"{book} log-loss (margin removed)", f"{ll_mkt:.4f}")
                        a3.metric("Matches compared", n_m)
                        if ll_model < ll_mkt:
                            st.success("The model's probabilities were MORE accurate than "
                                       "the market's this season.")
                        else:
                            st.warning("The market's probabilities were more accurate than "
                                       "the model's. Treat flagged edges with caution: "
                                       "most are likely model error.")
                        st.markdown("##### Calibration — when the model says X%, how often did it happen?")
                        show_df(cal.style.format({"Model said %": "{:.1f}",
                                                       "Happened %": "{:.1f}"}),
                                     hide_index=True)

    # ================= TAB 5: Paper trading =================
    with tab_paper:
        st.markdown("### Paper trading — is the edge real?")
        st.caption("Every green flag is saved with the price on offer when it was "
                   "first seen, then settled automatically from the results feed. "
                   "Beating the closing price (positive CLV) over a few hundred "
                   "bets is the most reliable sign of a genuine edge.")
        if log_note:
            st.info(log_note)
        log = read_paper_log()
        if log.empty:
            st.info("No bets logged yet. Flags are logged whenever the app (or the "
                    "scheduled `--log` run) sees a value bet before kick-off.")
        else:
            settled = settle_paper_log(log, results)
            sm = paper_summary(settled)
            p1, p2, p3, p4, p5 = st.columns(5)
            p1.metric("Logged", sm["logged"])
            p2.metric("Settled", sm["settled"])
            p3.metric("Profit", f"{sm['profit']:+.1f} u")
            p4.metric("ROI", "—" if np.isnan(sm["roi"]) else f"{sm['roi']:+.1f}%")
            p5.metric("Avg CLV", "—" if np.isnan(sm["clv"]) else f"{sm['clv']:+.2f}%",
                      help="Your logged price vs the closing price of the same book.")
            if not np.isnan(sm["lo"]):
                st.caption(f"ROI 95% range: {sm['lo']:+.1f}% to {sm['hi']:+.1f}%.")
            if sm["n_clv"] >= 100:
                if sm["clv"] > 1:
                    st.success("Consistently beating the closing line — evidence of a real edge.")
                elif sm["clv"] < 0:
                    st.warning("Not beating the closing line — profits so far are likely luck.")
            else:
                st.caption(f"{sm['n_clv']} bets with closing prices so far; "
                           "judge CLV after at least 100–200.")
            done = settled[settled["Profit"].notna()].sort_values("Kickoff (UTC)")
            if len(done) > 1:
                st.line_chart(done["Profit"].cumsum().reset_index(drop=True))
                st.caption("Cumulative profit (units), settled bets in kick-off order.")
            view = settled.sort_values("Kickoff (UTC)", ascending=False).copy()
            view["Market"] = view["Market"].map(MARKET_LABELS).fillna(view["Market"])
            show_df(view[["Kickoff (UTC)", "HomeTeam", "AwayTeam", "Market", "Book",
                          "Price", "Model %", "Edge %", "Score", "Profit", "Close",
                          "CLV %"]].style.format(
                {"Price": "{:.2f}", "Model %": "{:.1f}", "Edge %": "{:+.1f}",
                 "Profit": "{:+.2f}", "Close": "{:.2f}", "CLV %": "{:+.1f}"},
                na_rep="—"), hide_index=True)
            st.download_button("⬇️ Download log (CSV)", settled.to_csv(index=False),
                               file_name="paper_bets_settled.csv", mime="text/csv")
            st.caption(f"Log file: {PAPER_LOG}")

    st.markdown("---")
    st.caption("Data: Football-Data.co.uk (auto-refreshed hourly). "
               "Statistical estimates only — no outcome is guaranteed. Bet responsibly.")


def run_headless_log(argv=None) -> int:
    """
    Headless mode for a daily scheduled task (no browser needed):
        python epl_value_dashboard.py --log
    Fetches data, logs new value flags, writes predictions.md (predicted
    scores + value bets) and prints a paper-trading summary.
    """
    ap = argparse.ArgumentParser(description="Log EPL value bets (paper trading).")
    ap.add_argument("--log", action="store_true")
    ap.add_argument("--book", default=DEFAULTS["book"], choices=list(BOOKMAKERS))
    ap.add_argument("--min-edge", type=float, default=DEFAULTS["min_edge"])
    ap.add_argument("--max-edge", type=float, default=DEFAULTS["max_edge"],
                    help="100 or more = no cap")
    ap.add_argument("--goals-pct", type=int, default=DEFAULTS["goals_pct"])
    ap.add_argument("--half-life", type=int, default=DEFAULTS["half_life"])
    ap.add_argument("--seasons-back", type=int, default=DEFAULTS["seasons_back"])
    ap.add_argument("--bankroll", type=float, default=DEFAULTS["bankroll"])
    args = ap.parse_args(argv)
    logging.getLogger("streamlit").setLevel(logging.ERROR)

    results = load_results(args.seasons_back)
    if results.empty:
        print("ERROR: could not download results from Football-Data.co.uk")
        return 1
    model = fit_model(results, args.half_life, args.goals_pct / 100)
    max_edge = np.inf if args.max_edge >= 100 else args.max_edge
    window = upcoming_fixtures(load_fixtures())
    flags = flagged_bets(window, model, args.book, args.min_edge, max_edge)
    _, added, err = update_paper_log(flags)
    try:
        write_predictions_md(score_predictions(window, model), flags, PREDICTIONS_MD,
                             args.bankroll)
    except Exception as exc:
        print(f"WARNING: could not write {PREDICTIONS_MD}: {exc}")
    if err:
        print(f"ERROR writing {PAPER_LOG}: {err}")
        return 1
    sm = paper_summary(settle_paper_log(read_paper_log(), results))
    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M UTC")
    print(f"[{stamp}] fixtures checked: {len(window)} | new flags logged: {added}")
    print(f"  logged {sm['logged']} | settled {sm['settled']} | "
          f"profit {sm['profit']:+.2f}u | ROI "
          + ("n/a" if np.isnan(sm["roi"]) else f"{sm['roi']:+.1f}%")
          + " | avg CLV " + ("n/a" if np.isnan(sm["clv"]) else f"{sm['clv']:+.2f}%"))
    return 0


if __name__ == "__main__":
    if "--log" in sys.argv:
        sys.exit(run_headless_log(sys.argv[1:]))
    main()
