"""
Self-tests for the Football Value Dashboard.

They run automatically on GitHub after every change (see
.github/workflows/tests.yml) and never touch the internet: every data source is
replaced by simulated data saved in the backup-copy format.

Run by hand with:  pip install pytest  &&  pytest -q tests
"""
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# --- offline, isolated environment (must be set before the app is imported) --
TMP = Path(tempfile.mkdtemp(prefix="football_selftest_"))
os.environ["EPL_OFFLINE"] = "1"
os.environ["EPL_SNAPSHOT_DIR"] = str(TMP / "snap")
os.environ["EPL_PAPER_LOG"] = str(TMP / "paper_bets.csv")
os.environ["EPL_PREDICTIONS"] = str(TMP / "predictions.md")
os.environ["EPL_PICK_LOG"] = str(TMP / "best_picks_log.csv")

APP_PATH = Path(__file__).resolve().parents[1] / "epl_value_dashboard.py"
sys.path.insert(0, str(APP_PATH.parent))
import epl_value_dashboard as app  # noqa: E402

from scipy.stats import poisson  # noqa: E402

RNG = np.random.default_rng(2024)
TODAY = pd.Timestamp.now(tz="UTC").tz_convert(None).normalize()


# ----------------------------------------------------------------------------
# Simulated data
# ----------------------------------------------------------------------------
def _probs(lh, la):
    m = np.outer(poisson.pmf(range(11), lh), poisson.pmf(range(11), la))
    m /= m.sum()
    t = np.add.outer(range(11), range(11))
    return np.tril(m, -1).sum(), np.trace(m), np.triu(m, 1).sum(), m[t > 2.5].sum()


def _odds(p, margin, noise=0.06):
    return round(1 / (p * margin * np.exp(RNG.normal(0, noise))), 2)


def make_league(code, teams, strength, n_per_season=300):
    rows = []
    seasons = app.season_codes(TODAY.date(), 3)[::-1]          # oldest first
    for si, season in enumerate(seasons):
        start = TODAY - pd.Timedelta(days=365 * (len(seasons) - 1 - si) + 200)
        n = n_per_season if si < len(seasons) - 1 else 60
        for k in range(n):
            h, a = RNG.choice(len(teams), 2, replace=False)
            lh = np.exp(0.4 + strength[h] - 0.5 * strength[a])
            la = np.exp(0.15 + strength[a] - 0.5 * strength[h])
            pH, pD, pA, pO = _probs(lh, la)
            hst, ast = RNG.poisson(lh * 3), RNG.poisson(la * 3)
            rows.append({
                "Date": start + pd.Timedelta(days=int(k * 190 / n)), "HomeTeam": teams[h],
                "AwayTeam": teams[a], "FTHG": int(RNG.binomial(hst, .33)),
                "FTAG": int(RNG.binomial(ast, .33)), "HST": hst, "AST": ast, "Season": season,
                "B365H": _odds(pH, 1.05), "B365D": _odds(pD, 1.05), "B365A": _odds(pA, 1.05),
                "B365>2.5": _odds(pO, 1.05), "B365<2.5": _odds(1 - pO, 1.05),
                "B365CH": _odds(pH, 1.04, .02), "B365CD": _odds(pD, 1.04, .02),
                "B365CA": _odds(pA, 1.04, .02), "B365C>2.5": _odds(pO, 1.04, .02),
                "B365C<2.5": _odds(1 - pO, 1.04, .02), "League": code})
    df = pd.DataFrame(rows)
    return df[df["Date"] <= TODAY].sort_values("Date").reset_index(drop=True)


E0_TEAMS = ["Arsenal", "Chelsea", "Liverpool", "Man City", "Man United", "Tottenham",
            "Newcastle", "Aston Villa", "Brighton", "West Ham", "Everton", "Fulham"]
E0_STRENGTH = np.linspace(0.4, -0.4, len(E0_TEAMS))
LEAGUE_DATA = {
    "E0": make_league("E0", E0_TEAMS, E0_STRENGTH),
    "E1": make_league("E1", [f"Champ{i}" for i in range(12)], RNG.normal(0, .3, 12)),
    "D1": make_league("D1", ["Bayern Munich", "Dortmund", "Leverkusen", "RB Leipzig",
                             "Stuttgart", "Freiburg", "Mainz", "Augsburg"], RNG.normal(0, .3, 8)),
}


def make_fixtures():
    rows = []
    for code, df in LEAGUE_DATA.items():
        teams = sorted(set(df["HomeTeam"]))
        for i in range(0, 6, 2):
            ko = TODAY + pd.Timedelta(days=1 + i // 2, hours=15)
            rows.append({"Div": code, "Date": ko.normalize(), "Time": "15:00",
                         "HomeTeam": teams[i], "AwayTeam": teams[i + 1],
                         "Kickoff": ko.tz_localize("UTC"), "B365H": 2.6, "B365D": 3.4,
                         "B365A": 2.9, "B365>2.5": 1.95, "B365<2.5": 1.9})
    return pd.DataFrame(rows)


def make_uefa():
    clubs = ["Arsenal", "Chelsea", "Real Madrid", "Barcelona", "Salzburg", "Basel"]
    rows = []
    for k in range(160):
        h, a = RNG.choice(len(clubs), 2, replace=False)
        ko = TODAY - pd.Timedelta(days=600 - 3 * k)
        played = ko < TODAY
        rows.append({"Comp": "UCL", "Season": 2025, "Kickoff": ko.tz_localize("UTC"),
                     "HomeTeam": clubs[h], "AwayTeam": clubs[a],
                     "FTHG": RNG.poisson(1.5) if played else np.nan,
                     "FTAG": RNG.poisson(1.1) if played else np.nan,
                     "Round": 1, "Neutral": False, "Date": ko.normalize()})
    rows.append({"Comp": "UCL", "Season": 2026,
                 "Kickoff": (TODAY + pd.Timedelta(days=2, hours=19)).tz_localize("UTC"),
                 "HomeTeam": "Real Madrid", "AwayTeam": "Salzburg", "FTHG": np.nan,
                 "FTAG": np.nan, "Round": 2, "Neutral": False, "Date": TODAY})
    return pd.DataFrame(rows)


def make_international():
    nations = ["Cameroon", "Nigeria", "Senegal", "Ivory Coast", "England", "France",
               "Spain", "Brazil"]
    rows = []
    for k in range(900):
        h, a = RNG.choice(len(nations), 2, replace=False)
        tour = ("African Cup of Nations" if h < 4 and a < 4 else
                "UEFA Nations League" if h in (4, 5, 6) and a in (4, 5, 6) else "Friendly")
        rows.append({"Date": TODAY - pd.Timedelta(days=3500 - 3 * k),
                     "HomeTeam": nations[h], "AwayTeam": nations[a],
                     "FTHG": RNG.poisson(1.3), "FTAG": RNG.poisson(1.0),
                     "Tournament": tour, "Neutral": bool(RNG.random() < .3)})
    return pd.DataFrame(rows)


def make_intl_fixtures():
    ko = (TODAY + pd.Timedelta(days=3, hours=18)).tz_localize("UTC")
    return pd.DataFrame([{"Comp": "AFCON", "Season": 2027, "Kickoff": ko,
                          "HomeTeam": "Cameroon", "AwayTeam": "Ivory Coast", "FTHG": np.nan,
                          "FTAG": np.nan, "Round": 1, "Neutral": True, "Date": ko.normalize()}])


def build_backup():
    """Write every simulated source in the backup-copy format."""
    for code, df in LEAGUE_DATA.items():
        assert app.save_snapshot(f"results_{code}", df)
    assert app.save_snapshot("fixtures", make_fixtures())
    assert app.save_snapshot("uefa", make_uefa())
    assert app.save_snapshot("international", make_international())
    assert app.save_snapshot("intl_fixtures", make_intl_fixtures())


build_backup()


# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------
def test_probabilities_are_consistent():
    m = app.score_matrix(1.6, 1.1, -0.05)
    p = app.market_probs(m)
    assert abs(m.sum() - 1) < 1e-9
    assert abs(p["H"] + p["D"] + p["A"] - 1) < 1e-9
    assert abs(p["O25"] + p["U25"] - 1) < 1e-9
    assert p["H"] > p["A"]


def test_model_recovers_team_strength():
    model = app._fit_model(LEAGUE_DATA["E0"], 400, 0.5)
    assert model["converged"]
    fitted = [model["attack"][t] + model["defence"][t] for t in E0_TEAMS]
    assert np.corrcoef(E0_STRENGTH, fitted)[0, 1] > 0.8
    assert 1.0 < np.exp(model["home"]) < 1.6


def test_value_edge_and_stakes():
    rows = app.value_rows({"H": 0.5, "D": 0.3, "A": 0.2}, {"H": 2.2, "D": 3.0}, ["H", "D", "A"])
    assert abs(rows[0]["Edge %"] - 10.0) < 1e-6          # 0.5 x 2.2 - 1
    assert abs(rows[1]["Edge %"] - (-10.0)) < 1e-6
    assert np.isnan(rows[2]["Edge %"])                     # no odds -> no edge
    assert app.is_value(5, 3, 20) and not app.is_value(25, 3, 20) and not app.is_value(2, 3, 20)
    assert app.stake_pct(50, 2.0) == app.DEFAULTS["max_stake_pct"]   # capped
    assert 0 < app.stake_pct(5, 2.0) < app.DEFAULTS["max_stake_pct"]


def test_league_status_and_safety_switch():
    assert app.league_status("E0") == "bet"
    assert app.league_status("E1") == "watch"
    assert app.league_status("D1") == "predict"
    assert app.league_status("UCL") == "cup" and app.league_status("AFCON") == "cup"
    bad = pd.DataFrame({"League": ["E0"] * 160, "CLV %": -0.5,
                        "Profit": np.tile([-1.0, 0.8], 80)})
    over = app.safety_overrides(bad)
    assert over["E0"][0] == "watch"
    app.STATUS_OVERRIDES.update(over)
    try:
        assert app.league_status("E0") == "watch"
    finally:
        app.STATUS_OVERRIDES.clear()
    good = pd.DataFrame({"League": ["E0"] * 160, "CLV %": 1.5, "Profit": 0.1})
    assert app.safety_overrides(good) == {}
    assert app.safety_overrides(bad.head(50)) == {}            # too few bets to judge


def test_backup_copy_round_trip_and_fallback():
    df = app.load_snapshot("results_E0")
    assert len(df) == len(LEAGUE_DATA["E0"])
    assert all(isinstance(x, str) for x in df["Season"])      # season codes stay text
    got = app._live_or_backup("results_E0", lambda: pd.DataFrame(), prefer_backup=False)
    assert not got.empty and got.attrs["source"].startswith("backup")
    fx = app.load_snapshot("fixtures")
    assert str(fx["Kickoff"].dt.tz) == "UTC"


def test_paper_log_no_duplicates_and_settlement():
    model = app._fit_model(LEAGUE_DATA["E0"], 400, 0.5)
    fixtures = app.load_snapshot("fixtures")
    window = app.upcoming_fixtures(fixtures)
    flags = app.flagged_bets(window[window["Div"] == "E0"], model, "Bet365", 0.0, 100.0)
    assert not flags.empty
    log1, added1, err1 = app.update_paper_log(flags)
    log2, added2, err2 = app.update_paper_log(flags)
    assert err1 is None and added1 == len(flags) and added2 == 0
    first = flags.iloc[0]
    played = pd.DataFrame([{"Date": pd.Timestamp(first["Date"]), "HomeTeam": first["HomeTeam"],
                            "AwayTeam": first["AwayTeam"], "FTHG": 2, "FTAG": 1,
                            "B365CH": 2.4, "B365CD": 3.5, "B365CA": 3.0,
                            "B365C>2.5": 1.9, "B365C<2.5": 1.95}])
    settled = app.settle_paper_log(app.read_paper_log(), {"E0": played})
    row = settled.iloc[0]
    assert row["Score"] == "2-1"
    won = {"H": True, "D": False, "A": False, "O25": True, "U25": False}[row["Market"]]
    assert row["Profit"] == (row["Net odds"] - 1 if won else -1.0)
    assert not np.isnan(row["CLV %"])


def test_team_search():
    names = ["Man United", "Man City", "Paris SG", "Ivory Coast", "Cameroon", "Niger",
             "Nigeria", "Real Madrid", "Inter", "Inter Escaldes"]
    assert app.resolve_team("Man Utd", names)[0] == "Man United"
    assert app.resolve_team("PSG", names)[0] == "Paris SG"
    assert app.resolve_team("Côte d'Ivoire", names)[0] == "Ivory Coast"
    assert app.resolve_team("Cameroun", names)[0] == "Cameroon"
    assert app.resolve_team("Niger", names)[0] == "Niger"
    assert app.resolve_team("inter milan", names)[0] == "Inter"
    best, sugg, exact = app.resolve_team("real madird", names)
    assert best == "Real Madrid" and not exact                  # typo -> must confirm
    assert app.resolve_team("xyzzy", names)[0] is None


def test_match_query_picks_the_right_model():
    models = {c: app._fit_model(df, 400, 0.5) for c, df in LEAGUE_DATA.items()}
    uefa = app.load_snapshot("uefa")
    um = app.fit_uefa(uefa, app.uefa_domestic(sorted(set(uefa.HomeTeam) | set(uefa.AwayTeam)),
                                              models))
    im = app.fit_international(app.load_snapshot("international"))
    d = app.team_directory(models, um, im)
    fx = app.load_snapshot("fixtures")
    r = app.match_query("Arsenal", "Chelsea", d, models, um, im, fx, "Bet365", 3, 20, 100,
                        False, False)
    assert r["league"] == "E0" and r["status"] == "bet"
    r = app.match_query("Cameroon", "Nigeria", d, models, um, im, fx, "Bet365", 3, 20, 100,
                        False, True)
    assert r["competition"] == "National teams"
    r = app.match_query("Real Madrid", "Salzburg", d, models, um, im, fx, "Bet365", 3, 20,
                        100, False, False)
    assert r["status"] == "cup"
    r = app.match_query("Arsenal", "Cameroon", d, models, um, im, fx, "Bet365", 3, 20, 100,
                        False, False)
    assert "error" in r


def test_scheduled_run_offline_writes_day_by_day_file():
    code = app.run_headless_log(["--log", "--no-backup"])
    assert code == 0
    text = Path(os.environ["EPL_PREDICTIONS"]).read_text(encoding="utf-8")
    assert "day by day" in text and "Before you place any bet" in text
    assert "Africa Cup of Nations" in text                     # AFCON fixture included


def test_data_health_reports_every_source():
    res = {c: app.load_snapshot(f"results_{c}") for c in LEAGUE_DATA}
    h = app.data_health(res, app.load_snapshot("fixtures"), app.load_snapshot("uefa"),
                        app.load_snapshot("international"), app.load_snapshot("intl_fixtures"))
    assert set(h["Status"]) <= {"✅", "⚠️", "❌"}
    assert any("Backup copy" in s for s in h["Source"])
    assert len(h) >= len(LEAGUE_DATA) + 4


def test_full_app_runs_without_errors():
    """Start the real Streamlit app on the simulated data and check that no
    tab crashes (skipped automatically if Streamlit's test tools are missing)."""
    try:
        from streamlit.testing.v1 import AppTest
    except Exception:                                          # pragma: no cover
        import pytest
        pytest.skip("streamlit.testing not available")
    at = AppTest.from_file(str(APP_PATH), default_timeout=600)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    errors = [e.value for e in at.error if "Something went wrong" in str(e.value)]
    assert not errors, errors


def test_extra_league_format_and_fixtures():
    """Russia / Japan / China use Football-Data's second file layout."""
    import io
    results_csv = (
        "Country,League,Season,Date,Time,Home,Away,HG,AG,Res,PSCH,PSCD,PSCA,MaxCH,MaxCD,"
        "MaxCA,AvgCH,AvgCD,AvgCA,BFECH,BFECD,BFECA,B365CH,B365CD,B365CA\n"
        + "\n".join(f"Japan,J1 League,2026/2027,{(TODAY - pd.Timedelta(days=60 - i)):%d/%m/%Y},"
                    f"10:00,Team{i % 6},Team{(i + 1) % 6},{i % 3},{(i + 1) % 2},H,,,,2.1,3.4,3.6,"
                    "2.0,3.3,3.5,2.1,3.5,3.7,2.0,3.3,3.5" for i in range(40)))
    fixtures_csv = (
        "Country,League,Date,Time,Home,Away,PSH,PSD,PSA,MaxH,MaxD,MaxA,AvgH,AvgD,AvgA,"
        "BFEH,BFED,BFEA,B365H,B365D,B365A\n"
        f"Japan,J1 League,{(TODAY + pd.Timedelta(days=2)):%d/%m/%Y},10:00,Team1,Team2,,,,"
        "2.2,3.4,3.4,2.1,3.3,3.3,2.2,3.5,3.5,2.1,3.3,3.3\n"
        f"Brazil,Serie A,{(TODAY + pd.Timedelta(days=2)):%d/%m/%Y},22:00,X,Y,,,,2,3,4,2,3,4,"
        "2,3,4,2,3,4\n")
    original = app._fetch_csv

    def fake(url):
        if url.endswith("/new/JPN.csv"):
            return pd.read_csv(io.StringIO(results_csv))
        if url.endswith("new_league_fixtures.csv"):
            return pd.read_csv(io.StringIO(fixtures_csv))
        return pd.DataFrame()
    app._fetch_csv = fake
    try:
        res = app._load_results_uncached("JPN", 2)
        assert len(res) == 40 and set(res["League"]) == {"JPN"}
        assert all(isinstance(x, str) for x in res["Season"])
        assert "B365CH" in res.columns
        fx = app._load_fixtures_live()
        assert list(fx["Div"]) == ["JPN"]                      # Brazil is not covered
        assert fx.iloc[0]["HomeTeam"] == "Team1" and fx.iloc[0]["B365H"] == 2.1
        assert app.league_status("JPN") == "predict" and app.league_status("RUS") == "predict"
    finally:
        app._fetch_csv = original


def test_search_prefers_league_over_uefa_spelling():
    """'Man Utd' (UEFA feed) and 'Man United' (league data) are the same club:
    two league clubs must always be treated as a league match."""
    models = {"E0": app._fit_model(LEAGUE_DATA["E0"], 400, 0.5)}
    fake_uefa = {"teams": ["Man Utd", "Arsenal", "Salzburg"]}
    d = app.team_directory(models, fake_uefa, None)
    assert "Man Utd" not in d and ("uefa", "Man Utd") in d["Man United"]
    assert app.resolve_team("Man Utd", list(d))[0] == "Man United"
    r = app.match_query("Man United", "Arsenal", d, models, fake_uefa, None, pd.DataFrame(),
                        "Bet365", 3, 20, 100, False, False)
    assert r["league"] == "E0"


def test_gap_fill_adds_newer_results_from_backup_site():
    """When the main source is behind, newer results come from fixturedownload,
    with the feed's club names mapped to the main source's spelling."""
    season = app.season_codes(TODAY.date(), 0)[0]
    last = TODAY - pd.Timedelta(days=20)
    df = pd.DataFrame({"Date": [last, last], "HomeTeam": ["Man United", "Tottenham"],
                       "AwayTeam": ["Wolves", "Nott'm Forest"], "FTHG": [1, 2], "FTAG": [0, 2],
                       "Season": season, "League": "E0"})

    def iso(days_ago):
        return (TODAY - pd.Timedelta(days=days_ago)).strftime("%Y-%m-%d 14:00:00Z")

    feed = [
        {"DateUtc": iso(20), "HomeTeam": "Man Utd", "AwayTeam": "Wolves",
         "HomeTeamScore": 1, "AwayTeamScore": 0},                     # already known
        {"DateUtc": iso(5), "HomeTeam": "Spurs", "AwayTeam": "Man Utd",
         "HomeTeamScore": 3, "AwayTeamScore": 1},                     # new -> added
        {"DateUtc": iso(4), "HomeTeam": "Nottingham Forest", "AwayTeam": "Wolves",
         "HomeTeamScore": 0, "AwayTeamScore": 0},                     # new -> added
        {"DateUtc": iso(3), "HomeTeam": "Unknown FC", "AwayTeam": "Wolves",
         "HomeTeamScore": 2, "AwayTeamScore": 2},                     # unmatched -> skipped
        {"DateUtc": iso(-3), "HomeTeam": "Wolves", "AwayTeam": "Spurs",
         "HomeTeamScore": None, "AwayTeamScore": None},               # not played
    ]
    orig = app._fetch_json
    app._fetch_json = lambda url: feed
    try:
        out = app.gap_fill_results(df, "E0")
    finally:
        app._fetch_json = orig
    assert len(out) == 4
    new = out[out["Filled"] == True]  # noqa: E712
    assert set(zip(new["HomeTeam"], new["AwayTeam"])) == {
        ("Tottenham", "Man United"), ("Nott'm Forest", "Wolves")}
    assert (new["Season"] == season).all() and (new["League"] == "E0").all()
    # a broken feed never breaks loading
    app._fetch_json = lambda url: (_ for _ in ()).throw(RuntimeError("down"))
    try:
        assert len(app.gap_fill_results(df, "E0")) == 2
    finally:
        app._fetch_json = orig


def test_schedule_adds_matches_the_odds_feed_has_not_published():
    """Weekend league games appear before the odds feed publishes them,
    with every goal line, and never as a BET (no odds yet)."""
    res = LEAGUE_DATA["E0"]
    teams = sorted(set(res["HomeTeam"]))
    h, a, h2, a2 = teams[:4]
    soon = (pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=3)).strftime("%Y-%m-%d 14:00:00Z")
    feed = [{"DateUtc": soon, "HomeTeam": h, "AwayTeam": a, "HomeTeamScore": None,
             "AwayTeamScore": None},
            {"DateUtc": soon, "HomeTeam": h2, "AwayTeam": a2, "HomeTeamScore": None,
             "AwayTeamScore": None}]
    orig = app._fetch_json
    app._fetch_json = lambda url: feed if "/epl-" in url else None
    try:
        sched = app._load_schedule_live()
    finally:
        app._fetch_json = orig
    assert len(sched) == 2 and set(sched["Div"]) == {"E0"}
    # the odds feed already has the first match -> only the second is added
    fx = make_fixtures().head(0)
    known = pd.DataFrame({"Div": ["E0"], "HomeTeam": [h], "AwayTeam": [a],
                          "Date": [pd.Timestamp(soon[:10])],
                          "Kickoff": [pd.Timestamp(soon)], "B365H": [2.0]})
    out = app.add_schedule(pd.concat([fx, known], ignore_index=True), sched, {"E0": res})
    assert len(out) == 2
    added = out[out["NoOdds"] == True]  # noqa: E712
    assert list(zip(added["HomeTeam"], added["AwayTeam"])) == [(h2, a2)]
    adv, flags, _ = app.weekly_overview(out, {"E0": res}, 400, 0.5, "Bet365", 3, 20, 100)
    row = adv[adv["Match"] == f"{h2} v {a2}"].iloc[0]
    assert row["Advice"].startswith("Odds not out yet")
    for c in app.GOAL_COLS:
        assert 0 <= row[c] <= 100
    assert abs(row["Over 0.5 %"] + row["Under 0.5 %"] - 100) < 1e-6
    overs = [row[f"Over {g} %"] for g in app.GOAL_LINES]
    assert app.GOAL_LINES[-1] == "6.5" and overs == sorted(overs, reverse=True)
    assert abs(row["Over 6.5 %"] + row["Under 6.5 %"] - 100) < 1e-6


def test_phone_cards_show_every_match_safely():
    """Phone layout: one card per match, with advice, odds lines, and escaped names."""
    adv = app.cup_predictions(make_intl_fixtures(), app.fit_international(make_international()),
                              "Comp", 14, 3)
    adv.loc[0, "Match"] = "<b>A</b> v B&C"
    adv.loc[0, "Advice"] = "BET Over 2.5 @ 2.10"
    d = app.add_local_time(adv, "Africa/Douala")
    out = app.match_cards_html(d)
    assert out.count("class='bfe-card") == len(adv)
    assert "bfe-card bet" in out and "&lt;b&gt;A&lt;/b&gt; v B&amp;C" in out
    assert all(f"<th>{g}</th>" in out for g in app.GOAL_LINES)
    assert not app.phone_mode() or True          # never raises outside a browser


def test_best_pick_is_most_likely_option_worth_betting():
    model = app._fit_model(LEAGUE_DATA["E0"], 400, 0.5)
    _, _, _, p = app.predict(model, "Arsenal", "Fulham")
    assert abs(p["1X"] - (p["H"] + p["D"])) < 1e-9 and abs(p["12"] + p["D"] - 1) < 1e-9
    bp = app.best_pick(p, "Arsenal", "Fulham")
    allowed = [k for k in app.BEST_PICK_KEYS if p[k] <= app.BEST_PICK_MAX_P]
    assert bp["key"] in allowed and bp["prob"] == max(p[k] for k in allowed)
    assert bp["floor"] >= 1 / app.BEST_PICK_MAX_P and "bet at" in bp["text"]
    assert bp["tier"] in ("Banker", "Strong", "Good", "Fair")
    assert abs(p["HS"] - (1 - app.predict(model, "Arsenal", "Fulham")[2][0, :].sum())) < 1e-9
    assert "HS" in app.BEST_PICK_KEYS and app.best_pick(
        {**{k: 0.1 for k in app.BEST_PICK_KEYS}, "HS": 0.91}, "Arsenal", "Fulham")["label"] \
        == "Arsenal to score"
    # with bookmaker odds the pick blends model and market
    odds = {"H": 1.5, "D": 4.2, "A": 6.5, "O25": 1.7, "U25": 2.2}
    m = app.predict(model, "Arsenal", "Fulham")[2]
    pb, used = app.pick_probs(m, model["rho"], odds)
    assert used and abs(sum(pb[k] for k in ("H", "D", "A")) - 1) < 1e-6
    assert app.pick_probs(m, model["rho"], {})[1] is False
    rows = app.score_predictions(make_fixtures().head(2), model)
    assert rows["Best pick"].str.contains("bet at").all()


def test_every_best_pick_has_plain_english_instructions():
    for k in app.BEST_PICK_KEYS:
        txt = app.how_to_bet(k, "LASK", "Liverpool", 1.15)
        assert "WIN" in txt and "1.15 or higher" in txt
    assert "'12'" in app.how_to_bet("12", "LASK", "Liverpool", 1.15)
    assert "LOSE only if the match is a draw" in app.how_to_bet("12", "LASK", "Liverpool", 1.15)


def test_cup_and_international_picks_are_corrected_for_overconfidence():
    p = {"A": 0.9, "B": 0.5, "C": 0.1}
    q = app.shrink_probs(p, 0.86)
    assert abs(q["A"] - 0.844) < 1e-9 and q["B"] == 0.5 and app.shrink_probs(p, 1.0) is p
    im = app.fit_international(make_international())
    assert im["pick_shrink"] == app.INTL_PICK_SHRINK < 1
    rows = app.cup_predictions(make_intl_fixtures(), im, "Comp", 14, 3)
    _, _, _, raw = app.predict(im, "Cameroon", "Ivory Coast", neutral=True)
    shr = app.shrink_probs(raw, im["pick_shrink"])
    assert abs(rows["Best pick %"].iloc[0] / 100 - app.best_pick(shr, "Cameroon", "Ivory Coast")["prob"]) < 1e-9


def test_pick_won_covers_every_market():
    assert app.pick_won("U45", 2, 2) and not app.pick_won("U45", 3, 2)
    assert app.pick_won("O15", 1, 1) and not app.pick_won("O15", 1, 0)
    assert app.pick_won("12", 0, 1) and not app.pick_won("12", 1, 1)
    assert app.pick_won("AS", 3, 1) and not app.pick_won("HS", 0, 2)
    for k in app.BEST_PICK_KEYS:
        app.pick_won(k, 1, 0)


def test_best_pick_log_records_freezes_and_settles():
    if app.PICK_LOG.exists():
        app.PICK_LOG.unlink()
    model = app._fit_model(LEAGUE_DATA["E0"], 400, 0.5)
    fx = make_fixtures()
    fx = fx[fx["Div"] == "E0"].head(2).copy()
    adv = app.match_advice(app.score_predictions(fx, model), pd.DataFrame(), 100)
    adv.insert(0, "Competition", "England - Premier League")
    t0 = pd.Timestamp.now(tz="UTC")
    log, n, err = app.update_pick_log(adv, now=t0)
    assert err is None and n == 2 and len(log) == 2
    # before kick-off a newer pick replaces the old one (no duplicates)
    log, n, _ = app.update_pick_log(adv, now=t0)
    assert n == 0 and len(app.read_pick_log()) == 2
    # after kick-off the pick is frozen and then settled from the result
    first = app.read_pick_log().iloc[0]
    later = pd.Timestamp(first["Kickoff (UTC)"], tz="UTC") + pd.Timedelta(hours=3)
    changed = adv.copy()
    changed["Best pick"] = "changed"
    app.update_pick_log(changed, now=later)
    assert app.read_pick_log().iloc[0]["Best pick"] == first["Best pick"]
    res = pd.DataFrame({"League": ["E0"], "Date": [pd.Timestamp(first["Match day"])],
                        "HomeTeam": [first["HomeTeam"]], "AwayTeam": [first["AwayTeam"]],
                        "FTHG": [2], "FTAG": [1]})
    st_ = app.settle_pick_log(app.read_pick_log(), res)
    row = st_.iloc[0]
    assert row["Score"] == "2-1" and row["Won"] == app.pick_won(first["Pick key"], 2, 1)
    sm = app.pick_summary(st_)
    assert sm["n"] == 1 and sm["pending"] == 1 and "best picks won" in app.pick_record_lines(sm)[0]
    day, top = app.safest_picks(adv, "Africa/Douala")
    assert day is not None


def test_top_five_picks_per_day_are_ranked_highest_first():
    model = app._fit_model(LEAGUE_DATA["E0"], 400, 0.5)
    rows = []
    for code, df in LEAGUE_DATA.items():
        m = app._fit_model(df, 400, 0.5)
        fx = make_fixtures()
        rows.append(app.match_advice(app.score_predictions(fx[fx["Div"] == code], m),
                                     pd.DataFrame(), 100))
    adv = pd.concat(rows, ignore_index=True)
    adv.insert(0, "Competition", adv["League"])
    days = app.top_picks_by_day(adv, "Africa/Douala", 5)
    assert days and all(len(t) <= 5 for _, t in days)
    for _, t in days:
        assert list(t["Rank"]) == list(range(1, len(t) + 1))
        assert t["Best pick %"].is_monotonic_decreasing
    html_ = app.match_cards_html(days[0][1])
    assert "#1</b>" in html_


def test_ticket_line_multiplies_chances():
    top = pd.DataFrame({"Best pick %": [90.0, 90.0, 90.0, 90.0, 90.0]})
    txt = app.ticket_line(top)
    assert "about **59%**" in txt and "All 5" in txt


def test_cup_odds_parsing_matching_and_blend():
    ko = (TODAY + pd.Timedelta(days=2, hours=19)).tz_localize("UTC")
    ev = [{"commence_time": ko.strftime("%Y-%m-%dT%H:%M:%SZ"), "home_team": "Real Madrid CF",
           "away_team": "FC Red Bull Salzburg", "bookmakers": [
               {"markets": [{"key": "h2h", "outcomes": [
                   {"name": "Real Madrid CF", "price": 1.30}, {"name": "Draw", "price": 6.0},
                   {"name": "FC Red Bull Salzburg", "price": 9.0}]},
                   {"key": "totals", "outcomes": [{"name": "Over", "price": 1.6, "point": 2.5},
                                                  {"name": "Under", "price": 2.3, "point": 2.5}]}]},
               {"markets": [{"key": "h2h", "outcomes": [
                   {"name": "Real Madrid CF", "price": 1.34}, {"name": "Draw", "price": 5.8},
                   {"name": "FC Red Bull Salzburg", "price": 8.6}]}]}]}]
    df = app.odds_events_to_frame(ev, "UCL")
    assert len(df) == 1 and abs(df["H"].iloc[0] - 1.32) < 1e-9 and df["O25"].iloc[0] == 1.6
    df["_ko"] = pd.to_datetime(df["Kickoff"], utc=True)
    fx = {"Kickoff": ko, "HomeTeam": "Real Madrid", "AwayTeam": "Salzburg"}
    assert len(app.match_fixture_odds(fx, df)) == 5
    # a different match at the same time is never matched
    assert app.match_fixture_odds({"Kickoff": ko, "HomeTeam": "Arsenal",
                                   "AwayTeam": "Basel"}, df) == {}
    # wrong day -> no match
    assert app.match_fixture_odds({**fx, "Kickoff": ko + pd.Timedelta(days=1)}, df) == {}
    # cup predictions use the bookmaker odds when they match
    im = app.fit_international(make_international())
    nfx = make_intl_fixtures()
    o = pd.DataFrame([{"Comp": "AFCON", "Kickoff": nfx["Kickoff"].iloc[0], "HomeTeam": "Cameroon",
                       "AwayTeam": "Ivory Coast", "H": 2.6, "D": 3.0, "A": 2.9,
                       "O25": 2.2, "U25": 1.7}])
    o["_ko"] = pd.to_datetime(o["Kickoff"], utc=True)
    with_odds = app.cup_predictions(nfx, im, "Comp", 14, 3, o)
    without = app.cup_predictions(nfx, im, "Comp", 14, 3, None)
    assert with_odds["Advice"].iloc[0].startswith("Bookmakers:")
    assert without["Advice"].iloc[0].startswith("No odds feed")


def test_odds_refresh_without_key_spends_nothing():
    st_ = app.refresh_cup_odds("")
    assert st_["key_present"] is False and "no ODDS_API_KEY" in st_["result"]


def _fake_injury_table(home="Arsenal", away="Chelsea", api_home="Arsenal FC",
                       api_away="Chelsea FC"):
    ko = (TODAY + pd.Timedelta(days=1, hours=15)).tz_localize("UTC")
    t = pd.DataFrame([
        {"FixtureId": 1, "Kickoff": ko.isoformat(), "HomeId": 10, "AwayId": 20,
         "HomeTeam": api_home, "AwayTeam": api_away, "Season": 2026, "League": "E0",
         "TeamId": 10, "PlayerId": 100, "Player": "Top Scorer", "Type": "Missing Fixture",
         "Reason": "Hamstring", "Position": "Attacker", "Apps": 8, "Starts": 8,
         "Minutes": 700, "Goals": 6, "Assists": 2, "Side": "H"},
        {"FixtureId": 1, "Kickoff": ko.isoformat(), "HomeId": 10, "AwayId": 20,
         "HomeTeam": api_home, "AwayTeam": api_away, "Season": 2026, "League": "E0",
         "TeamId": 20, "PlayerId": 200, "Player": "Keeper", "Type": "Missing Fixture",
         "Reason": "Suspended", "Position": "Goalkeeper", "Apps": 8, "Starts": 8,
         "Minutes": 720, "Goals": 0, "Assists": 0, "Side": "A"}])
    t["_ko"] = pd.to_datetime(t["Kickoff"], utc=True)
    return t, ko


def test_missing_players_lower_attack_and_raise_leakiness():
    t, ko = _fake_injury_table()
    att, dfn, text, key = app.injury_effect(t[t["Side"] == "H"], 1.6)
    assert -app.INJ_ATT_CAP <= att < 0 and dfn == 0 and key == 1 and "Top Scorer" in text
    att2, dfn2, text2, _ = app.injury_effect(t[t["Side"] == "A"], 1.2)
    assert att2 == 0 and dfn2 == app.INJ_DEF_BUMP["Goalkeeper"] and "first-choice GK" in text2
    model = app._fit_model(LEAGUE_DATA["E0"], 400, 0.5)
    fx = {"Kickoff": ko, "HomeTeam": "Arsenal", "AwayTeam": "Chelsea"}
    adj, txt = app.fixture_injuries(fx, t, model)
    assert adj["h_att"] < 0 and adj["a_def"] > 0 and "Arsenal:" in txt and "Chelsea:" in txt
    base = app.predict(model, "Arsenal", "Chelsea")
    hit = app.predict(model, "Arsenal", "Chelsea", adj)
    assert abs(hit[0] - base[0] * (1 + adj["h_att"]) * (1 + adj["a_def"])) < 1e-9
    assert abs(hit[1] - base[1]) < 1e-9
    # rows carry the text; unknown fixture -> no adjustment
    fxs = make_fixtures()
    rows = app.score_predictions(fxs[fxs["Div"] == "E0"].head(1), model, t)
    assert "Missing players" in rows.columns
    assert app.fixture_injuries({"Kickoff": ko, "HomeTeam": "Everton", "AwayTeam": "Fulham"},
                                t, model) == (None, "")


def test_fixture_matching_handles_long_club_names():
    t, ko = _fake_injury_table(api_home="Wolverhampton Wanderers", api_away="Chelsea FC")
    assert app.match_fixture_row({"Kickoff": ko, "HomeTeam": "Wolves", "AwayTeam": "Chelsea"},
                                 t) is not None
    assert app.match_fixture_row({"Kickoff": ko, "HomeTeam": "Everton", "AwayTeam": "Brighton"},
                                 t) is None


def test_injury_refresh_without_key_spends_nothing():
    assert "no APIFOOTBALL_KEY" in app.refresh_injuries("")["result"]


def test_ready_to_bet_filter_removes_risky_picks():
    base = {"Best pick %": 90.0, "Pick key": "1X", "Odds used": True, "Key out H": 0,
            "Key out A": 0}
    assert app.ready_filter(base) == ""
    assert "below 85%" in app.ready_filter({**base, "Best pick %": 84.0})
    assert "no bookmaker odds" in app.ready_filter({**base, "Odds used": False, "Best pick %": 86.0})
    assert app.ready_filter({**base, "Odds used": False, "Best pick %": 89.0}) == ""
    assert "key player" in app.ready_filter({**base, "Key out H": 1})
    assert app.ready_filter({**base, "Key out A": 1}) == ""          # backing home: fine
    assert "key player" in app.ready_filter({**base, "Pick key": "AS", "Key out A": 1})
    assert "goals bet" in app.ready_filter({**base, "Pick key": "O15", "Key out H": 1,
                                            "Key out A": 1})
    assert app.short_bet("12", "LASK", "Liverpool") == "Double Chance → 12 (no draw)"
    assert app.short_bet("HS", "Brann", "X") == "Brann total goals → Over 0.5"
    assert app.short_bet("U45", "A", "B").endswith("Under 4.5")


def test_ready_picks_by_day_only_contains_passing_picks():
    model = app._fit_model(LEAGUE_DATA["E0"], 400, 0.5)
    fx = make_fixtures()
    adv = app.match_advice(app.score_predictions(fx[fx["Div"] == "E0"], model),
                           pd.DataFrame(), 100)
    adv.insert(0, "Competition", "England - Premier League")
    adv["Best pick %"] = 90.0
    adv.loc[adv.index[0], "Key out H"] = 2
    adv.loc[adv.index[0], "Pick key"] = "1X"
    days = app.ready_picks_by_day(adv, "Africa/Douala", 5)
    allrows = pd.concat([t for _, t in days])
    assert adv["Match"].iloc[0] not in set(allrows["Match"])
    assert allrows["Bet"].str.len().gt(0).all() and (allrows["Min odds"] > 1).all()
    assert "✅ BET:" in app.ready_cards_html(days[0][1])


def test_bookmaker_prices_parse_match_and_filter():
    item = {"fixture": {"id": 1}, "bookmakers": [
        {"name": "1xBet", "bets": [
            {"name": "Double Chance", "values": [{"value": "Home/Draw", "odd": "1.12"},
                                                 {"value": "Home/Away", "odd": "1.30"}]},
            {"name": "Goals Over/Under", "values": [{"value": "Under 4.5", "odd": "1.09"}]},
            {"name": "Total - Home", "values": [{"value": "Over 0.5", "odd": "1.15"}]},
            {"name": "Exact Score", "values": [{"value": "1:0", "odd": "7"}]}]},
        {"name": "Bet365", "bets": [
            {"name": "Double Chance", "values": [{"value": "Home/Draw", "odd": "1.14"}]}]}]}
    rows = pd.DataFrame(app.af_odds_rows(item))
    assert set(rows["Key"]) == {"1X", "12", "U45", "HS"} and len(rows) == 5
    t, ko = _fake_injury_table()
    fxs = t.drop_duplicates("FixtureId")[["FixtureId", "Kickoff", "HomeTeam", "AwayTeam", "_ko"]]
    bo = rows.merge(fxs, on="FixtureId")
    fx = {"Kickoff": ko, "HomeTeam": "Arsenal", "AwayTeam": "Chelsea"}
    assert app.book_prices(fx, bo, "1X") == (1.12, 1.14)
    assert all(np.isnan(v) for v in app.book_prices(fx, bo, "BTTS_Y"))
    base = {"Best pick %": 90.0, "Pick key": "1X", "Odds used": True, "Key out H": 0,
            "Key out A": 0}
    assert "1xBet pays only 1.12" in app.ready_filter({**base, "1xBet odds": 1.12})
    assert app.ready_filter({**base, "1xBet odds": 1.20}) == ""
    assert app.ready_filter({**base, "1xBet odds": np.nan}) == ""


def test_append_feed_uses_the_league_own_season_label():
    df = pd.DataFrame({"Date": [TODAY - pd.Timedelta(days=60)] * 2,
                       "HomeTeam": ["Zenit", "Spartak Moscow"], "AwayTeam": ["CSKA Moscow", "Rostov"],
                       "FTHG": [1, 2], "FTAG": [0, 2], "Season": "2026/2027", "League": "RUS"})
    feed = pd.DataFrame([{"DateUtc": (TODAY - pd.Timedelta(days=3)).strftime("%Y-%m-%dT16:00:00+00:00"),
                          "HomeTeam": "Zenit Saint Petersburg", "AwayTeam": "Rostov",
                          "HomeTeamScore": 3, "AwayTeamScore": 1}])
    out = app._append_feed(df, feed, "RUS", "API-Football")
    new = out[out["Filled"] == True]  # noqa: E712
    assert len(new) == 1 and new["Season"].iloc[0] == "2026/2027"
    assert new["HomeTeam"].iloc[0] == "Zenit" and new["FilledFrom"].iloc[0] == "API-Football"


def test_strongest_picks_use_the_same_safety_checks(tmp_path=None):
    model = app._fit_model(LEAGUE_DATA["E0"], 400, 0.5)
    fx = make_fixtures()
    adv = app.match_advice(app.score_predictions(fx[fx["Div"] == "E0"], model),
                           pd.DataFrame(), 100)
    adv.insert(0, "Competition", "England - Premier League")
    adv["Best pick %"] = 95.0
    adv["Pick key"] = "1X"
    adv["Key out H"] = 1                       # every pick fails the injury check
    out = TMP / "strong.md"
    app.write_predictions_md(adv, pd.DataFrame(), out)
    txt = out.read_text()
    sec = txt.split("## 🎯 Strongest best picks")[1].split("##")[0]
    assert "No 🔒 / ⭐⭐⭐ picks this week." in sec


def test_every_match_gets_a_bet_whatever_the_odds():
    model = app._fit_model(LEAGUE_DATA["E0"], 400, 0.5)
    fx = make_fixtures()
    adv = app.match_advice(app.score_predictions(fx[fx["Div"] == "E0"], model),
                           pd.DataFrame(), 100)
    adv.insert(0, "Competition", "England - Premier League")
    adv["1xBet odds"] = 1.01                   # terrible prices: excluded from 'Your bets'
    days = app.all_matches_by_day(adv, "Africa/Douala")
    allrows = pd.concat([g for _, g in days])
    assert len(allrows) == len(adv) and allrows["Bet"].str.len().gt(0).all()
    assert (allrows["Check"] != "").all()      # all flagged ⚠️, but still listed
    html_ = app.all_matches_cards_html(allrows)
    assert html_.count("BET:") == len(adv) and "⚠️ riskier" in html_


def test_past_days_show_the_bet_given_and_the_result():
    ko = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=26)).floor("h")
    log = pd.DataFrame([
        {"Kickoff (UTC)": ko.strftime("%Y-%m-%d %H:%M"), "Match day": ko.strftime("%Y-%m-%d"),
         "League": "E0", "Competition": "England - Premier League", "HomeTeam": "Arsenal",
         "AwayTeam": "Chelsea", "Pick key": "U45", "Best pick": "x", "Prob %": 90.0,
         "Bet at": 1.14, "Tier": "Banker", "Check": ""},
        {"Kickoff (UTC)": ko.strftime("%Y-%m-%d %H:%M"), "Match day": ko.strftime("%Y-%m-%d"),
         "League": "E0", "Competition": "England - Premier League", "HomeTeam": "Everton",
         "AwayTeam": "Fulham", "Pick key": "O15", "Best pick": "x", "Prob %": 84.0,
         "Bet at": 1.22, "Tier": "Fair", "Check": "below 85%"}])
    res = pd.DataFrame({"League": ["E0", "E0"], "Date": [ko.tz_localize(None)] * 2,
                        "HomeTeam": ["Arsenal", "Everton"], "AwayTeam": ["Chelsea", "Fulham"],
                        "FTHG": [2, 1], "FTAG": [1, 0]})
    settled = app.settle_pick_log(log, res)
    past = app.past_matches_by_day(settled, "UTC")
    g = list(past.values())[0]
    assert list(g["Result"]) == ["✅ WON 2-1", "❌ LOST 1-0"]
    assert list(g["Checked"]) == ["✅", "⚠️"]
    line = app.day_result_line(g)
    assert "1 of 2 bets won" in line and "checked bets: **1 of 1 won**" in line
    sm = app.pick_summary(settled)
    assert sm["checked"] == (1, 1)


def test_inplay_chance_updates_with_score_and_time():
    # Under 4.5 at 0-0 late in the game is near certain; at 3-1 with time left it is risky
    assert app.inplay_chance("U45", 1.5, 1.2, 85, 0, 0) > 0.99
    assert app.inplay_chance("U45", 1.5, 1.2, 30, 3, 1) < 0.6
    assert app.inplay_chance("HS", 1.5, 1.2, 10, 1, 0) == 1.0          # already won
    assert app.inplay_chance("U45", 1.5, 1.2, 60, 4, 1) == 0.0         # already lost
    pre = app.inplay_chance("1X", 1.5, 1.2, 0, 0, 0)
    assert app.inplay_chance("1X", 1.5, 1.2, 70, 1, 0) > pre > app.inplay_chance("1X", 1.5, 1.2, 70, 0, 1)
    live = pd.DataFrame([{"FixtureId": 9, "Kickoff": "2026-10-15T19:00:00+00:00",
                          "HomeTeam": "Arsenal FC", "AwayTeam": "Chelsea FC", "Minute": 63,
                          "Status": "2H", "GH": 1, "GA": 0}])
    live["_ko"] = pd.to_datetime(live["Kickoff"], utc=True)
    ko = pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=70)
    log = pd.DataFrame([{"Kickoff (UTC)": ko.strftime("%Y-%m-%d %H:%M"),
                         "Match day": ko.strftime("%Y-%m-%d"), "League": "E0",
                         "Competition": "PL", "HomeTeam": "Arsenal", "AwayTeam": "Chelsea",
                         "Pick key": "HS", "Best pick": "x", "Prob %": 90.0, "Bet at": 1.14,
                         "Tier": "Banker", "Check": "", "xG": "1.60 - 1.10"}])
    live["Kickoff"] = ko.isoformat()
    live["_ko"] = pd.to_datetime(live["Kickoff"], utc=True)
    st_ = app.settle_pick_log(log, pd.DataFrame())
    g = list(app.past_matches_by_day(st_, "UTC", live=live).values())[0]
    assert g["Result"].iloc[0].startswith("🔴 LIVE 63' · 1-0") and "already won" in g["Result"].iloc[0]
