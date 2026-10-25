"""
===============================================================================
 FOOTBALL VALUE DASHBOARD (25 leagues, UEFA cups + national teams)
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
import html
import io
import json
import logging
import os
import sys
import tempfile
import time
from contextlib import contextmanager
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
INTL_URL = ("https://raw.githubusercontent.com/martj42/"
            "international_results/master/results.csv")

# Leagues covered (Football-Data.co.uk main files): code -> (country, name)
LEAGUES = {
    "E0": ("England", "Premier League"), "E1": ("England", "Championship"),
    "E2": ("England", "League One"), "E3": ("England", "League Two"),
    "EC": ("England", "National League"),
    "SC0": ("Scotland", "Premiership"), "SC1": ("Scotland", "Championship"),
    "SC2": ("Scotland", "League One"), "SC3": ("Scotland", "League Two"),
    "D1": ("Germany", "Bundesliga"), "D2": ("Germany", "2. Bundesliga"),
    "I1": ("Italy", "Serie A"), "I2": ("Italy", "Serie B"),
    "SP1": ("Spain", "La Liga"), "SP2": ("Spain", "Segunda Division"),
    "F1": ("France", "Ligue 1"), "F2": ("France", "Ligue 2"),
    "N1": ("Netherlands", "Eredivisie"), "B1": ("Belgium", "Pro League"),
    "P1": ("Portugal", "Primeira Liga"), "T1": ("Turkey", "Super Lig"),
    "G1": ("Greece", "Super League"),
    # Second Football-Data format: all seasons in one file, closing odds only
    "RUS": ("Russia", "Premier League"), "JPN": ("Japan", "J1 League"),
    "CHN": ("China", "Super League"),
}
# (Country, League) as written in Football-Data's extra-leagues files -> code
NEW_FORMAT = {"RUS": ("Russia", "Premier League"), "JPN": ("Japan", "J1 League"),
              "CHN": ("China", "Super League")}
DEFAULT_LEAGUE = "E0"

# Betting status per league, from walk-forward backtests (edge 3-20%, Betfair
# Exchange prices, model refitted weekly on past matches only):
#   "bet"     - profitable in BOTH test seasons -> BET advice with stakes
#   "watch"   - about break-even -> advice logged for paper trading only
#   "predict" - lost money in testing -> predictions only, never BET advice
LEAGUE_STATUS = {"E0": "bet", "E1": "watch"}
# Automatic safety switch: a BET league drops to WATCH when paper trading shows
# no edge (filled at run time from the settled paper-trading log)
SAFETY = {"min_bets": 150, "min_avg_clv": 0.0}
STATUS_OVERRIDES: dict = {}
STATUS_LABEL = {"bet": "✅ Bet", "watch": "👀 Watch (paper only)",
                "predict": "📊 Predictions only",
                "cup": "🏆 Predictions only (no odds feed)"}
STATUS_RANK = {"bet": 0, "watch": 1, "predict": 2, "cup": 3}
# Flagged-bet ROI in testing: 2025/26 (and 2024/25 where tested)
LEAGUE_EVIDENCE = {
    "E0": "+16.7% (2025/26), +18.8% (2024/25)", "E1": "+5.2% (2025/26), +2.0% (2024/25)",
    "E2": "-4.5% (2025/26)", "E3": "-9.0% (2025/26)", "EC": "-4.5% (2025/26)",
    "SC0": "+8.7% (2025/26, 1 season only)", "SC1": "-4.5% (2025/26)",
    "SC2": "+1.3% (2025/26), -8.4% (2024/25)", "SC3": "-1.1% (2025/26)",
    "D1": "-17.0% (2025/26)", "D2": "-15.1% (2025/26)", "I1": "-6.1% (2025/26)",
    "I2": "-31.8% (2025/26)", "SP1": "-14.9% (2025/26)", "SP2": "-12.2% (2025/26)",
    "F1": "-16.1% (2025/26)", "F2": "-4.6% (2025/26)",
    "N1": "+5.7% (2025/26), -18.6% (2024/25)", "B1": "+2.3% (2025/26), -19.2% (2024/25)",
    "P1": "+3.2% (2025/26), -34.9% (2024/25)", "T1": "-11.9% (2025/26)",
    "G1": "-17.0% (2025/26), -21.4% (2024/25)",
    "RUS": "-15.3% (2023/24), -26.8% (2024/25), +7.7% (2025/26) - average closing odds",
    "JPN": "-23.6% (2025, Betfair closing odds)", "CHN": "-7.8% (2025, Betfair closing odds)",
}


CUP_CODES = set()          # filled below: every code that has no odds feed


def league_name(code: str) -> str:
    if code in UEFA_COMPS:
        return UEFA_COMPS[code][0]
    if code == NATIONS_LEAGUE[0]:
        return NATIONS_LEAGUE[1]
    for c, name, _ in NATION_TOURNAMENTS:
        if code == c:
            return name
    country, name = LEAGUES.get(code, ("", code))
    return f"{country} - {name}" if country else name


def league_status(code: str, allow_all: bool = False) -> str:
    if code in CUP_CODES:
        return "cup"                     # no odds feed -> predictions only
    if allow_all:
        return "bet"
    if code in STATUS_OVERRIDES:         # safety switch from paper-trading results
        return STATUS_OVERRIDES[code][0]
    return LEAGUE_STATUS.get(code, "predict")

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
    "timezone": "Africa/Douala",   # times in the day-by-day view
}
TIMEZONES = ["Africa/Douala", "Africa/Lagos", "Africa/Accra", "Africa/Abidjan",
             "Africa/Cairo", "Africa/Johannesburg", "Africa/Nairobi",
             "Europe/London", "Europe/Paris", "UTC"]
# National-team model (tested on 1,915 competitive internationals, 2024-2026)
INTL_DEFAULTS = {"half_life": 1825, "years": 10, "friendly_weight": 1.0}
# ---- UEFA competitions (fixtures + results: fixturedownload.com JSON feeds) ----
UEFA_FEED = "https://fixturedownload.com/feed/json/{slug}"
UEFA_COMPS = {"UCL": ("UEFA Champions League", "champions-league"),
              "UEL": ("UEFA Europa League", "europa-league"),
              "UECL": ("UEFA Conference League", "conference-league")}
NATIONS_LEAGUE = ("UNL", "UEFA Nations League", "nations-league")
# Tested on the 2025/26 European season (531 matches, refitted monthly on past
# matches only): right result 56.5%, log-loss 0.944 vs 1.022 naive baseline.
UEFA_DEFAULTS = {"half_life": 730, "offset_penalty": 1.0, "club_penalty": 5.0,
                 "seasons": 3}
# Older / alternative spellings in the UEFA feed -> current spelling
UEFA_ALIASES = {"Atlético de Madrid": "Atleti", "Dortmund": "B. Dortmund",
                "Crvena zvezda": "Crvena Zvezda", "Shakhtar Donetsk": "Shakhtar",
                "FC Porto": "Porto", "Bayern": "Bayern München", "Man United": "Man Utd",
                "Omonoia": "Omonia"}
# UEFA club name -> (league code, Football-Data name) for clubs in covered leagues
UEFA_CLUB_MAP = {
    "AEK Athens": ("G1", "AEK"), "AZ Alkmaar": ("N1", "AZ Alkmaar"),
    "Aberdeen": ("SC0", "Aberdeen"), "Ajax": ("N1", "Ajax"),
    "Anderlecht": ("B1", "Anderlecht"), "Antwerp": ("B1", "Antwerp"),
    "Arsenal": ("E0", "Arsenal"), "Aston Villa": ("E0", "Aston Villa"),
    "Atalanta": ("I1", "Atalanta"), "Athletic Club": ("SP1", "Ath Bilbao"),
    "Atleti": ("SP1", "Ath Madrid"), "B. Dortmund": ("D1", "Dortmund"),
    "Barcelona": ("SP1", "Barcelona"), "Basaksehir": ("T1", "Buyuksehyr"),
    "Bayern München": ("D1", "Bayern Munich"), "Benfica": ("P1", "Benfica"),
    "Besiktas": ("T1", "Besiktas"), "Bologna": ("I1", "Bologna"),
    "Bournemouth": ("E0", "Bournemouth"), "Braga": ("P1", "Sp Braga"),
    "Brest": ("F1", "Brest"), "Brighton": ("E0", "Brighton"),
    "Celta": ("SP1", "Celta"), "Celtic": ("SC0", "Celtic"),
    "Cercle Brugge": ("B1", "Cercle Brugge"), "Chelsea": ("E0", "Chelsea"),
    "Club Brugge": ("B1", "Club Brugge"), "Como": ("I1", "Como"),
    "Crystal Palace": ("E0", "Crystal Palace"), "Fenerbahçe": ("T1", "Fenerbahce"),
    "Feyenoord": ("N1", "Feyenoord"), "Fiorentina": ("I1", "Fiorentina"),
    "Frankfurt": ("D1", "Ein Frankfurt"), "Freiburg": ("D1", "Freiburg"),
    "Galatasaray": ("T1", "Galatasaray"), "Genk": ("B1", "Genk"), "Gent": ("B1", "Gent"),
    "Getafe": ("SP1", "Getafe"), "Girona": ("SP1", "Girona"),
    "Go Ahead Eagles": ("N1", "Go Ahead Eagles"), "Hearts": ("SC0", "Hearts"),
    "Heidenheim": ("D1", "Heidenheim"), "Hoffenheim": ("D1", "Hoffenheim"),
    "Inter": ("I1", "Inter"), "Juventus": ("I1", "Juventus"), "Lazio": ("I1", "Lazio"),
    "Leipzig": ("D1", "RB Leipzig"), "Lens": ("F1", "Lens"),
    "Leverkusen": ("D1", "Leverkusen"), "Lille": ("F1", "Lille"),
    "Liverpool": ("E0", "Liverpool"), "Lyon": ("F1", "Lyon"), "Mainz": ("D1", "Mainz"),
    "Man City": ("E0", "Man City"), "Man Utd": ("E0", "Man United"),
    "Marseille": ("F1", "Marseille"), "Milan": ("I1", "Milan"), "Monaco": ("F1", "Monaco"),
    "N.E.C.": ("N1", "Nijmegen"), "Napoli": ("I1", "Napoli"),
    "Newcastle": ("E0", "Newcastle"), "Nice": ("F1", "Nice"),
    "Nott'm Forest": ("E0", "Nott'm Forest"), "OFI Crete": ("G1", "OFI Crete"),
    "Olympiacos": ("G1", "Olympiakos"), "PAOK": ("G1", "PAOK"),
    "PSV": ("N1", "PSV Eindhoven"), "Panathinaikos": ("G1", "Panathinaikos"),
    "Paris": ("F1", "Paris SG"), "Porto": ("P1", "Porto"), "Rangers": ("SC0", "Rangers"),
    "Rayo Vallecano": ("SP1", "Vallecano"), "Real Betis": ("SP1", "Betis"),
    "Real Madrid": ("SP1", "Real Madrid"), "Real Sociedad": ("SP1", "Sociedad"),
    "Rennes": ("F1", "Rennes"), "Roma": ("I1", "Roma"), "Samsunspor": ("T1", "Samsunspor"),
    "Sevilla": ("SP1", "Sevilla"), "Sint-Truidense": ("B1", "St Truiden"),
    "Sporting CP": ("P1", "Sp Lisbon"), "Strasbourg": ("F1", "Strasbourg"),
    "Stuttgart": ("D1", "Stuttgart"), "Sunderland": ("E0", "Sunderland"),
    "Tottenham": ("E0", "Tottenham"), "Toulouse": ("F1", "Toulouse"),
    "Trabzonspor": ("T1", "Trabzonspor"), "Twente": ("N1", "Twente"),
    "Union Berlin": ("D1", "Union Berlin"), "Union SG": ("B1", "St. Gilloise"),
    "Utrecht": ("N1", "Utrecht"), "Villarreal": ("SP1", "Villarreal"),
    "Vitória SC": ("P1", "Guimaraes"), "West Ham": ("E0", "West Ham"),
}
# Clubs from countries without a covered league -> country group
UEFA_CLUB_COUNTRY = {
    "AEK Larnaca": "CYP", "APOEL": "CYP", "Aarhus": "DEN", "Ararat-Armenia": "ARM",
    "Aris Limassol": "CYP", "Astana": "KAZ", "Basel": "SUI", "Bodø/Glimt": "NOR",
    "Borac": "BIH", "Brann": "NOR", "Breiðablik": "ISL", "CSKA Sofia": "BUL",
    "Celje": "SVN", "Copenhagen": "DEN", "Crvena Zvezda": "SRB", "Dinamo-Minsk": "BLR",
    "Djurgården": "SWE", "Drita": "KOS", "Dynamo Kyiv": "UKR", "Egnatia": "ALB",
    "Elfsborg": "SWE", "FCSB": "ROU", "Ferencváros": "HUN", "GNK Dinamo": "CRO",
    "H. Beer-Sheva": "ISR", "HJK": "FIN", "Hajduk Split": "CRO", "Hamrun Spartans": "MLT",
    "Häcken": "SWE", "Iberia Tbilisi": "GEO", "Inter Escaldes": "AND", "Jablonec": "CZE",
    "Jagiellonia": "POL", "Kairat Almaty": "KAZ", "Kauno Žalgiris": "LTU",
    "KuPS Kuopio": "FIN", "L. Red Imps": "GIB", "LASK": "AUT", "Larne": "NIR",
    "Lausanne-Sport": "SUI", "Lech Poznan": "POL", "Legia Warszawa": "POL",
    "Levski Sofia": "BUL", "Lillestrøm": "NOR", "Ludogorets": "BUL", "Lugano": "SUI",
    "M. Tel-Aviv": "ISR", "Maccabi Haifa": "ISR", "Malmö": "SWE", "Midtjylland": "DEN",
    "Mjällby": "SWE", "Mladá Boleslav": "CZE", "Molde": "NOR", "Noah": "ARM",
    "Nordsjælland": "DEN", "Olimpija": "SVN", "Omonia": "CYP", "Pafos": "CYP",
    "Petrocub": "MDA", "Qarabag": "AZE", "RFS": "LVA", "Raków": "POL", "Riga": "LVA",
    "Rijeka": "CRO", "S. Bratislava": "SVK", "SK Rapid": "AUT", "Sabah": "AZE",
    "Salzburg": "AUT", "Servette": "SUI", "Shakhtar": "UKR", "Shamrock Rovers": "IRL",
    "Shelbourne": "IRL", "Sheriff": "MDA", "Shkëndija": "MKD", "Sigma Olomouc": "CZE",
    "Slavia Praha": "CZE", "Sparta Praha": "CZE", "St. Gallen": "SUI", "Sturm Graz": "AUT",
    "TSC": "SRB", "The New Saints": "WAL", "Thun": "SUI", "Torreense": "POR2",
    "U. Craiova": "ROU", "Viking": "NOR", "Viktoria Plzen": "CZE", "Víkingur R.": "ISL",
    "Young Boys": "SUI", "Zrinjski": "BIH",
}
UEFA_LEAGUES = sorted({lg for lg, _ in UEFA_CLUB_MAP.values()})
# National-team tournaments picked up automatically as soon as their fixtures
# are published (code, name, feed slug prefix). Played at neutral venues.
NATION_TOURNAMENTS = [("AFCON", "Africa Cup of Nations", "afcon"),
                      ("WC", "FIFA World Cup", "fifa-world-cup"),
                      ("EURO", "UEFA European Championship", "uefa-euro"),
                      ("COPA", "Copa America", "copa-america")]
# Fixture-feed names -> names in the international results dataset
NATION_ALIASES = {"Czechia": "Czech Republic", "Türkiye": "Turkey",
                  "Côte d'Ivoire": "Ivory Coast", "Cabo Verde": "Cape Verde",
                  "Congo DR": "DR Congo", "Korea Republic": "South Korea",
                  "USA": "United States", "IR Iran": "Iran"}
# Region of each national team, from the tournaments it plays in
CONFEDERATIONS = [("Africa", r"African Cup of Nations|African Nations Championship"),
                  ("Europe", r"UEFA Euro|UEFA Nations League"),
                  ("Asia", r"AFC Asian Cup|Gulf Cup|Arab Cup|WAFF|EAFF|AFF Championship|SAFF"),
                  ("North & Central America", r"Gold Cup|CONCACAF"),
                  ("South America", r"Copa América"),
                  ("Oceania", r"OFC Nations Cup|Oceania Nations Cup")]
CUP_CODES.update(UEFA_COMPS, [NATIONS_LEAGUE[0]], [c for c, _, _ in NATION_TOURNAMENTS])
MIDDLE_EAST = {"Saudi Arabia", "Qatar", "United Arab Emirates", "Iran", "Iraq", "Kuwait",
               "Bahrain", "Oman", "Jordan", "Syria", "Lebanon", "Palestine", "Yemen"}

PAPER_LOG = Path(os.environ.get("EPL_PAPER_LOG",
                                Path(__file__).with_name("paper_bets.csv")))
PICK_LOG = Path(os.environ.get("EPL_PICK_LOG",
                               Path(__file__).with_name("best_picks_log.csv")))
PREDICTIONS_MD = Path(os.environ.get("EPL_PREDICTIONS",
                                     Path(__file__).with_name("predictions.md")))
UK_TZ = "Europe/London"
# Backup copy of every data source, refreshed by the scheduled GitHub run
SNAPSHOT_DIR = Path(os.environ.get("EPL_SNAPSHOT_DIR",
                                   Path(__file__).with_name("data_snapshot")))
SNAPSHOT_FRESH_HOURS = 30       # backups younger than this are used for history
OFFLINE = os.environ.get("EPL_OFFLINE") == "1"   # self-tests: never use the network
PREFER_SNAPSHOT = True          # the scheduled run switches this off (it downloads)

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
    "O05": "Over 0.5", "U05": "Under 0.5",
    "O15": "Over 1.5", "U15": "Under 1.5",
    "O25": "Over 2.5", "U25": "Under 2.5",
    "O35": "Over 3.5", "U35": "Under 3.5",
    "O45": "Over 4.5", "U45": "Under 4.5",
    "O55": "Over 5.5", "U55": "Under 5.5",
    "O65": "Over 6.5", "U65": "Under 6.5",
    "BTTS_Y": "BTTS Yes", "BTTS_N": "BTTS No",
    "1X": "Home or Draw (1X)", "X2": "Draw or Away (X2)", "12": "Home or Away (12)",
    "HS": "Home team to score", "AS": "Away team to score",
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
        resp = _http_get(url)
        if resp is None:
            return pd.DataFrame()
        try:
            text = resp.content.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = resp.content.decode("latin-1")
        df = pd.read_csv(io.StringIO(text), on_bad_lines="skip")
        df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
        return df.dropna(how="all")
    except Exception:
        return pd.DataFrame()


# -----------------------------------------------------------------------------
# Downloads with retries + a backup copy of every source (data_snapshot/)
# -----------------------------------------------------------------------------
def _http_get(url: str):
    """GET with 3 attempts and back-off. None on failure (or in offline mode)."""
    if OFFLINE:
        return None
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=HTTP_HEADERS, timeout=20)
            if resp.status_code == 404:
                return None                      # not published (yet) - no retry
            resp.raise_for_status()
            return resp
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    return None


def _with_source(df: pd.DataFrame, source: str) -> pd.DataFrame:
    df.attrs["source"] = source
    return df


def _snap_file(name: str) -> Path:
    return SNAPSHOT_DIR / f"{name}.csv"


def snapshot_meta() -> dict:
    try:
        return json.loads((SNAPSHOT_DIR / "meta.json").read_text(encoding="utf-8"))
    except Exception:
        return {}


def snapshot_age_hours(name: str):
    info = snapshot_meta().get("files", {}).get(name)
    if not info:
        return None
    try:
        saved = pd.Timestamp(info["saved_at"])
        return (pd.Timestamp.now(tz="UTC") - saved).total_seconds() / 3600
    except Exception:
        return None


def save_snapshot(name: str, df: pd.DataFrame) -> bool:
    """Write one source to the backup folder (plain CSV: git stores changes
    compactly) and record when it was saved."""
    if df is None or df.empty:
        return False
    try:
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(_snap_file(name), index=False)
        meta = snapshot_meta()
        meta.setdefault("files", {})[name] = {
            "saved_at": pd.Timestamp.now(tz="UTC").isoformat(), "rows": int(len(df))}
        meta["saved_at"] = pd.Timestamp.now(tz="UTC").isoformat()
        (SNAPSHOT_DIR / "meta.json").write_text(json.dumps(meta, indent=1, sort_keys=True),
                                               encoding="utf-8")
        return True
    except Exception:
        return False


def load_snapshot(name: str) -> pd.DataFrame:
    """Read one source from the backup folder (empty if missing/unreadable)."""
    try:
        f = _snap_file(name)
        if not f.exists():
            return pd.DataFrame()
        df = pd.read_csv(f, dtype={"Season": str} if name.startswith("results_") else None,
                         low_memory=False)
        if "Kickoff" in df.columns:
            df["Kickoff"] = pd.to_datetime(df["Kickoff"], utc=True, errors="coerce")
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        if "Neutral" in df.columns:
            df["Neutral"] = df["Neutral"].astype(str).str.upper().eq("TRUE")
        return df
    except Exception:
        return pd.DataFrame()


def _live_or_backup(name: str, live_fn, prefer_backup: bool) -> pd.DataFrame:
    """
    Slow-changing data: use the backup if it is fresh (fast start-up).
    Otherwise download; if the source is down, fall back to the backup.
    The returned frame says where it came from in .attrs["source"].
    """
    if prefer_backup and PREFER_SNAPSHOT:
        age = snapshot_age_hours(name)
        if age is not None and age < SNAPSHOT_FRESH_HOURS:
            snap = load_snapshot(name)
            if not snap.empty:
                return _with_source(snap, "backup copy")
    live = live_fn()
    if live is not None and not live.empty:
        return _with_source(live, "live")
    snap = load_snapshot(name)
    if not snap.empty:
        return _with_source(snap, "backup copy (source down)")
    return _with_source(pd.DataFrame() if live is None else live, "unavailable")


def filter_seasons(df: pd.DataFrame, seasons_back: int) -> pd.DataFrame:
    if df.empty or "Season" not in df.columns:
        return df
    lg = str(df["League"].iloc[0]) if "League" in df.columns else ""
    if lg in NEW_FORMAT:                       # extra leagues: keep by date
        cutoff = pd.Timestamp(dt.date.today()) - pd.Timedelta(days=365 * (seasons_back + 1))
        out = df[df["Date"] >= cutoff].reset_index(drop=True)
        return _with_source(out, df.attrs.get("source", "live"))
    wanted = set(season_codes(dt.date.today(), seasons_back))
    out = df[df["Season"].astype(str).isin(wanted)].reset_index(drop=True)
    return _with_source(out, df.attrs.get("source", "live"))


def _parse_dates(series: pd.Series) -> pd.Series:
    """Football-Data uses dd/mm/yyyy (older files dd/mm/yy)."""
    s = series.astype(str).str.strip()
    out = pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")
    missing = out.isna()
    if missing.any():
        out[missing] = pd.to_datetime(s[missing], format="%d/%m/%y", errors="coerce")
    return out


def _load_new_format(league: str, seasons_back: int) -> pd.DataFrame:
    """Extra leagues (Russia, Japan, China): one file with every season."""
    df = _fetch_csv(f"{BASE_URL}/new/{league}.csv")
    need = {"Date", "Home", "Away", "HG", "AG", "Season"}
    if df.empty or not need.issubset(df.columns):
        return pd.DataFrame()
    df = df.rename(columns={"Home": "HomeTeam", "Away": "AwayTeam",
                            "HG": "FTHG", "AG": "FTAG"})
    keep = (["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "Season"]
            + [c for c in ODDS_COLUMNS if c in df.columns])
    df = df[keep].copy()
    df["Season"] = df["Season"].astype(str).str.strip()
    df["Date"] = _parse_dates(df["Date"])
    cutoff = pd.Timestamp(dt.date.today()) - pd.Timedelta(days=365 * (seasons_back + 1))
    df = df[df["Date"] >= cutoff]
    df["Date"] = df["Date"].dt.strftime("%d/%m/%Y")          # cleaned below like the rest
    return df


# -----------------------------------------------------------------------------
# Gap-filling: when Football-Data.co.uk is behind, add the newest results from
# fixturedownload.com (scores only - no shots or odds) for the big leagues
# -----------------------------------------------------------------------------
GAPFILL_FEEDS = {"E0": "epl", "E1": "championship", "E2": "efl-league-one",
                 "E3": "efl-league-two", "SP1": "la-liga", "I1": "serie-a",
                 "D1": "bundesliga", "F1": "ligue-1", "F2": "ligue-2", "N1": "eredivisie",
                 "P1": "primeira-liga", "T1": "super-lig", "SC0": "scottish-premiership"}
# fixturedownload spelling -> Football-Data spelling (names too different to match)
FEED_ALIASES = {
    # China: API-Football still uses older / sponsor names
    "Chengdu Better City": "Chengdu Rongcheng", "Chongqing Tongliang Long": "Chongqing Tonglianglong",
    "Dalian Zhixing": "Dalian Yingbo", "Hangzhou Greentown": "Zhejiang Professional",
    "Henan Jianye": "Henan Songshan Longmen", "Qingdao Jonoon": "Qingdao Hainiu",
    "Qingdao Youth Island": "Qingdao West Coast", "SHANGHAI SIPG": "Shanghai Port",
    "Shanghai SIPG": "Shanghai Port", "Shandong Luneng": "Shandong Taishan",
    "Shenyang Urban": "Liaoning Tieren", "Sichuan Jiuniu": "Shenzhen Xinpengcheng",
    "Tianjin Teda": "Tianjin Jinmen Tiger",
    # API-Football names the automatic matcher can't pair safely on its own
    "OH Leuven": "Oud-Heverlee Leuven", "Union St. Gilloise": "St. Gilloise",
    "Zulte Waregem": "Waregem", "Bayer Leverkusen": "Leverkusen",
    "Arminia Bielefeld": "Bielefeld", "VfL Bochum": "Bochum",
    "Eintracht Braunschweig": "Braunschweig", "Energie Cottbus": "Cottbus",
    "Dynamo Dresden": "Dresden", "Manchester City": "Man City",
    "Manchester United": "Man United", "Paris FC": "Paris FC",
    "Paris Saint Germain": "Paris SG", "Hellas Verona": "Verona",
    "JEF United Chiba": "Chiba", "FC Tokyo": "FC Tokyo", "Fagiano Okayama": "Okayama",
    "Tokyo Verdy": "Verdy", "Rodina Moskva": "Rodina Moscow", "Dundee": "Dundee",
    "Dundee Utd": "Dundee United", "Heart Of Midlothian": "Hearts",
    "Queen of the South": "Queen of Sth", "Atletico Madrid": "Ath Madrid",
    "Deportivo La Coruna": "La Coruna", "Racing Santander": "Santander",
    "AD Ceuta FC": "Ceuta", "Celta de Vigo II": "Celta B", "Real Sociedad II": "Sociedad B",
    "Sporting Gijon": "Sp Gijon", "Başakşehir": "Buyuksehyr",
    "Spurs": "Tottenham", "Man Utd": "Man United", "Queens Park Rangers": "QPR",
    "West Bromwich Albion": "West Brom", "Wolverhampton Wanderers": "Wolves",
    "MK Dons": "Milton Keynes Dons", "Peterborough United": "Peterboro",
    "Sheffield Wednesday": "Sheffield Weds", "Bristol Rovers": "Bristol Rvs",
    "Athletic Club": "Ath Bilbao", "Atlético de Madrid": "Ath Madrid",
    "RCD Espanyol de Barcelona": "Espanol", "RC Deportivo": "La Coruna",
    "R. Racing Club": "Santander", "Real Betis": "Betis", "Real Sociedad": "Sociedad",
    "Rayo Vallecano": "Vallecano", "Deportivo Alavés": "Alaves", "Internazionale": "Inter",
    "FC Bayern München": "Bayern Munich", "Bayer 04 Leverkusen": "Leverkusen",
    "Borussia Dortmund": "Dortmund", "Borussia Mönchengladbach": "M'gladbach",
    "Eintracht Frankfurt": "Ein Frankfurt", "1. FC Köln": "FC Koln", "Hamburger SV": "Hamburg",
    "Sport-Club Freiburg": "Freiburg", "Havre Athletic Club": "Le Havre",
    "Paris Saint-Germain": "Paris SG", "Olympique Lyonnais": "Lyon",
    "Olympique de Marseille": "Marseille", "Stade Brestois 29": "Brest",
    "Stade Rennais FC": "Rennes", "Estac Troyes": "Troyes", "Stade Lavallois MFC": "Laval",
    "AS Saint-Étienne": "St Etienne", "EN Avant Guingamp": "Guingamp",
    "US Boulogne CO": "Boulogne", "Stade DE Reims": "Reims", "N.E.C. Nijmegen": "Nijmegen",
    "PSV": "PSV Eindhoven", "AZ": "AZ Alkmaar", "ADO Den Haag": "Den Haag",
    "Fortuna Sittard": "For Sittard", "PEC Zwolle": "Zwolle", "Académico": "Academico Viseu",
    "SC Braga": "Sp Braga", "Sporting CP": "Sp Lisbon", "Vitória SC": "Guimaraes",
    "Marítimo M.": "Maritimo", "Istanbul Basaksehir": "Buyuksehyr", "Göztepe": "Goztep",
    "Çaykur Rizespor": "Rizespor", "Heart of Midlothian": "Hearts",
}
_GENERIC_TOKENS = {
    "fc", "cf", "sc", "afc", "ac", "ud", "cd", "rc", "rcd", "sv", "vfb", "fsv", "tsg", "as",
    "ogc", "losc", "sl", "ca", "club", "de", "del", "la", "1", "04", "05", "07", "29", "38",
    "63", "calcio", "sco", "aj", "stade", "us", "usl", "en", "foot", "football", "mfc", "co",
    "cp", "fco", "pec", "s", "r", "m", "city", "town", "athletic", "rovers", "united",
    "county", "wanderers", "albion", "alexandra", "argyle", "stanley", "utd"}


def _name_key(s: str, core: bool = False) -> str:
    import re
    import unicodedata
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    words = re.sub(r"[^a-z0-9 ]", " ", s.replace("'", "")).split()
    if core:
        kept = [w for w in words if w not in _GENERIC_TOKENS]
        words = kept or words
    return " ".join(words)


def match_club_names(feed_names, fd_names) -> dict:
    """{feed name: Football-Data name} - only confident, one-to-one matches;
    anything uncertain is left out (its games are skipped, never guessed)."""
    import difflib
    from collections import Counter
    fd = list(fd_names)
    by_norm = {_name_key(n): n for n in fd}
    by_core = {}
    for n in fd:
        by_core.setdefault(_name_key(n, True), []).append(n)
    out = {}
    for f in feed_names:
        alias = FEED_ALIASES.get(f)
        if alias in fd:
            out[f] = alias
            continue
        if _name_key(f) in by_norm:
            out[f] = by_norm[_name_key(f)]
            continue
        c = _name_key(f, True)
        if len(by_core.get(c, [])) == 1:
            out[f] = by_core[c][0]
            continue
        best, score = None, 0.0
        for n in fd:
            cn = _name_key(n, True)
            sc_ = difflib.SequenceMatcher(None, c, cn).ratio()
            if cn and (c.startswith(cn) or cn.startswith(c)):
                sc_ = max(sc_, 0.9)
            if sc_ > score:
                best, score = n, sc_
        if best is not None and score >= 0.8:
            out[f] = best
    counts = Counter(out.values())
    return {f: n for f, n in out.items() if counts[n] == 1}


def _fd_feed(league: str):
    """Played matches from fixturedownload.com (DateUtc/HomeTeam/... columns)."""
    if league not in GAPFILL_FEEDS:
        return None
    start = _season_start_year(dt.date.today())
    data = _fetch_json(UEFA_FEED.format(slug=f"{GAPFILL_FEEDS[league]}-{start}"))
    return pd.DataFrame(data) if data else None


AF_FILL_STATUS: dict = {}     # league -> what API-Football returned (daily-run diagnostics)


def _af_current_season(key: str, af_id: int, league: str) -> int:
    """The season label API-Football itself uses for the league's current season
    (e.g. Japan's J1 moved to an autumn-spring calendar and is now labelled 2027).
    Falls back to a calendar guess if the lookup fails."""
    today = dt.date.today()
    guess = today.year if league in ("JPN", "CHN") else _season_start_year(today)
    try:
        js = _apifootball_get("/leagues", key, id=af_id, current="true") or {}
        for item in js.get("response") or []:
            for s in item.get("seasons") or []:
                if s.get("current") and s.get("year"):
                    return int(s["year"])
    except Exception:
        pass
    return guess


def _af_feed(league: str):
    """Played matches from API-Football (only when the APIFOOTBALL_KEY secret is
    set, i.e. in the scheduled GitHub run). Same columns as _fd_feed."""
    key = os.environ.get("APIFOOTBALL_KEY", "").strip()
    af_id = next((i for i, c in AF_LEAGUES.items() if c == league), None)
    if not key or af_id is None:
        return None
    season = _af_current_season(key, af_id, league)
    js = _apifootball_get("/fixtures", key, league=af_id, season=season, status="FT-AET-PEN")
    resp = (js or {}).get("response") or []
    AF_FILL_STATUS[league] = {"season": season, "finished": len(resp),
                              "latest": max(((f.get("fixture") or {}).get("date") or "")[:10]
                                            for f in resp) if resp else None,
                              "errors": (js or {}).get("errors") or None,
                              "teams": sorted({t for f in resp for t in
                                               (f["teams"]["home"]["name"], f["teams"]["away"]["name"])})}
    rows = []
    for f in (js or {}).get("response") or []:
        g = f.get("goals") or {}
        rows.append({"DateUtc": (f.get("fixture") or {}).get("date"),
                     "HomeTeam": f["teams"]["home"]["name"], "AwayTeam": f["teams"]["away"]["name"],
                     "HomeTeamScore": g.get("home"), "AwayTeamScore": g.get("away")})
    return pd.DataFrame(rows) if rows else None


def _append_feed(df: pd.DataFrame, feed, league: str, source: str) -> pd.DataFrame:
    """Append the feed's matches that are newer than df's latest result."""
    if feed is None or feed.empty:
        return df
    feed = feed.dropna(subset=["HomeTeamScore", "AwayTeamScore", "HomeTeam", "AwayTeam"]).copy()
    if feed.empty:
        return df
    feed["Date"] = (pd.to_datetime(feed["DateUtc"], utc=True, errors="coerce")
                    .dt.tz_convert(UK_TZ).dt.tz_localize(None).dt.normalize())
    newer = feed[feed["Date"] > df["Date"].max()]
    if newer.empty:
        return df
    current = str(df.loc[df["Date"].idxmax(), "Season"])
    cur_teams = df.loc[df["Season"].astype(str) == current, ["HomeTeam", "AwayTeam"]]
    fd_names = set(cur_teams["HomeTeam"]) | set(cur_teams["AwayTeam"]) or \
        set(df["HomeTeam"]) | set(df["AwayTeam"])
    # match against the feed's FULL team list, so a similar name (FC Tokyo / Tokyo Verdy)
    # can never be paired with the wrong club just because the other one is absent
    names = match_club_names(set(feed["HomeTeam"]) | set(feed["AwayTeam"]), fd_names)
    newer = newer[newer["HomeTeam"].isin(names) & newer["AwayTeam"].isin(names)]
    if newer.empty:
        return df
    add = pd.DataFrame({
        "Date": newer["Date"], "HomeTeam": newer["HomeTeam"].map(names),
        "AwayTeam": newer["AwayTeam"].map(names),
        "FTHG": pd.to_numeric(newer["HomeTeamScore"]).astype(int),
        "FTAG": pd.to_numeric(newer["AwayTeamScore"]).astype(int),
        "Season": current, "League": league, "Filled": True, "FilledFrom": source})
    add = add[add["FTHG"].between(0, 15) & add["FTAG"].between(0, 15)]
    out = pd.concat([df.assign(Filled=False) if "Filled" not in df.columns else df, add],
                    ignore_index=True)
    out = out.drop_duplicates(["Date", "HomeTeam", "AwayTeam"], keep="first")
    return out.sort_values("Date").reset_index(drop=True)


def gap_fill_results(df: pd.DataFrame, league: str) -> pd.DataFrame:
    """Append results newer than Football-Data's latest: first from
    fixturedownload.com (free), then from API-Football (paid key, GitHub run)."""
    if df.empty:
        return df
    for source, getter in (("fixturedownload.com", _fd_feed), ("API-Football", _af_feed)):
        try:
            df = _append_feed(df, getter(league), league, source)
        except Exception:
            pass                       # gap-filling is a bonus - never break the load
    return df


def _load_results_uncached(league: str, seasons_back: int) -> pd.DataFrame:
    """Fetch and clean completed results of one league, current + past seasons."""
    frames = []
    if league in NEW_FORMAT:
        nf = _load_new_format(league, seasons_back)
        if not nf.empty:
            frames.append(nf)
    for code in ([] if league in NEW_FORMAT else season_codes(dt.date.today(), seasons_back)):
        df = _fetch_csv(f"{BASE_URL}/mmz4281/{code}/{league}.csv")
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
    # Sanity checks: impossible scores, future dates, duplicated rows
    df = df[df["FTHG"].between(0, 15) & df["FTAG"].between(0, 15)]
    df = df[df["Date"] <= pd.Timestamp(dt.date.today()) + pd.Timedelta(days=1)]
    df = df.drop_duplicates(["Date", "HomeTeam", "AwayTeam"], keep="last")
    for col in SHOT_COLUMNS + ODDS_COLUMNS:        # shots/xG + odds as numbers
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["League"] = league
    df = gap_fill_results(df.sort_values("Date").reset_index(drop=True), league)
    return df


def _results_with_backup(league: str, seasons_back: int) -> pd.DataFrame:
    """Results of one league: fresh backup -> download -> any backup."""
    df = _live_or_backup(f"results_{league}",
                         lambda: _load_results_uncached(league, max(seasons_back, 3)),
                         prefer_backup=True)
    return filter_seasons(df, seasons_back)


@st.cache_data(ttl=3600, show_spinner=False)
def load_results(league: str = DEFAULT_LEAGUE, seasons_back: int = 2) -> pd.DataFrame:
    """Cached results of one league."""
    return _results_with_backup(league, seasons_back)


@st.cache_data(ttl=3600, show_spinner=False)
def load_all_results(leagues: tuple, seasons_back: int) -> dict:
    """Results for several leagues, in parallel. {code: DataFrame}"""
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=8) as ex:
        frames = list(ex.map(lambda lg: _results_with_backup(lg, seasons_back), leagues))
    return {lg: df for lg, df in zip(leagues, frames) if not df.empty}


@st.cache_data(ttl=3600, show_spinner=False)
def load_fixtures() -> pd.DataFrame:
    """Upcoming fixtures + odds: always downloaded fresh (odds move); the
    backup is only used if the source is down."""
    return _live_or_backup("fixtures", _load_fixtures_live, prefer_backup=False)


def _load_fixtures_live() -> pd.DataFrame:
    df = _fetch_csv(f"{BASE_URL}/fixtures.csv")
    extra = _fetch_csv(f"{BASE_URL}/new_league_fixtures.csv")
    if not extra.empty and {"Country", "League", "Home", "Away"}.issubset(extra.columns):
        names = {v: k for k, v in NEW_FORMAT.items()}
        extra["Div"] = [names.get((str(c).strip(), str(l).strip()))
                        for c, l in zip(extra["Country"], extra["League"])]
        extra = extra.dropna(subset=["Div"]).rename(columns={"Home": "HomeTeam",
                                                              "Away": "AwayTeam"})
        df = pd.concat([df, extra], ignore_index=True) if not df.empty else extra
    if df.empty or "Div" not in df.columns:
        return pd.DataFrame()
    df["Div"] = df["Div"].astype(str).str.strip()
    df = df[df["Div"].isin(LEAGUES)].copy()
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


@st.cache_data(ttl=3600, show_spinner=False)
def load_schedule() -> pd.DataFrame:
    """Full-season fixture lists (no odds) for the leagues fixturedownload.com
    covers. Football-Data only publishes the next round, mid-week."""
    return _live_or_backup("schedule", _load_schedule_live, prefer_backup=True)


def _load_schedule_live() -> pd.DataFrame:
    from concurrent.futures import ThreadPoolExecutor
    start = _season_start_year(dt.date.today())

    def one(item):
        lg, slug = item
        data = _fetch_json(UEFA_FEED.format(slug=f"{slug}-{start}"))
        if not data:
            return None
        f = pd.DataFrame(data)
        if not {"DateUtc", "HomeTeam", "AwayTeam"}.issubset(f.columns):
            return None
        if "HomeTeamScore" in f.columns:
            f = f[f["HomeTeamScore"].isna()]
        f = f.assign(Div=lg)[["Div", "DateUtc", "HomeTeam", "AwayTeam"]]
        return f
    with ThreadPoolExecutor(max_workers=8) as ex:
        frames = [f for f in ex.map(one, GAPFILL_FEEDS.items()) if f is not None]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["Kickoff"] = pd.to_datetime(df["DateUtc"], utc=True, errors="coerce")
    df = df.dropna(subset=["Kickoff", "HomeTeam", "AwayTeam"])
    return df.drop(columns="DateUtc").reset_index(drop=True)


def add_schedule(fixtures: pd.DataFrame, schedule: pd.DataFrame,
                 all_results: dict) -> pd.DataFrame:
    """Add every scheduled league match that the odds feed has not published
    yet (marked NoOdds). Club names are mapped to the results' spelling."""
    if schedule is None or schedule.empty:
        return fixtures
    try:
        sch = schedule.copy()
        sch["Kickoff"] = pd.to_datetime(sch["Kickoff"], utc=True)
        add = []
        for lg, grp in sch.groupby("Div"):
            res = all_results.get(lg)
            if res is None or res.empty:
                continue
            current = season_codes(dt.date.today(), 0)[0]
            cur = res[res["Season"].astype(str) == current]
            cur = cur if not cur.empty else res
            names = match_club_names(set(grp["HomeTeam"]) | set(grp["AwayTeam"]),
                                     set(cur["HomeTeam"]) | set(cur["AwayTeam"]))
            g = grp[grp["HomeTeam"].isin(names) & grp["AwayTeam"].isin(names)].copy()
            g["HomeTeam"] = g["HomeTeam"].map(names)
            g["AwayTeam"] = g["AwayTeam"].map(names)
            add.append(g)
        if not add:
            return fixtures
        add = pd.concat(add, ignore_index=True)
        add["Date"] = (add["Kickoff"].dt.tz_convert(UK_TZ).dt.tz_localize(None)
                       .dt.normalize())
        if fixtures is not None and not fixtures.empty:
            known = fixtures[["Div", "HomeTeam", "AwayTeam", "Date"]]
            m = add.merge(known, on=["Div", "HomeTeam", "AwayTeam"], how="left",
                          suffixes=("", "_fd"))
            dup = (m["Date_fd"].notna()
                   & ((m["Date"] - m["Date_fd"]).abs() <= pd.Timedelta(days=3)))
            dup_keys = set(zip(m.loc[dup, "Div"], m.loc[dup, "HomeTeam"],
                               m.loc[dup, "AwayTeam"], m.loc[dup, "Date"]))
            add = add[[k not in dup_keys for k in
                       zip(add["Div"], add["HomeTeam"], add["AwayTeam"], add["Date"])]]
        if add.empty:
            return fixtures
        add = add.assign(NoOdds=True)
        base = fixtures if fixtures is not None else pd.DataFrame()
        out = pd.concat([base, add], ignore_index=True) if not base.empty else add
        out.attrs.update(getattr(fixtures, "attrs", {}) or {})
        return out.sort_values("Kickoff").reset_index(drop=True)
    except Exception:
        return fixtures                # the schedule is a bonus - never break fixtures


def upcoming_fixtures(fixtures: pd.DataFrame, days: int = 7) -> pd.DataFrame:
    """Fixtures that have not kicked off yet, within the next `days`."""
    if fixtures.empty or "Kickoff" not in fixtures.columns:
        return fixtures.iloc[0:0] if not fixtures.empty else fixtures
    now = pd.Timestamp.now(tz="UTC")
    return fixtures[(fixtures["Kickoff"] > now)
                    & (fixtures["Kickoff"] <= now + pd.Timedelta(days=days))]


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def load_international() -> pd.DataFrame:
    """All men's full internationals (open dataset, updated after each window)."""
    return _live_or_backup("international", _load_international_live, prefer_backup=True)


def _load_international_live() -> pd.DataFrame:
    df = _fetch_csv(INTL_URL)
    need = {"date", "home_team", "away_team", "home_score", "away_score",
            "tournament", "neutral"}
    if df.empty or not need.issubset(df.columns):
        return pd.DataFrame()
    out = pd.DataFrame({
        "Date": pd.to_datetime(df["date"], errors="coerce"),
        "HomeTeam": df["home_team"].astype(str).str.strip(),
        "AwayTeam": df["away_team"].astype(str).str.strip(),
        "FTHG": pd.to_numeric(df["home_score"], errors="coerce"),
        "FTAG": pd.to_numeric(df["away_score"], errors="coerce"),
        "Tournament": df["tournament"].astype(str),
        "Neutral": df["neutral"].astype(str).str.upper().eq("TRUE"),
    })
    out = out.dropna(subset=["Date", "FTHG", "FTAG"])
    out = out[out["Date"] >= pd.Timestamp(dt.date.today()) - pd.Timedelta(days=365 * 12)]
    return out.sort_values("Date").reset_index(drop=True)


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


def _poisson_mle(hi, ai, yh, ya, w, n, home_flag=None):
    """
    Weighted Poisson maximum likelihood with an L2 penalty on team ratings.
    Parameters: [mu, home, attack(n), defence(n)]. home_flag (0/1 per match)
    switches home advantage off for neutral venues. Returns (params, ok).
    """
    hf = np.ones(len(hi)) if home_flag is None else np.asarray(home_flag, float)

    def nll_and_grad(p):
        mu, home = p[0], p[1]
        att, dfn = p[2:2 + n], p[2 + n:]
        eta_h = mu + home * hf + att[hi] - dfn[ai]
        eta_a = mu + att[ai] - dfn[hi]
        lam_h, lam_a = np.exp(eta_h), np.exp(eta_a)

        nll = np.sum(w * (lam_h - yh * eta_h + lam_a - ya * eta_a))
        nll += SHRINKAGE * (att @ att + dfn @ dfn)

        # Analytic gradient (fast + exact)
        rh, ra = w * (lam_h - yh), w * (lam_a - ya)
        g = np.zeros_like(p)
        g[0] = rh.sum() + ra.sum()
        g[1] = (rh * hf).sum()
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
    return res.x, bool(res.success)


def _dixon_coles_rho(gh, ga, lam_h, lam_a, w) -> float:
    """Dixon-Coles low-score correction, by 1-D profile likelihood."""
    best_rho, best_ll = 0.0, -np.inf
    m00 = (gh == 0) & (ga == 0)
    m01 = (gh == 0) & (ga == 1)
    m10 = (gh == 1) & (ga == 0)
    m11 = (gh == 1) & (ga == 1)
    for rho in np.arange(-0.25, 0.1001, 0.0025):
        tau = np.ones_like(lam_h)
        tau[m00] = 1 - lam_h[m00] * lam_a[m00] * rho
        tau[m01] = 1 + lam_h[m01] * rho
        tau[m10] = 1 + lam_a[m10] * rho
        tau[m11] = 1 - rho
        if np.any(tau <= 0):
            continue
        ll = np.sum(w * np.log(tau))
        if ll > best_ll:
            best_ll, best_rho = ll, rho
    return float(best_rho)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def fit_international(df: pd.DataFrame, half_life_days: int = INTL_DEFAULTS["half_life"],
                      years: int = INTL_DEFAULTS["years"],
                      friendly_weight: float = INTL_DEFAULTS["friendly_weight"]) -> dict:
    """National-team ratings: same Poisson model, goals only, home advantage
    only when the match is not at a neutral venue."""
    last = df["Date"].max()
    train = df[df["Date"] >= last - pd.Timedelta(days=365 * years)].reset_index(drop=True)
    teams = sorted(set(train["HomeTeam"]) | set(train["AwayTeam"]))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    hi = train["HomeTeam"].map(idx).to_numpy()
    ai = train["AwayTeam"].map(idx).to_numpy()
    gh = train["FTHG"].to_numpy(float)
    ga = train["FTAG"].to_numpy(float)
    age = (last - train["Date"]).dt.days.to_numpy()
    w = np.exp(-np.log(2) / half_life_days * age)
    w = w * np.where(train["Tournament"].eq("Friendly"), friendly_weight, 1.0)
    hf = (~train["Neutral"]).to_numpy(float)
    p, ok = _poisson_mle(hi, ai, gh, ga, w, n, home_flag=hf)
    mu, home = p[0], p[1]
    att, dfn = p[2:2 + n], p[2 + n:]
    lam_h = np.exp(mu + home * hf + att[hi] - dfn[ai])
    lam_a = np.exp(mu + att[ai] - dfn[hi])
    recent = train[train["Date"] >= last - pd.Timedelta(days=3 * 365)]
    current = sorted(set(recent["HomeTeam"]) | set(recent["AwayTeam"]))
    return {
        "mu": float(mu), "home": float(home),
        "rho": _dixon_coles_rho(gh, ga, lam_h, lam_a, w),
        "prior": {"attack": 0.0, "defence": 0.0},
        "attack": {t: float(att[i]) for t, i in idx.items()},
        "defence": {t: float(dfn[i]) for t, i in idx.items()},
        "teams": teams, "current_teams": current or teams, "pick_shrink": INTL_PICK_SHRINK,
        "n_matches": int(len(train)), "last_date": last, "converged": ok,
        "goals_weight": 1.0,
    }


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

    p, converged = _poisson_mle(hi, ai, yh, ya, w, n)
    mu, home = p[0], p[1]
    att, dfn = p[2:2 + n], p[2 + n:]

    # ---- Dixon-Coles rho: 1-D profile likelihood on low scorelines ----------
    lam_h = np.exp(mu + home + att[hi] - dfn[ai])
    lam_a = np.exp(mu + att[ai] - dfn[hi])
    best_rho = _dixon_coles_rho(gh, ga, lam_h, lam_a, w)

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
        "converged": converged,
        "goals_per_sot": per_sot,
        "n_xg": n_xg,
        "goals_weight": goals_weight,
    }


def expected_goals(model: dict, home: str, away: str, adj: dict | None = None,
                   neutral: bool = False):
    """
    Expected goals (lambda) for each side, with optional 'What If' multipliers.
    adj keys: h_att, h_def, a_att, a_def  (fractions, e.g. -0.15 = -15%)
      *_att : change to that team's scoring power
      *_def : change to that team's defensive leakiness (+ = concedes more)
    """
    def r(kind, team):  # unknown team -> promoted-team prior
        return model[kind].get(team, model["prior"][kind])

    lam_h = np.exp(model["mu"] + (0.0 if neutral else model["home"])
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
    probs["1X"] = probs["H"] + probs["D"]          # double chance
    probs["X2"] = probs["D"] + probs["A"]
    probs["12"] = probs["H"] + probs["A"]
    probs["HS"] = float(1 - m[0, :].sum())          # home team scores at least once
    probs["AS"] = float(1 - m[:, 0].sum())          # away team scores at least once
    for line, key in [(0.5, "05"), (1.5, "15"), (2.5, "25"), (3.5, "35"), (4.5, "45"),
                      (5.5, "55"), (6.5, "65")]:
        probs[f"O{key}"] = m[total > line].sum()
        probs[f"U{key}"] = 1 - probs[f"O{key}"]
    return {k: float(v) for k, v in probs.items()}


def predict(model, home, away, adj=None, neutral=False):
    lam_h, lam_a = expected_goals(model, home, away, adj, neutral)
    m = score_matrix(lam_h, lam_a, model["rho"])
    return lam_h, lam_a, m, market_probs(m)


def top_scores(m: np.ndarray, k: int = 5) -> list[tuple[str, float]]:
    """The k most likely exact scores as [("1-1", 0.12), ...]. In testing the
    top 5 contained the real score ~50% of the time (top 3: ~33%)."""
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


GOAL_LINES = ["0.5", "1.5", "2.5", "3.5", "4.5", "5.5", "6.5"]
GOAL_COLS = [f"{side} {g} %" for g in GOAL_LINES for side in ("Over", "Under")]


def goal_line_cols(p: dict) -> dict:
    """Over / Under 0.5, 1.5 and 2.5 goals, in %."""
    out = {}
    for g in GOAL_LINES:
        k = g.replace(".", "")
        out[f"Over {g} %"] = p[f"O{k}"] * 100
        out[f"Under {g} %"] = p[f"U{k}"] * 100
    return out


# "Best pick": the option most likely to win, among 17 common markets, capped
# at fair odds ~1.09 (92%). Where bookmaker odds exist, the model's score
# matrix is blended 50/50 with one fitted to those odds (more accurate).
# Walk-forward backtest, 12 leagues, 2024/25 + 2025/26, 8,591 matches:
# picks won 88.3% overall (same in both seasons); BANKER (>= 90%) 92.5%,
# STRONG (88-90%) 90.4%, GOOD (85-88%) 88.1%, FAIR 84.2%. Bookmakers almost
# never pay the fair price on such short odds (about -4% at Bet365), so this
# is a "most likely to win" pick, NOT a value signal.
BEST_PICK_KEYS = ["H", "D", "A", "1X", "X2", "12", "O15", "U15", "O25", "U25",
                  "O35", "U35", "U45", "BTTS_Y", "BTTS_N", "HS", "AS"]
BEST_PICK_MAX_P = 0.92
BEST_PICK_BLEND = 0.5
BEST_PICK_TIERS = [(0.90, "🔒", "Banker", "~92%"), (0.88, "⭐⭐⭐", "Strong", "~90%"),
                   (0.85, "⭐⭐", "Good", "~88%"), (0.0, "⭐", "Fair", "~84%")]
BEST_PICK_STRONG = 0.88             # 'strongest picks' lists: Banker + Strong (~91%)
BEST_PICK_HIT = "88%"
# European cups: walk-forward test of all 531 UEFA games of 2025/26 showed the
# cup model ~5 points over-confident on best picks (expected 87.4%, won 82.1%),
# so cup pick probabilities are shrunk towards 50% by this factor.
UEFA_PICK_SHRINK = 0.86
# Out-of-sample check on 2024/25 (441 games): 77.6% -> 82.5% won with it.
# Both seasons together (972 games): best picks won ~85%.
# National teams: 1,315 competitive games 2025-26, model ~3 points
# over-confident (87.5% expected, 84.7% won). With 0.92: 88.2% won, 87.3% expected.
INTL_PICK_SHRINK = 0.92


def shrink_probs(p: dict, k: float) -> dict:
    """Pull every probability towards 50% (k=1: unchanged)."""
    if k is None or k >= 1:
        return p
    return {key: 0.5 + (v - 0.5) * k for key, v in p.items()}


def raw_odds(row) -> dict:
    """1X2 + O/U 2.5 decimal odds of a fixture (Bet365, else market average)."""
    for book in ("Bet365", "Market average"):
        cols = BOOKMAKERS[book]
        try:
            o = {k: float(row[c]) for k, c in cols.items()}
        except (KeyError, TypeError, ValueError):
            continue
        if all(np.isfinite(v) and v > 1 for v in o.values()):
            return o
    return {}


def market_matrix(odds: dict, rho: float):
    """Score matrix whose 1X2 (and O/U 2.5 when given) match the bookmaker's,
    margin removed."""
    ok = lambda k: (odds or {}).get(k) is not None and np.isfinite(odds[k]) and odds[k] > 1
    if not odds or not all(ok(k) for k in ("H", "D", "A")):
        return None
    ih, idr, ia = 1 / odds["H"], 1 / odds["D"], 1 / odds["A"]
    s_ = ih + idr + ia
    ph, pa = ih / s_, ia / s_
    has_ou = ok("O25") and ok("U25")       # some books publish only 1X2
    po = (1 / odds["O25"]) / (1 / odds["O25"] + 1 / odds["U25"]) if has_ou else None

    def loss(x):
        pr = market_probs(score_matrix(np.exp(x[0]), np.exp(x[1]), rho))
        return ((pr["H"] - ph) ** 2 + (pr["A"] - pa) ** 2
                + ((pr["O25"] - po) ** 2 if has_ou else 0.0))
    try:
        x = minimize(loss, [0.3, 0.1], method="Nelder-Mead",
                     options={"xatol": 1e-4, "fatol": 1e-10}).x
        return score_matrix(np.exp(x[0]), np.exp(x[1]), rho)
    except Exception:
        return None


def pick_probs(m: np.ndarray, rho: float, odds) -> tuple:
    """Probabilities for the best pick: model blended with the odds when known."""
    mm = market_matrix(odds or {}, rho)
    if mm is None:
        return market_probs(m), False
    return market_probs((1 - BEST_PICK_BLEND) * m + BEST_PICK_BLEND * mm), True


def best_pick(p: dict, home: str, away: str,
              min_edge: float = DEFAULTS["min_edge"]) -> dict:
    """The most likely option among the main markets, capped at fair odds ~1.09."""
    ok = [k for k in BEST_PICK_KEYS if p[k] <= BEST_PICK_MAX_P]
    k = max(ok or BEST_PICK_KEYS, key=lambda x: p[x])
    label = {"H": f"{home} win", "D": "Draw", "A": f"{away} win",
             "1X": f"{home} or draw", "X2": f"{away} or draw",
             "12": f"{home} or {away} (no draw)",
             "BTTS_Y": "Both teams to score", "BTTS_N": "Not both teams to score",
             "HS": f"{home} to score", "AS": f"{away} to score",
             }.get(k, MARKET_LABELS[k] + " goals")
    floor = (1 / p[k]) * (1 + min_edge / 100)
    stars, tier, tier_hit = next((st_, t, h) for lim, st_, t, h in BEST_PICK_TIERS
                                 if p[k] >= lim)
    return {"key": k, "label": label, "prob": p[k], "floor": floor, "stars": stars,
            "tier": tier, "tier_hit": tier_hit, "how": how_to_bet(k, home, away, floor),
            "text": f"{stars} {label} · {p[k]*100:.0f}% · bet at {floor:.2f}+"}


def how_to_bet(k: str, home: str, away: str, floor: float) -> str:
    """Plain-English instruction: which market to open, what to pick, when it
    wins and when it loses."""
    goals = {"O15": ("Over 1.5", "2 or more goals", "0 or 1 goal"),
             "U15": ("Under 1.5", "0 or 1 goal", "2 or more goals"),
             "O25": ("Over 2.5", "3 or more goals", "0, 1 or 2 goals"),
             "U25": ("Under 2.5", "0, 1 or 2 goals", "3 or more goals"),
             "O35": ("Over 3.5", "4 or more goals", "0 to 3 goals"),
             "U35": ("Under 3.5", "0, 1, 2 or 3 goals", "4 or more goals"),
             "U45": ("Under 4.5", "0, 1, 2, 3 or 4 goals", "5 or more goals")}
    if k in goals:
        pick, win, lose = goals[k]
        how = (f"Open the 'Total goals' (Over/Under) market and choose '{pick}'. "
               f"You WIN if the match has {win} in total (both teams added together). "
               f"You LOSE if it has {lose}.")
    else:
        how = {
            "H": f"Open the 'Match result' (1X2) market and choose '1' ({home} to win). "
                 f"You WIN only if {home} wins. You LOSE if it is a draw or {away} wins.",
            "D": "Open the 'Match result' (1X2) market and choose 'X' (draw). You WIN "
                 "only if the match ends level. You LOSE if either team wins.",
            "A": f"Open the 'Match result' (1X2) market and choose '2' ({away} to win). "
                 f"You WIN only if {away} wins. You LOSE if it is a draw or {home} wins.",
            "1X": f"Open the 'Double chance' market and choose '1X' ({home} or draw). "
                  f"You WIN if {home} wins OR it is a draw. You LOSE only if {away} wins.",
            "X2": f"Open the 'Double chance' market and choose 'X2' ({away} or draw). "
                  f"You WIN if {away} wins OR it is a draw. You LOSE only if {home} wins.",
            "12": f"Open the 'Double chance' market and choose '12' (either team wins). "
                  f"You WIN if {home} wins OR {away} wins - it does not matter which. "
                  "You LOSE only if the match is a draw.",
            "BTTS_Y": "Open the 'Both teams to score' (GG/NG) market and choose 'Yes' "
                      f"(GG). You WIN if {home} AND {away} each score at least 1 goal. "
                      "You LOSE if either team scores 0.",
            "BTTS_N": "Open the 'Both teams to score' (GG/NG) market and choose 'No' "
                      "(NG). You WIN if at least one team scores 0 goals. You LOSE if "
                      "both teams score.",
            "HS": f"Open the '{home} total goals' (team goals) market and choose 'Over 0.5' "
                  f"(some bookmakers call it '{home} to score: Yes'). You WIN if {home} "
                  f"scores at least 1 goal - the final result does not matter. You LOSE "
                  f"only if {home} scores 0.",
            "AS": f"Open the '{away} total goals' (team goals) market and choose 'Over 0.5' "
                  f"(some bookmakers call it '{away} to score: Yes'). You WIN if {away} "
                  f"scores at least 1 goal - the final result does not matter. You LOSE "
                  f"only if {away} scores 0.",
        }[k]
    return (how + f" Only place it if your bookmaker's odds for this choice are "
            f"{floor:.2f} or higher.")


def score_predictions(window: pd.DataFrame, model: dict,
                      injuries: pd.DataFrame | None = None,
                      book_odds: pd.DataFrame | None = None) -> pd.DataFrame:
    """Predicted score, top-5 scores and key probabilities per fixture
    (adjusted for missing players when that data is available)."""
    rows = []
    for _, fx in window.iterrows():
        adj, missing, key_h, key_a = fixture_injuries_full(fx, injuries, model)
        lh, la, m, p = predict(model, fx["HomeTeam"], fx["AwayTeam"], adj)
        has_odds = bool(raw_odds(fx))
        pred, pp = predicted_score(m, p)
        q_ = pick_probs(m, model["rho"], raw_odds(fx))[0]
        bp = best_pick(q_, fx["HomeTeam"], fx["AwayTeam"])
        rows.append({
            "League": fx.get("Div", fx.get("League", DEFAULT_LEAGUE)),
            "Kickoff (UK)": fx["Kickoff"].tz_convert(UK_TZ).strftime("%a %d %b %H:%M"),
            "Kickoff UTC": fx["Kickoff"],
            "Match": f"{fx['HomeTeam']} v {fx['AwayTeam']}",
            "Best pick": bp["text"], "Best pick %": bp["prob"] * 100, "How to bet": bp["how"],
            "Pick key": bp["key"], "Missing players": missing,
            "Key out H": key_h, "Key out A": key_a, "Odds used": has_odds,
            **dict(zip(("1xBet odds", "Best odds"), book_prices(fx, book_odds, bp["key"]))),
            "U45 %": q_["U45"] * 100, "U55 %": q_["U55"] * 100,
            "1xBet U4.5": book_prices(fx, book_odds, "U45")[0],
            "1xBet U5.5": book_prices(fx, book_odds, "U55")[0],
            "Predicted score": pred,
            "Top 5 scores": ", ".join(f"{sc} ({pr*100:.0f}%)" for sc, pr in top_scores(m)),
            "Home %": p["H"] * 100, "Draw %": p["D"] * 100, "Away %": p["A"] * 100,
            **goal_line_cols(p), "BTTS %": p["BTTS_Y"] * 100,
            "xG": f"{lh:.2f} - {la:.2f}",
            "NoOdds": str(fx.get("NoOdds", "")) == "True",
            "_best": max(("H", "D", "A"), key=lambda k: p[k]),
            "_pbest": max(p["H"], p["D"], p["A"]),
        })
    return pd.DataFrame(rows)


def stake_pct(edge_pct: float, net_odds_: float) -> float:
    """Quarter-Kelly stake as % of bankroll, capped (protects against
    over-betting when the model is wrong, which it often is)."""
    if net_odds_ <= 1 or edge_pct <= 0:
        return 0.0
    kelly = (edge_pct / 100) / (net_odds_ - 1)
    return min(kelly * DEFAULTS["kelly_fraction"] * 100, DEFAULTS["max_stake_pct"])


def match_advice(preds: pd.DataFrame, flags: pd.DataFrame, bankroll: float,
                 allow_all: bool = False) -> pd.DataFrame:
    """
    One plain-English verdict per fixture, depending on the league's tested
    status: BET (with stake), WATCH (paper-trade only) or NO BET.
    Most matches should be NO BET - that is normal.
    """
    preds = preds.copy()
    advice, stakes, status_col = [], [], []
    for _, r in preds.iterrows():
        lg = r.get("League", DEFAULT_LEAGUE)
        status = league_status(lg, allow_all)
        status_col.append(STATUS_LABEL[status])
        home, away = r["Match"].split(" v ", 1)
        f = flags
        if not flags.empty:
            f = flags[(flags["HomeTeam"] == home) & (flags["AwayTeam"] == away)]
            if "League" in flags.columns:
                f = f[f["League"] == lg]
        if f.empty:
            if status == "predict":
                advice.append("NO BET (model not profitable in this league)")
            elif r.get("NoOdds", False):
                lab = {"H": home, "D": "Draw", "A": away}[r["_best"]]
                floor = (1 / r["_pbest"]) * (1 + DEFAULTS["min_edge"] / 100)
                advice.append(f"Odds not out yet: back {lab} only at {floor:.2f}+ "
                              "(re-check when odds appear)")
            else:
                advice.append("NO BET (price is fair)")
            stakes.append("")
            continue
        b = f.sort_values("Edge %", ascending=False).iloc[0]
        what = (f"{MARKET_LABELS[b['Market']]} @ {b['Price']:.2f} "
                f"({b['Book']}, edge +{b['Edge %']:.1f}%)")
        if status == "bet":
            pct = stake_pct(b["Edge %"], b["Net odds"])
            advice.append(f"BET {what}")
            stakes.append(f"{pct:.1f}% = {bankroll * pct / 100:,.2f}")
        elif status == "watch":
            advice.append(f"WATCH {what} - paper only")
            stakes.append("")
        else:
            advice.append("NO BET (model not profitable in this league)")
            stakes.append("")
    preds.insert(1, "Status", status_col)
    preds.insert(4, "Advice", advice)
    preds.insert(5, "Stake", stakes)
    return preds.drop(columns=["NoOdds", "_best", "_pbest"], errors="ignore")


def weekly_overview(fixtures: pd.DataFrame, all_results: dict, half_life: int,
                    goals_weight: float, book: str, min_edge: float,
                    max_edge: float, bankroll: float, allow_all: bool = False,
                    injuries: pd.DataFrame | None = None,
                    book_odds: pd.DataFrame | None = None):
    """
    Predictions + advice for every upcoming fixture in every covered league.
    Returns (advice table, value flags, {league: model}).
    """
    window = upcoming_fixtures(fixtures)
    adv_frames, flag_frames, models = [], [], {}
    if window.empty:
        return pd.DataFrame(), pd.DataFrame(columns=PAPER_COLS), models
    for lg in LEAGUES:
        w = window[window["Div"] == lg]
        res = all_results.get(lg)
        if w.empty or res is None or res.empty:
            continue
        try:
            model = fit_model(res, half_life, goals_weight)
            flags = flagged_bets(w, model, book, min_edge, max_edge)
            adv_frames.append(match_advice(score_predictions(w, model, injuries, book_odds), flags,
                                           bankroll, allow_all))
        except Exception:
            continue                       # one broken league never blocks the rest
        models[lg] = model
        flag_frames.append(flags)
    if not adv_frames:
        return pd.DataFrame(), pd.DataFrame(columns=PAPER_COLS), models
    adv = pd.concat(adv_frames, ignore_index=True)
    adv["_rank"] = adv["League"].map(lambda c: STATUS_RANK[league_status(c, allow_all)])
    adv = adv.sort_values(["_rank", "League", "Kickoff (UK)"]).drop(columns="_rank")
    adv.insert(0, "Competition", adv["League"].map(league_name))
    flags = pd.concat(flag_frames, ignore_index=True) if flag_frames else \
        pd.DataFrame(columns=PAPER_COLS)
    return adv.reset_index(drop=True), flags, models


SAFE_RULES = (
    "**🛡️ Rules for low-odds betting (read every time)**\n"
    "1. **Bet singles** (one match per bet). If you combine, use at most 2-3 picks: "
    "4 picks at 85% all win only ~52% of the time.\n"
    "2. **Only bet when your bookmaker's odds are at least the 'bet at' number.** "
    "Below it, you lose money over time even if most bets win.\n"
    "3. **Prefer 🔒 Banker and ⭐⭐⭐ Strong picks, and league matches** (tested on "
    "the most games). Skip ⭐ Fair picks unless the odds are clearly above 'bet at'.\n"
    "4. **Stake the same small amount every time** - about 1-2% of your betting money "
    "per pick. Never raise stakes to win back a loss.\n"
    "5. **Expect losses.** At odds of 1.20 you must win 5 bets out of 6 just to break "
    "even; one loss cancels about five wins. Judge the system on 50+ bets, not one day.\n"
    "6. **Check team news** before kick-off; skip a pick if a key player is missing.")

PICK_COLS = ["Kickoff (UTC)", "Match day", "League", "Competition", "HomeTeam", "AwayTeam",
             "Pick key", "Best pick", "Prob %", "Bet at", "Tier", "Check", "xG"]
PICK_KEY = ["League", "Match day", "HomeTeam", "AwayTeam"]


def check_passed(log: pd.DataFrame) -> pd.Series:
    """True where a logged pick passed every safety check. Passing picks are
    saved as 'OK'; older rows saved a blank (read back as NaN) - those count as
    passed when the row is from after checks were logged (it has an xG value)."""
    if log is None or log.empty or "Check" not in log.columns:
        return pd.Series(False, index=getattr(log, "index", None))
    c = log["Check"]
    has_xg = log["xG"].notna() if "xG" in log.columns else pd.Series(False, index=log.index)
    return c.astype(str).isin(["OK", ""]) | (c.isna() & has_xg)


def pick_won(k: str, h: int, a: int) -> bool:
    """Did best-pick market k win with final score h-a?"""
    t = h + a
    if k[:1] in "OU" and k[1:].isdigit():
        line = int(k[1:]) / 10
        return t > line if k[0] == "O" else t < line
    return {"H": h > a, "D": h == a, "A": h < a, "1X": h >= a, "X2": h <= a, "12": h != a,
            "BTTS_Y": h > 0 and a > 0, "BTTS_N": not (h > 0 and a > 0),
            "HS": h > 0, "AS": a > 0}[k]


def picks_from_overview(adv: pd.DataFrame) -> pd.DataFrame:
    """One log row per upcoming match: the system's best pick, as shown."""
    if adv is None or adv.empty or "Pick key" not in adv.columns:
        return pd.DataFrame(columns=PICK_COLS)
    d = adv.dropna(subset=["Pick key", "Best pick %"]).copy()
    ko = pd.to_datetime(d["Kickoff UTC"], utc=True)
    teams = d["Match"].str.split(" v ", n=1, expand=True)
    prob = d["Best pick %"] / 100
    return pd.DataFrame({
        "Kickoff (UTC)": ko.dt.strftime("%Y-%m-%d %H:%M"),
        "Match day": ko.dt.strftime("%Y-%m-%d"), "League": d["League"].astype(str),
        "Competition": d["Competition"], "HomeTeam": teams[0], "AwayTeam": teams[1],
        "Pick key": d["Pick key"], "Best pick": d["Best pick"],
        "Prob %": (prob * 100).round(1),
        "Bet at": ((1 / prob) * (1 + DEFAULTS["min_edge"] / 100)).round(2),
        "Tier": [next(t for lim, _, t, _ in BEST_PICK_TIERS if q >= lim) for q in prob],
        "Check": [ready_filter(r) or "OK" for _, r in d.iterrows()],
        "xG": d["xG"] if "xG" in d.columns else "",
    })[PICK_COLS]


def read_pick_log() -> pd.DataFrame:
    try:
        if PICK_LOG.exists():
            return pd.read_csv(PICK_LOG, dtype={"League": str})
    except Exception:
        pass
    return pd.DataFrame(columns=PICK_COLS)


def update_pick_log(adv: pd.DataFrame, now=None):
    """Record every best pick. Until kick-off the newest pick replaces the old
    one (odds and news change); after kick-off it is frozen - that is the pick
    the system really gave. Returns (log, n_new, error)."""
    now = now or pd.Timestamp.now(tz="UTC")
    log = read_pick_log()
    new = picks_from_overview(adv)
    if new.empty:
        return log, 0, None
    new = new[pd.to_datetime(new["Kickoff (UTC)"], utc=True) > now]
    if not log.empty:
        started = pd.to_datetime(log["Kickoff (UTC)"], utc=True) <= now
        frozen = log[started]
        open_ = log[~started]
    else:
        frozen = open_ = log
    keys = set(map(tuple, frozen[PICK_KEY].astype(str).values)) if not frozen.empty else set()
    new = new[[tuple(r) not in keys for r in new[PICK_KEY].astype(str).values]]
    old_keys = set(map(tuple, log[PICK_KEY].astype(str).values)) if not log.empty else set()
    n_new = sum(tuple(r) not in old_keys for r in new[PICK_KEY].astype(str).values)
    # open picks not in this run (e.g. feed gap) are kept as they were
    new_keys = set(map(tuple, new[PICK_KEY].astype(str).values))
    keep_open = open_[[tuple(r) not in new_keys for r in open_[PICK_KEY].astype(str).values]] \
        if not open_.empty else open_
    out = pd.concat([frozen, keep_open, new], ignore_index=True).reindex(columns=PICK_COLS)
    out = out.sort_values("Kickoff (UTC)").reset_index(drop=True)
    try:
        PICK_LOG.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=PICK_LOG.parent, suffix=".tmp")
        with os.fdopen(fd, "w", newline="") as f:
            out.to_csv(f, index=False)
        os.replace(tmp, PICK_LOG)
    except Exception as exc:
        return log, 0, str(exc)
    return out, n_new, None


def result_frames(all_results: dict, *cup_frames) -> pd.DataFrame:
    """Every finished match as League / Date / HomeTeam / AwayTeam / FTHG / FTAG."""
    frames = []
    for lg, df in (all_results or {}).items():
        if df is not None and not df.empty:
            frames.append(pd.DataFrame({"League": lg, "Date": pd.to_datetime(df["Date"]),
                                        "HomeTeam": df["HomeTeam"], "AwayTeam": df["AwayTeam"],
                                        "FTHG": df["FTHG"], "FTAG": df["FTAG"]}))
    for cf in cup_frames:
        if cf is None or cf.empty or "Comp" not in cf.columns:
            continue
        c = cf.dropna(subset=["FTHG", "FTAG"])
        frames.append(pd.DataFrame({
            "League": c["Comp"].astype(str),
            "Date": pd.to_datetime(c["Kickoff"], utc=True).dt.tz_localize(None),
            "HomeTeam": c["HomeTeam"], "AwayTeam": c["AwayTeam"],
            "FTHG": c["FTHG"], "FTAG": c["FTAG"]}))
    if not frames:
        return pd.DataFrame(columns=["League", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    return pd.concat(frames, ignore_index=True)


def settle_pick_log(log: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """Attach the final score and whether each logged best pick won."""
    if log is None or log.empty:
        return pd.DataFrame(columns=PICK_COLS + ["Score", "Won"])
    out = log.copy()
    out["Score"] = "pending"
    out["Won"] = pd.Series([None] * len(out), index=out.index, dtype="object")
    if results is None or results.empty:
        return out
    r = results.copy()
    r["League"] = r["League"].astype(str)
    m = out.reset_index().merge(r, on=["League", "HomeTeam", "AwayTeam"], how="inner")
    if m.empty:
        return out
    gap = (pd.to_datetime(m["Date"]).dt.normalize()
           - pd.to_datetime(m["Match day"])).abs()
    m = m[gap <= pd.Timedelta(days=2)].drop_duplicates("index")
    for _, row in m.iterrows():
        h, a = int(row["FTHG"]), int(row["FTAG"])
        out.loc[row["index"], "Score"] = f"{h}-{a}"
        out.loc[row["index"], "Won"] = bool(pick_won(row["Pick key"], h, a))
    return out


def pick_summary(settled: pd.DataFrame) -> dict:
    """Win rate of the system's real (logged before kick-off) best picks."""
    if settled is None or settled.empty or "Won" not in settled.columns:
        return {"n": 0}
    done = settled[settled["Won"].notna()].copy()
    if done.empty:
        return {"n": 0, "pending": int(len(settled))}
    done["Won"] = done["Won"].astype(bool)
    tiers = {t: (int(g["Won"].sum()), int(len(g))) for t, g in done.groupby("Tier")}
    strong = done[done["Tier"].isin(["Banker", "Strong"])]
    chk = done[check_passed(done)]
    return {"n": int(len(done)), "won": int(done["Won"].sum()),
            "pct": 100 * done["Won"].mean(), "tiers": tiers,
            "strong": (int(strong["Won"].sum()), int(len(strong))),
            "checked": (int(chk["Won"].sum()), int(len(chk))),
            "pending": int(settled["Won"].isna().sum()),
            "recent": done.sort_values("Kickoff (UTC)", ascending=False).head(20)}


def pick_record_lines(sm: dict) -> list:
    """Plain-text track record (for predictions.md and the console)."""
    if not sm.get("n"):
        return ["Track record starts now: every best pick is saved before kick-off and "
                "marked won/lost after the match. Check back after the next matchday."]
    order = ["Banker", "Strong", "Good", "Fair"]
    tiers = " · ".join(f"{t} {w}/{n} ({100*w/n:.0f}%)" for t in order
                       for w, n in [sm["tiers"].get(t, (0, 0))] if n)
    sw, sn = sm["strong"]
    cw, cn = sm.get("checked", (0, 0))
    return [f"**{sm['won']} of {sm['n']} best picks won ({sm['pct']:.0f}%)**"
            + (f" · ✅ 'Your bets' (passed all checks): {cw}/{cn} ({100*cw/cn:.0f}%)" if cn else "")
            + (f" · 🔒+⭐⭐⭐ picks: {sw}/{sn} ({100*sw/sn:.0f}%)" if sn else "")
            + f" · {sm['pending']} waiting for results", tiers]


def short_bet(k: str, home: str, away: str) -> str:
    """What to select on the betting site, in its own words."""
    goals = {"O15": "Over 1.5", "U15": "Under 1.5", "O25": "Over 2.5", "U25": "Under 2.5",
             "O35": "Over 3.5", "U35": "Under 3.5", "U45": "Under 4.5"}
    if k in goals:
        return f"Over/Under (total goals) → {goals[k]}"
    return {"H": f"1X2 → 1 ({home} wins)", "D": "1X2 → X (draw)",
            "A": f"1X2 → 2 ({away} wins)",
            "1X": f"Double Chance → 1X ({home} or draw)",
            "X2": f"Double Chance → X2 ({away} or draw)",
            "12": "Double Chance → 12 (no draw)",
            "BTTS_Y": "Both teams to score → Yes (GG)",
            "BTTS_N": "Both teams to score → No (NG)",
            "HS": f"{home} total goals → Over 0.5",
            "AS": f"{away} total goals → Over 0.5"}[k]


READY_MIN_P = 0.85            # 'Good' or better
READY_MIN_P_NO_ODDS = 0.88    # cup / national games without bookmaker odds


def ready_filter(row) -> str:
    """'' if the pick passes every automatic safety check, else the reason."""
    p = row.get("Best pick %", np.nan) / 100
    k = row.get("Pick key")
    if not isinstance(k, str) or not np.isfinite(p):
        return "no pick"
    if p < READY_MIN_P:
        return "below 85%"
    if not bool(row.get("Odds used", False)) and p < READY_MIN_P_NO_ODDS:
        return "no bookmaker odds and below 88%"
    mine = row.get("1xBet odds", np.nan)
    floor = (1 / p) * (1 + DEFAULTS["min_edge"] / 100)
    if pd.notna(mine) and float(mine) < floor - 1e-9:
        return f"1xBet pays only {float(mine):.2f} (needs {floor:.2f})"
    kh, ka = int(row.get("Key out H", 0) or 0), int(row.get("Key out A", 0) or 0)
    backs_home = k in ("H", "1X", "HS")
    backs_away = k in ("A", "X2", "AS")
    if backs_home and kh > 0:
        return "key player missing for the team backed"
    if backs_away and ka > 0:
        return "key player missing for the team backed"
    if k == "12" and kh + ka > 1:
        return "several key players missing"
    if k in ("O15", "O25", "O35", "BTTS_Y") and kh + ka > 1:
        return "key attackers missing in a goals bet"
    if kh + ka > 3:
        return "too many key players missing"
    return ""


def ready_picks_by_day(adv: pd.DataFrame, tz: str, n: int = 5) -> list:
    """[(day, DataFrame)] - every day's n safest picks that pass all automatic
    checks, highest win chance first, with the exact selection to make."""
    if adv is None or adv.empty or "Best pick %" not in adv.columns:
        return []
    d = add_local_time(adv, tz)
    d = d[pd.to_datetime(d["Kickoff UTC"], utc=True) > pd.Timestamp.now(tz="UTC")]
    d = d.dropna(subset=["Best pick %"]).copy()
    if d.empty:
        return []
    d["Skip reason"] = [ready_filter(r) for _, r in d.iterrows()]
    d = d[d["Skip reason"] == ""]
    if d.empty:
        return []
    teams = d["Match"].str.split(" v ", n=1, expand=True)
    d["Bet"] = [short_bet(k, h, a) for k, h, a in zip(d["Pick key"], teams[0], teams[1])]
    d["Min odds"] = (100 / d["Best pick %"]) * (1 + DEFAULTS["min_edge"] / 100)
    out = []
    for day in sorted(d["_day"].unique()):
        top = (d[d["_day"] == day].sort_values(["Best pick %", "Kickoff UTC"],
                                                ascending=[False, True]).head(n).copy())
        top.insert(0, "Rank", range(1, len(top) + 1))
        out.append((day, top))
    return out


def all_matches_by_day(adv: pd.DataFrame, tz: str) -> list:
    """[(day, DataFrame)] - EVERY upcoming match with the one bet the system
    would make, whatever the odds, plus whether it passes the safety checks."""
    if adv is None or adv.empty or "Pick key" not in adv.columns:
        return []
    d = add_local_time(adv, tz)
    d = d[pd.to_datetime(d["Kickoff UTC"], utc=True) > pd.Timestamp.now(tz="UTC")]
    d = d.dropna(subset=["Best pick %", "Pick key"]).copy()
    if d.empty:
        return []
    teams = d["Match"].str.split(" v ", n=1, expand=True)
    d["Bet"] = [short_bet(k, h, a) for k, h, a in zip(d["Pick key"], teams[0], teams[1])]
    d["Min odds"] = (100 / d["Best pick %"]) * (1 + DEFAULTS["min_edge"] / 100)
    d["Check"] = [ready_filter(r) for _, r in d.iterrows()]
    out = []
    for day in sorted(d["_day"].unique()):
        g = d[d["_day"] == day].sort_values(["Best pick %", "Kickoff UTC"],
                                            ascending=[False, True])
        out.append((day, g))
    return out


def app_apifootball_key() -> str:
    """API-Football key for the app itself (Streamlit secret or environment)."""
    try:
        k = st.secrets.get("APIFOOTBALL_KEY", "")
    except Exception:
        k = ""
    return str(k or os.environ.get("APIFOOTBALL_KEY", "")).strip()


@st.cache_data(ttl=60, show_spinner=False)
def live_fixtures(key: str) -> pd.DataFrame:
    """Matches being played right now in our competitions (refreshed every minute)."""
    js = _apifootball_get("/fixtures", key, live="all") or {}
    rows = []
    for f in js.get("response") or []:
        if (f.get("league") or {}).get("id") not in AF_LEAGUES:
            continue
        fx, g = f.get("fixture") or {}, f.get("goals") or {}
        rows.append({"FixtureId": fx.get("id"), "Kickoff": fx.get("date"),
                     "HomeTeam": f["teams"]["home"]["name"], "AwayTeam": f["teams"]["away"]["name"],
                     "Minute": (fx.get("status") or {}).get("elapsed") or 0,
                     "Status": (fx.get("status") or {}).get("short"),
                     "GH": g.get("home") or 0, "GA": g.get("away") or 0})
    df = pd.DataFrame(rows)
    if not df.empty:
        df["_ko"] = pd.to_datetime(df["Kickoff"], utc=True, errors="coerce")
    return df


def inplay_chance(k: str, lam_h: float, lam_a: float, minute: int, gh: int, ga: int,
                  status: str = "") -> float:
    """Chance the pick still wins from the current score, assuming goals keep
    coming at the pre-match rate for the time left (about 95 minutes in all)."""
    if status == "HT":
        minute = 45
    left = max(0.0, (95.0 - float(minute)) / 95.0)
    g = np.arange(11)
    ph, pa = poisson.pmf(g, lam_h * left), poisson.pmf(g, lam_a * left)
    m = np.outer(ph, pa)
    won = np.array([[pick_won(k, gh + i, ga + j) for j in g] for i in g])
    return float((m * won).sum() / m.sum())


def _parse_xg(x) -> tuple:
    try:
        a, b = str(x).split(" - ")
        return float(a), float(b)
    except Exception:
        return np.nan, np.nan


def past_matches_by_day(settled: pd.DataFrame, tz: str, days_back: int = 14,
                        live: pd.DataFrame | None = None) -> dict:
    """{day: DataFrame} of matches already started, with the bet the system gave
    before kick-off (frozen in the pick log) and the result."""
    if settled is None or settled.empty:
        return {}
    d = settled.copy()
    ko = pd.to_datetime(d["Kickoff (UTC)"], utc=True, errors="coerce")
    now = pd.Timestamp.now(tz="UTC")
    d = d[(ko <= now) & (ko > now - pd.Timedelta(days=days_back))].copy()
    if d.empty:
        return {}
    local = pd.to_datetime(d["Kickoff (UTC)"], utc=True).dt.tz_convert(tz)
    d["_day"], d["Time"] = local.dt.date, local.dt.strftime("%H:%M")
    d["Match"] = d["HomeTeam"] + " v " + d["AwayTeam"]
    d["Bet"] = [short_bet(k, h, a) for k, h, a in zip(d["Pick key"], d["HomeTeam"], d["AwayTeam"])]
    d["Best pick %"] = pd.to_numeric(d["Prob %"], errors="coerce")
    known = d["Check"].notna() | (d["xG"].notna() if "xG" in d.columns else False) \
        if "Check" in d.columns else pd.Series(False, index=d.index)
    d["Checked"] = np.where(check_passed(d), "✅", np.where(known, "⚠️", ""))
    d["Result"] = [("✅ WON " if w is True else "❌ LOST " if w is False else "") + str(sc)
                   if sc != "pending" else "⏳ playing / waiting for result"
                   for w, sc in zip(d["Won"], d["Score"])]
    if live is not None and not live.empty:
        for idx, r in d[d["Score"] == "pending"].iterrows():
            lv = match_fixture_row({"Kickoff": pd.Timestamp(r["Kickoff (UTC)"], tz="UTC"),
                                    "HomeTeam": r["HomeTeam"], "AwayTeam": r["AwayTeam"]}, live)
            if lv is None:
                continue
            gh, ga, mi = int(lv["GH"]), int(lv["GA"]), int(lv["Minute"])
            lh, la = _parse_xg(r.get("xG"))
            txt = f"🔴 LIVE {'HT' if lv['Status'] == 'HT' else str(mi) + chr(39)} · {gh}-{ga}"
            if np.isfinite(lh) and np.isfinite(la):
                c = inplay_chance(r["Pick key"], lh, la, mi, gh, ga, lv["Status"])
                txt += (" · ✅ already won" if c > 0.999 else " · ❌ can no longer win"
                        if c < 0.001 else f" · chance now {100*c:.0f}%")
            d.loc[idx, "Result"] = txt
    return {day: g.sort_values(["Best pick %", "Kickoff (UTC)"], ascending=[False, True])
            for day, g in d.groupby("_day")}


def past_cards_html(g: pd.DataFrame) -> str:
    e = html.escape
    out = []
    for _, r in g.iterrows():
        cls = "bet" if str(r["Result"]).startswith("✅") else "watch" if str(r["Result"]).startswith("❌") else ""
        out.append(f"<div class='bfe-card {cls}'>"
                   f"<div class='bfe-top'>{e(str(r['Time']))} · {e(str(r['Competition']))} "
                   f"{e(str(r['Checked']))}</div>"
                   f"<div class='bfe-match'>{e(str(r['Match']))}</div>"
                   f"<div class='bfe-row'>Bet given before kick-off: <b>{e(str(r['Bet']))}</b> "
                   f"({r['Best pick %']:.0f}%)</div>"
                   f"<div class='bfe-adv'>{e(str(r['Result']))}</div></div>")
    return "".join(out)


def day_result_line(g: pd.DataFrame) -> str:
    done = g[g["Won"].isin([True, False])]
    if done.empty:
        return "⏳ No finished results for this day yet."
    w = int((done["Won"] == True).sum())  # noqa: E712
    chk = done[done["Checked"] == "✅"]
    cw = int((chk["Won"] == True).sum())  # noqa: E712
    return (f"Results: **{w} of {len(done)} bets won** ({100*w/len(done):.0f}%)"
            + (f" · ✅ checked bets: **{cw} of {len(chk)} won**" if len(chk) else "")
            + (f" · {len(g) - len(done)} still waiting" if len(g) > len(done) else ""))


def all_matches_cards_html(g: pd.DataFrame) -> str:
    e = html.escape
    cards = []
    for _, r in g.iterrows():
        ok = r["Check"] == ""
        price = r.get("1xBet odds", np.nan)
        cards.append(
            f"<div class='bfe-card {'bet' if ok else ''}'>"
            f"<div class='bfe-top'>{e(str(r['Time']))} · {e(str(r['Competition']))}</div>"
            f"<div class='bfe-match'>{e(str(r['Match']))}</div>"
            f"<div class='bfe-adv'>BET: {e(str(r['Bet']))}</div>"
            f"<div class='bfe-row'>{r['Best pick %']:.0f}% chance · fair minimum odds "
            f"{r['Min odds']:.2f}"
            + (f" · 1xBet {float(price):.2f}" if pd.notna(price) else "") + "</div>"
            + (f"<div class='bfe-row bfe-small'>✅ passes all checks (also in 'Your bets')</div>"
               if ok else
               f"<div class='bfe-row bfe-small'>⚠️ riskier: {e(str(r['Check']))}</div>")
            + "</div>")
    return "".join(cards)


UNDER_MIN_P = 80.0


def under_picks_by_day(adv: pd.DataFrame, tz: str, min_p: float = UNDER_MIN_P) -> list:
    """[(day, DataFrame)] - every upcoming match where Under 4.5 (preferred: it
    pays more) or else Under 5.5 has at least min_p % chance, safest first."""
    if adv is None or adv.empty or "U45 %" not in adv.columns:
        return []
    d = add_local_time(adv, tz)
    d = d[pd.to_datetime(d["Kickoff UTC"], utc=True) > pd.Timestamp.now(tz="UTC")].copy()
    if d.empty:
        return []
    use45 = d["U45 %"] >= min_p
    use55 = ~use45 & (d["U55 %"] >= min_p)
    d = d[use45 | use55].copy()
    if d.empty:
        return []
    is45 = d["U45 %"] >= min_p
    d["Bet"] = np.where(is45, "Over/Under → Under 4.5", "Over/Under → Under 5.5")
    d["Chance %"] = np.where(is45, d["U45 %"], d["U55 %"])
    d["1xBet odds"] = np.where(is45, d.get("1xBet U4.5", np.nan), d.get("1xBet U5.5", np.nan))
    d["Fair min odds"] = (100 / d["Chance %"]) * (1 + DEFAULTS["min_edge"] / 100)
    d["Value"] = np.where(pd.isna(d["1xBet odds"]), "",
                          np.where(d["1xBet odds"] >= d["Fair min odds"] - 1e-9, "💚", ""))
    out = []
    for day in sorted(d["_day"].unique()):
        g = d[d["_day"] == day].sort_values(["Chance %", "Kickoff UTC"], ascending=[False, True])
        g.insert(0, "Rank", range(1, len(g) + 1))
        out.append((day, g))
    return out


def under_ticket_steps(g: pd.DataFrame) -> pd.DataFrame:
    """Taking the top 1, 2, 3 ... matches: total odds (1xBet where known) and
    the chance the whole ticket wins."""
    rows, odds, p, known = [], 1.0, 1.0, True
    for _, r in g.iterrows():
        p *= r["Chance %"] / 100
        if pd.notna(r["1xBet odds"]):
            odds *= float(r["1xBet odds"])
        else:
            known = False
        rows.append({"Matches": int(r["Rank"]),
                     "Total odds (1xBet)": odds if known else np.nan,
                     "Ticket wins %": 100 * p})
    return pd.DataFrame(rows)


def under_cards_html(g: pd.DataFrame) -> str:
    e = html.escape
    out = []
    for _, r in g.iterrows():
        price = r["1xBet odds"]
        out.append(
            "<div class='bfe-card bet'>"
            f"<div class='bfe-top'><b class='bfe-rank'>#{int(r['Rank'])}</b> "
            f"{e(str(r['Time']))} · {e(str(r['Competition']))}</div>"
            f"<div class='bfe-match'>{e(str(r['Match']))}</div>"
            f"<div class='bfe-adv'>✅ BET: {e(str(r['Bet']))}</div>"
            f"<div class='bfe-row'><b>{r['Chance %']:.0f}% chance</b> · fair minimum odds "
            f"{r['Fair min odds']:.2f}"
            + (f" · 1xBet {float(price):.2f} {r['Value']}" if pd.notna(price) else "")
            + "</div></div>")
    return "".join(out)


def render_under_section(adv: pd.DataFrame, tz: str):
    st.markdown("#### 🟢 Under 4.5 / Under 5.5 — matches with 80%+ chance")
    st.caption("Every match of the day where 'Under 4.5 goals' (or, if that is below 80%, "
               "'Under 5.5 goals') has at least an 80% chance, safest first. Under 4.5 = the "
               "match ends with 4 goals or fewer in total; Under 5.5 = 5 goals or fewer. "
               "💚 = 1xBet pays at least the fair minimum. The more matches on one ticket, "
               "the lower the chance that ALL win - see the ticket table.")
    days = under_picks_by_day(adv, tz)
    if not days:
        st.info("No matches with an 80%+ Under bet yet. The list updates four times a day.")
        return
    labels = {day: f"{pd.Timestamp(day):%A %d %B} — {len(g)} matches" for day, g in days}
    day = st.selectbox("Day", [d_ for d_, _ in days], format_func=labels.get, key="under_day")
    g = dict(days)[day]
    comps = ["All competitions"] + sorted(g["Competition"].unique())
    comp = st.selectbox("Competition", comps, key="under_comp")
    if comp != comps[0]:
        g = g[g["Competition"] == comp].copy()
        g["Rank"] = range(1, len(g) + 1)
    if phone_mode():
        st.markdown(under_cards_html(g), unsafe_allow_html=True)
    else:
        show_df(pd.DataFrame({
            "#": g["Rank"], "Time": g["Time"], "Competition": g["Competition"],
            "Match": g["Match"], "Bet": g["Bet"], "Chance %": g["Chance %"],
            "Fair min odds": g["Fair min odds"], "1xBet odds": g["1xBet odds"],
            "💚": g["Value"]}).style.format(
            {"Chance %": "{:.0f}", "Fair min odds": "{:.2f}", "1xBet odds": "{:.2f}"},
            na_rep="—"), hide_index=True, height=min(900, 38 + 35 * len(g)))
    st.markdown("**🎟️ If you put the top matches on one ticket**")
    steps = under_ticket_steps(g)
    show_df(steps.style.format({"Total odds (1xBet)": "{:.2f}", "Ticket wins %": "{:.0f}"},
                               na_rep="price not known"), hide_index=True,
            height=min(500, 38 + 35 * len(steps)))
    st.caption("Example: the row '5' = the top 5 matches on one ticket. Under bets can lose "
               "together on a high-scoring day, so a few small tickets are safer than one big one.")


def ready_cards_html(top: pd.DataFrame) -> str:
    """Big, simple cards: what to select and the minimum odds - nothing else."""
    e = html.escape
    cards = []
    for _, r in top.iterrows():
        cards.append(
            "<div class='bfe-card bet'>"
            f"<div class='bfe-top'><b class='bfe-rank'>#{int(r['Rank'])}</b> "
            f"{e(str(r['Time']))} · {e(str(r['Competition']))}</div>"
            f"<div class='bfe-match'>{e(str(r['Match']))}</div>"
            f"<div class='bfe-adv'>✅ BET: {e(str(r['Bet']))}</div>"
            f"<div class='bfe-row'>Only if the odds are <b>{r['Min odds']:.2f} or higher</b> · "
            f"{r['Best pick %']:.0f}% chance</div>"
            + (f"<div class='bfe-row'>💚 1xBet pays <b>{float(r['1xBet odds']):.2f}</b> - "
               f"good to go on 1xBet</div>" if pd.notna(r.get("1xBet odds", np.nan)) else
               "<div class='bfe-row bfe-small'>1xBet / betPawa price not known - check it "
               "is at least the number above</div>")
            + "</div>")
    return "".join(cards)


def top_picks_by_day(adv: pd.DataFrame, tz: str, n: int = 5) -> list:
    """[(day, DataFrame)] - for every upcoming day, its n best picks ranked from
    the highest to the lowest win probability (matches not yet kicked off)."""
    if adv is None or adv.empty or "Best pick %" not in adv.columns:
        return []
    d = add_local_time(adv, tz)
    d = d[pd.to_datetime(d["Kickoff UTC"], utc=True) > pd.Timestamp.now(tz="UTC")]
    d = d.dropna(subset=["Best pick %"])
    out = []
    for day in sorted(d["_day"].unique()):
        top = (d[d["_day"] == day].sort_values(["Best pick %", "Kickoff UTC"],
                                                ascending=[False, True]).head(n).copy())
        top.insert(0, "Rank", range(1, len(top) + 1))
        out.append((day, top))
    return out


# Walk-forward test, 12 leagues, 2024/25 + 2025/26 (472 match days): how often
# the day's top-N best picks ALL won on one ticket.
TESTED_TICKET = {1: 89, 2: 83, 3: 79, 4: 72, 5: 67}


def ticket_line(top: pd.DataFrame) -> str:
    """What happens if all of a day's top picks go on one ticket."""
    if top is None or top.empty:
        return ""
    p = top["Best pick %"].to_numpy() / 100
    floors = (1 / p) * (1 + DEFAULTS["min_edge"] / 100)
    n = len(p)
    two = (f" A 2-pick ticket (#1 + #2) wins about {100*p[:2].prod():.0f}%."
           if n > 2 else "")
    tested = (f" In testing on 472 past match days, the top-{n} ticket won on "
              f"{TESTED_TICKET[n]}% of days - it still lost about 1 day in "
              f"{100 / (100 - TESTED_TICKET[n]):.0f}." if n in TESTED_TICKET else "")
    return (f"🎟️ All {n} on one ticket: all {n} win about **{100*p.prod():.0f}%** of the time "
            f"(lose about {100*(1-p.prod()):.0f}%). Take it only if the ticket's total odds "
            f"are at least **{floors.prod():.2f}**.{two}{tested}")


def safest_picks(adv: pd.DataFrame, tz: str, n: int = 6) -> tuple:
    """Today's (or the next match day's) most likely picks, safest first."""
    if adv is None or adv.empty or "Best pick %" not in adv.columns:
        return None, pd.DataFrame()
    d = add_local_time(adv, tz)
    d = d[pd.to_datetime(d["Kickoff UTC"], utc=True) > pd.Timestamp.now(tz="UTC")]
    d = d.dropna(subset=["Best pick %"])
    if d.empty:
        return None, d
    day = sorted(d["_day"].unique())[0]
    top = d[d["_day"] == day].sort_values("Best pick %", ascending=False)
    return day, top[top["Best pick %"] >= BEST_PICK_TIERS[2][0] * 100].head(n)


def write_predictions_md(adv: pd.DataFrame, flags: pd.DataFrame, path: Path,
                         bankroll: float = DEFAULTS["bankroll"],
                         allow_all: bool = False, tz: str = DEFAULTS["timezone"],
                         record_lines: list | None = None,
                         settled_log: pd.DataFrame | None = None):
    """Phone-friendly summary of the week, day by day (shown nicely on GitHub)."""
    record_lines = record_lines or pick_record_lines({"n": 0})
    stamp = pd.Timestamp.now(tz=tz).strftime("%a %d %b %Y %H:%M")
    lines = ["# Football predictions — day by day",
             f"_Updated {stamp} ({tz} time). Bankroll for stakes: {bankroll:,.2f}_", ""]
    if adv.empty:
        lines += ["No fixtures in the next 7 days are in the feeds yet. League rounds "
                  "are usually published mid-week.", ""]
    else:
        d = add_local_time(adv, tz)
        # 0) safest picks + rules + live track record
        udays = under_picks_by_day(adv, tz)
        if udays:
            lines += ["## 🟢 Under 4.5 / Under 5.5 — matches with 80%+ chance", "",
                      "_Safest first. 💚 = 1xBet pays at least the fair minimum._", ""]
            for day, g in udays:
                lines += [f"### {pd.Timestamp(day):%A %d %B} ({len(g)} matches)", ""]
                for _, r in g.iterrows():
                    pr = r["1xBet odds"]
                    lines.append(f"{int(r['Rank'])}. {r['Time']} · **{r['Match']}** "
                                 f"({r['Competition']}) — {r['Bet']} · {r['Chance %']:.0f}%"
                                 + (f" · 1xBet {float(pr):.2f} {r['Value']}" if pd.notna(pr) else ""))
                st_ = under_ticket_steps(g)
                tk = [f"top {int(x.Matches)}: {x['Ticket wins %']:.0f}%"
                      + (f" @ {x['Total odds (1xBet)']:.2f}" if pd.notna(x['Total odds (1xBet)']) else "")
                      for _, x in st_.iterrows() if int(x.Matches) in (2, 3, 5, 8, 10)]
                lines += ["", "🎟️ One ticket with the top matches — " + " · ".join(tk), ""]
        days = ready_picks_by_day(adv, tz, 5)
        if days:
            lines += ["## ✅ Your bets for each day (already checked - just place them)", "",
                      "_Only 85%+ picks, backed by bookmaker odds (or 88%+ without), with no key "
                      "player missing for the team you back. Skip a bet if your odds are lower "
                      "than the number shown._", ""]
            for day, top in days:
                lines += [f"### {pd.Timestamp(day):%A %d %B}", ""]
                for _, r in top.iterrows():
                    lines += [f"{int(r['Rank'])}. {r['Time']} · **{r['Match']}** "
                              f"({r['Competition']})  ",
                              f"   ✅ **BET: {r['Bet']}** — only if odds are "
                              f"**{r['Min odds']:.2f}+** ({r['Best pick %']:.0f}% chance)"
                              + (f" · 💚 1xBet pays {float(r['1xBet odds']):.2f}"
                                 if pd.notna(r.get("1xBet odds", np.nan)) else "")]
                lines += ["", ticket_line(top), ""]
        all_days = all_matches_by_day(adv, tz)
        if all_days:
            lines += ["## 📋 Every match: what to bet (whatever the odds)", "",
                      "_✅ = passes all checks (same as 'Your bets'); ⚠️ = riskier._", ""]
            for day, g in all_days:
                lines += [f"### {pd.Timestamp(day):%A %d %B} ({len(g)} matches)", ""]
                for _, r in g.iterrows():
                    price = r.get("1xBet odds", np.nan)
                    lines.append(f"- {'✅' if r['Check'] == '' else '⚠️'} {r['Time']} · "
                                 f"**{r['Match']}** ({r['Competition']}) — BET: {r['Bet']} · "
                                 f"{r['Best pick %']:.0f}%"
                                 + (f" · 1xBet {float(price):.2f}" if pd.notna(price) else ""))
                lines.append("")
        past = past_matches_by_day(settled_log, tz, days_back=3) if settled_log is not None else {}
        if past:
            lines += ["## 🧾 Results of the last days (bets given before kick-off)", ""]
            for day in sorted(past, reverse=True):
                g = past[day]
                lines += [f"### {pd.Timestamp(day):%A %d %B}", "", day_result_line(g), ""]
                for _, r in g.iterrows():
                    lines.append(f"- {r['Checked']} {r['Time']} · **{r['Match']}** — "
                                 f"{r['Bet']} ({r['Best pick %']:.0f}%) → {r['Result']}")
                lines.append("")
        lines += ["## 📈 Live track record of best picks", ""] + record_lines + [""]
        lines += [SAFE_RULES, ""]
        # 1) the week's bets at a glance
        lines += ["## 💰 Value bets (where the model beat the bookmaker in testing)", ""]
        any_bet = False
        for day, grp in d.groupby("_day", sort=True):
            slip = bet_slip_lines(grp)
            if slip:
                any_bet = True
                lines.append(f"**{grp['Day'].iloc[0]}**")
                lines += [f"- {s_}" for s_ in slip]
                lines.append("")
        if not any_bet:
            lines += ["No bets this week - the prices are fair. That is normal.", ""]
        # 2) strongest best picks per day
        if "Best pick %" in d.columns:
            lines += ["## 🎯 Strongest best picks (🔒 Banker + ⭐⭐⭐ Strong, same safety checks "
                      "as 'Your bets'; won ~91% in testing)", ""]
            any_pick = False
            for day, grp in d.groupby("_day", sort=True):
                top = grp[(grp["Best pick %"] >= BEST_PICK_STRONG * 100)
                          & np.array([ready_filter(r) == "" for _, r in grp.iterrows()], dtype=bool)].sort_values(
                    "Best pick %", ascending=False).head(8)
                if top.empty:
                    continue
                any_pick = True
                lines.append(f"**{grp['Day'].iloc[0]}**")
                lines += [f"- {r['Time']} · **{r['Match']}** ({r['Competition']}) — {r['Best pick']}"
                          + (f"  \n  📝 {r['How to bet']}" if isinstance(r.get("How to bet"), str)
                             else "")
                          for _, r in top.iterrows()]
                lines.append("")
            if not any_pick:
                lines += ["No 🔒 / ⭐⭐⭐ picks this week.", ""]
        # 3) every match, every day
        lines += ["## 📅 Every match, day by day", ""]
        for day, grp in d.groupby("_day", sort=True):
            lines += [f"### {grp['Day'].iloc[0]} ({len(grp)} matches)", "",
                      "| Time | Competition | Match | Advice | Stake | 🎯 Best pick | Predicted | H / D / A % "
                      f"| Over {' / '.join(GOAL_LINES)} % "
                      f"| Under {' / '.join(GOAL_LINES)} % |",
                      "|---|---|---|---|---|---|---|---|---|---|"]
            for _, r in grp.iterrows():
                adv_txt = r["Advice"]
                if adv_txt.startswith(("BET", "WATCH")):
                    adv_txt = f"**{adv_txt}**"
                lines.append(f"| {r['Time']} | {r['Competition']} | {r['Match']} | {adv_txt} | "
                             f"{r['Stake'] or '-'} | {r.get('Best pick', '-')} | {r['Predicted score']} | "
                             f"{r['Home %']:.0f} / {r['Draw %']:.0f} / {r['Away %']:.0f} | "
                             + " / ".join(f"{r.get(f'Over {g} %', np.nan):.0f}" for g in GOAL_LINES)
                             + " | "
                             + " / ".join(f"{r.get(f'Under {g} %', np.nan):.0f}" for g in GOAL_LINES)
                             + " |")
            lines.append("")
    lines += [BEFORE_YOU_BET.replace("**Before you place any bet**",
                                     "## ☑️ Before you place any bet"), "",
              "_BET = the bookmaker's price is higher than the model's fair price, in a "
              "league where that made money in testing (currently the Premier League "
              "only). WATCH = paper-trade only. Cups and national teams have no odds "
              "feed: bet only at or above the 'back only at' price. Stakes are 1/4 "
              f"Kelly, capped at {DEFAULTS['max_stake_pct']:.0f}% of bankroll. Exact "
              "scores are hard: the predicted score was right ~1 in 9 matches in "
              "testing, the winner ~1 in 2. The edge is not yet proven - paper-trade "
              "before staking real money._", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# UEFA CLUB COMPETITIONS + NATIONS LEAGUE
# =============================================================================
def _season_start_year(today: dt.date) -> int:
    return today.year if today.month >= 7 else today.year - 1


def _fetch_json(url: str):
    try:
        resp = _http_get(url)
        return None if resp is None else resp.json()
    except Exception:
        return None


def _feed_to_frame(data, code: str, season: int) -> pd.DataFrame:
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    need = {"DateUtc", "HomeTeam", "AwayTeam", "HomeTeamScore", "AwayTeamScore",
            "RoundNumber"}
    if not need.issubset(df.columns):
        return pd.DataFrame()
    df = df.dropna(subset=["HomeTeam", "AwayTeam"])
    df = df[~df["HomeTeam"].astype(str).str.match(r"^(To be|TBD|Winner|Loser|\d)", case=False)]
    out = pd.DataFrame({
        "Comp": code, "Season": season,
        "Kickoff": pd.to_datetime(df["DateUtc"], utc=True, errors="coerce"),
        "HomeTeam": df["HomeTeam"].astype(str).str.strip().replace(UEFA_ALIASES),
        "AwayTeam": df["AwayTeam"].astype(str).str.strip().replace(UEFA_ALIASES),
        "FTHG": pd.to_numeric(df["HomeTeamScore"], errors="coerce"),
        "FTAG": pd.to_numeric(df["AwayTeamScore"], errors="coerce"),
        "Round": pd.to_numeric(df["RoundNumber"], errors="coerce"),
    })
    # The final (last round, a single match) is played at a neutral venue
    last = out["Round"].max()
    out["Neutral"] = (out["Round"] == last) & ((out["Round"] == last).sum() == 1)
    out["Date"] = out["Kickoff"].dt.tz_convert(None).dt.normalize()
    return out.dropna(subset=["Kickoff"])


@st.cache_data(ttl=3600, show_spinner=False)
def load_uefa() -> pd.DataFrame:
    """Champions / Europa / Conference League fixtures and results,
    current season + previous ones (for the model)."""
    return _live_or_backup("uefa", _load_uefa_live, prefer_backup=True)


def _load_uefa_live() -> pd.DataFrame:
    start = _season_start_year(dt.date.today())
    jobs = [(code, slug, start - k) for code, (_, slug) in UEFA_COMPS.items()
            for k in range(UEFA_DEFAULTS["seasons"])]
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=6) as ex:
        datas = list(ex.map(lambda j: _fetch_json(UEFA_FEED.format(slug=f"{j[1]}-{j[2]}")),
                            jobs))
    frames = [_feed_to_frame(d, code, yr) for d, (code, _, yr) in zip(datas, jobs)]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values("Kickoff").reset_index(drop=True)


@st.cache_data(ttl=3600, show_spinner=False)
def load_nations_league() -> pd.DataFrame:
    """Current UEFA Nations League fixtures (names mapped to the results dataset)."""
    start = _season_start_year(dt.date.today())
    code, _, slug = NATIONS_LEAGUE
    for yr in (start, start - 1):
        df = _feed_to_frame(_fetch_json(UEFA_FEED.format(slug=f"{slug}-{yr}")), code, yr)
        if not df.empty:
            df["HomeTeam"] = df["HomeTeam"].replace(NATION_ALIASES)
            df["AwayTeam"] = df["AwayTeam"].replace(NATION_ALIASES)
            df["Neutral"] = False      # league-phase games are home and away
            return df
    return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def load_intl_fixtures() -> pd.DataFrame:
    """Nations League + any published major tournament (AFCON, World Cup,
    Euros, Copa America) - names mapped to the results dataset."""
    return _live_or_backup("intl_fixtures", _load_intl_fixtures_live, prefer_backup=True)


def _load_intl_fixtures_live() -> pd.DataFrame:
    frames = [load_nations_league()]
    year = dt.date.today().year
    for code, _, slug in NATION_TOURNAMENTS:
        for yr in range(year - 1, year + 3):
            df = _feed_to_frame(_fetch_json(UEFA_FEED.format(slug=f"{slug}-{yr}")), code, yr)
            if df.empty or df["FTHG"].notna().all():
                continue                       # not published, or already finished
            df["HomeTeam"] = df["HomeTeam"].replace(NATION_ALIASES)
            df["AwayTeam"] = df["AwayTeam"].replace(NATION_ALIASES)
            df["Neutral"] = True
            frames.append(df)
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values("Kickoff").reset_index(drop=True)


def nation_confederations(intl: pd.DataFrame) -> dict:
    """{national team: region} from the tournaments it has played since 2010."""
    recent = intl[intl["Date"] >= "2010-01-01"]
    counts = {}
    for region, pattern in CONFEDERATIONS:
        m = recent[recent["Tournament"].str.contains(pattern, regex=True)]
        for t in pd.concat([m["HomeTeam"], m["AwayTeam"]]):
            counts.setdefault(t, {}).setdefault(region, 0)
            counts[t][region] += 1
    # the region whose tournaments the team plays most often
    return {t: max(c, key=c.get) for t, c in counts.items()}


def uefa_group(team: str, league_teams: dict) -> str:
    """League code (covered leagues) or country code for a UEFA club."""
    if team in UEFA_CLUB_MAP:
        return UEFA_CLUB_MAP[team][0]
    if team in league_teams:                     # same spelling in a covered league
        return league_teams[team]
    return UEFA_CLUB_COUNTRY.get(team, "OTHER")


def uefa_domestic(teams, models: dict) -> dict:
    """{UEFA club: (group, domestic attack, domestic defence)} from league models."""
    league_teams = {t: lg for lg, m in models.items() for t in m.get("attack", {})}
    out = {}
    for t in teams:
        g = uefa_group(t, league_teams)
        fd = UEFA_CLUB_MAP[t][1] if t in UEFA_CLUB_MAP else t
        m = models.get(g)
        if m is not None and fd in m["attack"]:
            out[t] = (g, float(m["attack"][fd]), float(m["defence"][fd]))
        else:
            out[t] = (g, 0.0, 0.0)
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def fit_uefa(uefa: pd.DataFrame, dom: dict,
             half_life_days: int = UEFA_DEFAULTS["half_life"],
             offset_penalty: float = UEFA_DEFAULTS["offset_penalty"],
             club_penalty: float = UEFA_DEFAULTS["club_penalty"]) -> dict:
    """
    European-cup model. A club's strength = its domestic rating (if its league
    is covered) + how strong its league/country is in Europe + a small
    club-specific adjustment, all fitted on past UEFA matches.
    """
    from scipy.sparse import csr_matrix
    played = uefa.dropna(subset=["FTHG", "FTAG"]).reset_index(drop=True)
    teams_all = sorted(set(uefa["HomeTeam"]) | set(uefa["AwayTeam"]))
    groups = sorted({dom[t][0] for t in teams_all})
    clubs = sorted(set(played["HomeTeam"]) | set(played["AwayTeam"]))
    gi = {g: i for i, g in enumerate(groups)}
    ci = {c: i for i, c in enumerate(clubs)}
    G, C = len(groups), len(clubs)
    P = 2 + 2 * G + 2 * C
    iA = lambda g: 2 + gi[g]                       # noqa: E731
    iD = lambda g: 2 + G + gi[g]                   # noqa: E731
    iRa = lambda c: 2 + 2 * G + ci[c]              # noqa: E731
    iRd = lambda c: 2 + 2 * G + C + ci[c]          # noqa: E731

    last = played["Kickoff"].max()
    rows, cols, vals, off, y, w = [], [], [], [], [], []
    r = 0
    for _, m in played.iterrows():
        h, a = m["HomeTeam"], m["AwayTeam"]
        gh, ga = dom[h][0], dom[a][0]
        wt = float(np.exp(-np.log(2) / half_life_days
                          * (last - m["Kickoff"]).total_seconds() / 86400))
        hf = 0.0 if m["Neutral"] else 1.0
        for (j, c) in [(0, 1.0), (1, hf), (iA(gh), 1.0), (iRa(h), 1.0),
                       (iD(ga), -1.0), (iRd(a), -1.0)]:
            rows.append(r); cols.append(j); vals.append(c)
        off.append(dom[h][1] - dom[a][2]); y.append(m["FTHG"]); w.append(wt); r += 1
        for (j, c) in [(0, 1.0), (iA(ga), 1.0), (iRa(a), 1.0),
                       (iD(gh), -1.0), (iRd(h), -1.0)]:
            rows.append(r); cols.append(j); vals.append(c)
        off.append(dom[a][1] - dom[h][2]); y.append(m["FTAG"]); w.append(wt); r += 1
    X = csr_matrix((vals, (rows, cols)), shape=(r, P))
    off, y, w = np.array(off), np.array(y, float), np.array(w)
    pen = np.zeros(P)
    pen[2:2 + 2 * G] = offset_penalty
    pen[2 + 2 * G:] = club_penalty

    def nll_and_grad(p):
        eta = X @ p + off
        lam = np.exp(eta)
        nll = np.sum(w * (lam - y * eta)) + np.sum(pen * p * p)
        g = X.T @ (w * (lam - y)) + 2 * pen * p
        return nll, g

    x0 = np.zeros(P)
    x0[0], x0[1] = 0.2, 0.2
    res = minimize(nll_and_grad, x0, jac=True, method="L-BFGS-B",
                   options={"maxiter": 5000, "ftol": 1e-15, "gtol": 1e-9})
    p = res.x

    def strength(t):
        g, da, dd = dom.get(t, ("OTHER", 0.0, 0.0))
        A = p[iA(g)] if g in gi else 0.0
        D = p[iD(g)] if g in gi else 0.0
        ra = p[iRa(t)] if t in ci else 0.0
        rd = p[iRd(t)] if t in ci else 0.0
        return da + A + ra, dd + D + rd

    att = {t: strength(t)[0] for t in teams_all}
    dfn = {t: strength(t)[1] for t in teams_all}
    hf = (~played["Neutral"]).to_numpy(float)
    lam_h = np.exp(p[0] + p[1] * hf + np.array([att[t] for t in played["HomeTeam"]])
                   - np.array([dfn[t] for t in played["AwayTeam"]]))
    lam_a = np.exp(p[0] + np.array([att[t] for t in played["AwayTeam"]])
                   - np.array([dfn[t] for t in played["HomeTeam"]]))
    rho = _dixon_coles_rho(played["FTHG"].to_numpy(float), played["FTAG"].to_numpy(float),
                           lam_h, lam_a, w[::2])      # one weight per match
    group_strength = {g: (float(p[iA(g)]), float(p[iD(g)])) for g in groups}
    return {"mu": float(p[0]), "home": float(p[1]), "rho": rho,
            "attack": {k: float(v) for k, v in att.items()},
            "defence": {k: float(v) for k, v in dfn.items()},
            "prior": {"attack": 0.0, "defence": 0.0},
            "teams": teams_all, "current_teams": teams_all,
            "groups": group_strength, "dom": dom, "pick_shrink": UEFA_PICK_SHRINK,
            "n_matches": int(len(played)),
            "last_date": played["Kickoff"].max(), "converged": bool(res.success),
            "goals_weight": 1.0}


# -----------------------------------------------------------------------------
# Bookmaker odds for cups and national teams (The Odds API, free key stored as
# the GitHub secret ODDS_API_KEY). Only the scheduled GitHub run calls the API;
# it saves the odds in data_snapshot/cup_odds.csv and the app reads that file.
# Free plan: 500 credits/month; one call = 2 credits (1X2 + totals, EU region).
# -----------------------------------------------------------------------------
ODDS_API = "https://api.the-odds-api.com/v4"
ODDS_SPORTS = {"soccer_uefa_champs_league": "UCL", "soccer_uefa_europa_league": "UEL",
               "soccer_uefa_europa_conference_league": "UECL",
               "soccer_uefa_nations_league": "UNL", "soccer_africa_cup_of_nations": "AFCON",
               "soccer_fifa_world_cup": "WC", "soccer_uefa_european_championship": "EURO",
               "soccer_conmebol_copa_america": "COPA"}
ODDS_STATUS = "odds_status"


def _odds_api_get(path: str, key: str, **params):
    """One GET (no retries: failed calls must not burn credits). (json, headers)."""
    if OFFLINE or not key:
        return None, {}
    try:
        r = requests.get(f"{ODDS_API}{path}", params={"apiKey": key, **params},
                         headers=HTTP_HEADERS, timeout=20)
        if r.status_code != 200:
            return {"_error": f"HTTP {r.status_code}"}, dict(r.headers)
        return r.json(), dict(r.headers)
    except Exception as exc:
        return {"_error": type(exc).__name__}, {}


def _median_price(event: dict, market: str, name: str, point=None):
    prices = []
    for bk in event.get("bookmakers", []):
        for mk in bk.get("markets", []):
            if mk.get("key") != market:
                continue
            for o in mk.get("outcomes", []):
                if o.get("name") == name and (point is None or o.get("point") == point):
                    try:
                        prices.append(float(o["price"]))
                    except (KeyError, TypeError, ValueError):
                        pass
    return float(np.median(prices)) if prices else np.nan


def odds_events_to_frame(events: list, comp: str) -> pd.DataFrame:
    """The Odds API events -> one row per match with median 1X2 and O/U 2.5."""
    rows = []
    for ev in events or []:
        h, a = ev.get("home_team"), ev.get("away_team")
        if not h or not a:
            continue
        rows.append({"Comp": comp, "Kickoff": ev.get("commence_time"), "HomeTeam": h,
                     "AwayTeam": a, "H": _median_price(ev, "h2h", h),
                     "D": _median_price(ev, "h2h", "Draw"), "A": _median_price(ev, "h2h", a),
                     "O25": _median_price(ev, "totals", "Over", 2.5),
                     "U25": _median_price(ev, "totals", "Under", 2.5),
                     "Books": len(ev.get("bookmakers", []))})
    return pd.DataFrame(rows)


def refresh_cup_odds(key: str, now=None) -> dict:
    """Fetch cup / national-team odds when useful and save them. Spends credits
    only for competitions with a match in the next 3 days, at most every 4 h
    (every 20 h when nothing kicks off within 10 h). Returns a status dict."""
    now = now or pd.Timestamp.now(tz="UTC")
    status = {"time": now.strftime("%Y-%m-%d %H:%M UTC"), "key_present": bool(key)}
    if not key:
        status["result"] = "no ODDS_API_KEY secret"
        return status
    age = snapshot_age_hours("cup_odds")
    sports, hdr = _odds_api_get("/sports", key)
    if not isinstance(sports, list):
        status["result"] = f"sports list failed: {(sports or {}).get('_error', 'no reply')}"
        return status
    active = [sp["key"] for sp in sports if sp.get("key") in ODDS_SPORTS and sp.get("active")]
    soon, very_soon = [], False
    for sk in active:                                   # /events is free
        evs, _ = _odds_api_get(f"/sports/{sk}/events", key)
        if not isinstance(evs, list):
            continue
        ko = pd.to_datetime([e.get("commence_time") for e in evs], utc=True, errors="coerce")
        if ((ko > now) & (ko <= now + pd.Timedelta(days=3))).any():
            soon.append(sk)
        very_soon |= bool(((ko > now) & (ko <= now + pd.Timedelta(hours=10))).any())
    status.update(active=active, with_matches_soon=soon)
    if not soon:
        status["result"] = "no cup / national-team matches in the next 3 days"
        return status
    if age is not None and (age < 4 or (age < 20 and not very_soon)):
        status["result"] = f"odds still fresh ({age:.1f} h old) - no credits spent"
        return status
    frames, remaining = [], None
    for sk in soon:
        evs, hdr = _odds_api_get(f"/sports/{sk}/odds", key, regions="eu",
                                 markets="h2h,totals", oddsFormat="decimal")
        remaining = hdr.get("x-requests-remaining", remaining)
        if isinstance(evs, list):
            frames.append(odds_events_to_frame(evs, ODDS_SPORTS[sk]))
        else:
            status.setdefault("errors", []).append(f"{sk}: {(evs or {}).get('_error')}")
    df = pd.concat([f for f in frames if not f.empty], ignore_index=True) if any(
        not f.empty for f in frames) else pd.DataFrame()
    status["credits_remaining"] = remaining
    status["matches_with_odds"] = int(len(df))
    if not df.empty and save_snapshot("cup_odds", df):
        status["result"] = f"saved odds for {len(df)} matches"
    else:
        status.setdefault("result", "no odds returned")
    return status


def save_odds_status(status: dict):
    try:
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        (SNAPSHOT_DIR / f"{ODDS_STATUS}.json").write_text(json.dumps(status, indent=1, default=str))
    except Exception:
        pass


def _team_sim(ours: str, theirs: str) -> float:
    """0..1 similarity of two spellings of a club / national team."""
    import difflib
    if FEED_ALIASES.get(theirs) == ours or theirs == ours:
        return 1.0
    a, b = _name_key(ours, core=True), _name_key(theirs, core=True)
    if not a or not b:
        return 0.0
    if a in b or b in a:
        return 1.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def match_fixture_row(fx, df: pd.DataFrame, hours: int = 3):
    """The row of `df` (Kickoff, HomeTeam, AwayTeam) that is the same match as
    fixture `fx`: kick-off within +/- `hours` and both names similar enough.
    Doubtful cases return None - a wrong match is worse than none."""
    if df is None or df.empty:
        return None
    ko = pd.Timestamp(fx["Kickoff"])
    ko = ko.tz_localize("UTC") if ko.tzinfo is None else ko.tz_convert("UTC")
    c = df if "_ko" in df.columns else df.assign(
        _ko=pd.to_datetime(df["Kickoff"], utc=True, errors="coerce"))
    c = c[(c["_ko"] - ko).abs() <= pd.Timedelta(hours=hours)]
    best, best_score = None, 0.0
    for _, r in c.iterrows():
        sh, sa = _team_sim(fx["HomeTeam"], r["HomeTeam"]), _team_sim(fx["AwayTeam"], r["AwayTeam"])
        score = (sh + sa) / 2
        ok = (min(sh, sa) >= 0.5 and score >= 0.7) or (max(sh, sa) >= 0.95 and min(sh, sa) >= 0.25)
        if ok and score > best_score:
            best, best_score = r, score
    return best


def match_fixture_odds(fx, cup_odds: pd.DataFrame) -> dict:
    """Bookmaker odds for one cup fixture (see match_fixture_row)."""
    best = match_fixture_row(fx, cup_odds)
    if best is None:
        return {}
    return {k: float(best[k]) for k in ("H", "D", "A", "O25", "U25")
            if pd.notna(best.get(k)) and float(best[k]) > 1}


# -----------------------------------------------------------------------------
# Missing players (API-Football Pro, GitHub secret APIFOOTBALL_KEY). The
# scheduled run saves the next 3 days' injuries / suspensions and the missing
# players' season stats; predictions lower a team's attack for missing scorers
# and raise its leakiness for missing regular goalkeepers / defenders.
# These sizes are deliberately cautious and not yet backtested.
# -----------------------------------------------------------------------------
AF_LEAGUES = {39: "E0", 40: "E1", 41: "E2", 42: "E3", 43: "EC", 140: "SP1", 141: "SP2",
              135: "I1", 136: "I2", 78: "D1", 79: "D2", 61: "F1", 62: "F2", 88: "N1",
              94: "P1", 203: "T1", 179: "SC0", 180: "SC1", 183: "SC2", 184: "SC3",
              144: "B1", 197: "G1", 235: "RUS", 98: "JPN", 169: "CHN",
              2: "UCL", 3: "UEL", 848: "UECL", 5: "UNL"}
INJ_WEIGHT = {"Missing Fixture": 1.0, "Questionable": 0.5}
INJ_ATT_KEEP = 0.6          # share of a missing player's goal threat that is lost
INJ_ATT_CAP, INJ_DEF_CAP = 0.30, 0.15
INJ_DEF_BUMP = {"Goalkeeper": 0.06, "Defender": 0.03, "Midfielder": 0.015}


def refresh_injuries(key: str, now=None, days: int = 3, max_players: int = 400) -> dict:
    """Fetch upcoming fixtures, injuries/suspensions and the missing players'
    season stats from API-Football; save them in the backup folder."""
    now = now or pd.Timestamp.now(tz="UTC")
    status = {"time": now.strftime("%Y-%m-%d %H:%M UTC"), "key_present": bool(key)}
    if not key:
        status["result"] = "no APIFOOTBALL_KEY secret"
        return status
    dates = [(now + pd.Timedelta(days=d)).strftime("%Y-%m-%d") for d in range(days + 1)]
    fx_rows, leagues_seen, used = [], {}, 0
    for d in dates:
        js = _apifootball_get("/fixtures", key, date=d, timezone="UTC") or {}
        used += 1
        if js.get("errors"):
            status.setdefault("errors", []).append(f"fixtures {d}: {js['errors']}")
            continue
        for f in js.get("response") or []:
            lg = (f.get("league") or {}).get("id")
            if lg not in AF_LEAGUES:
                continue
            if ((f.get("fixture") or {}).get("status") or {}).get("short") not in ("NS", "TBD", "PST"):
                continue
            leagues_seen[AF_LEAGUES[lg]] = f"{f['league'].get('name')} ({f['league'].get('country')})"
            fx_rows.append({"FixtureId": f["fixture"]["id"], "Kickoff": f["fixture"]["date"],
                            "League": AF_LEAGUES[lg], "Season": f["league"].get("season"),
                            "HomeId": f["teams"]["home"]["id"], "HomeTeam": f["teams"]["home"]["name"],
                            "AwayId": f["teams"]["away"]["id"], "AwayTeam": f["teams"]["away"]["name"]})
    fixtures = pd.DataFrame(fx_rows).drop_duplicates("FixtureId") if fx_rows else pd.DataFrame()
    status["leagues_found"] = leagues_seen
    status["fixtures"] = int(len(fixtures))
    if fixtures.empty:
        status["result"] = "no upcoming fixtures found"
        status["requests_used"] = used
        return status
    ids = set(fixtures["FixtureId"])
    inj_rows = []
    for d in dates:
        js = _apifootball_get("/injuries", key, date=d) or {}
        used += 1
        if js.get("errors"):
            status.setdefault("errors", []).append(f"injuries {d}: {js['errors']}")
            continue
        for r in js.get("response") or []:
            fid = (r.get("fixture") or {}).get("id")
            if fid in ids:
                pl = r.get("player") or {}
                inj_rows.append({"FixtureId": fid, "TeamId": (r.get("team") or {}).get("id"),
                                 "PlayerId": pl.get("id"), "Player": pl.get("name"),
                                 "Type": pl.get("type"), "Reason": pl.get("reason")})
    injuries = (pd.DataFrame(inj_rows).drop_duplicates(["FixtureId", "PlayerId"])
                if inj_rows else pd.DataFrame(columns=["FixtureId", "TeamId", "PlayerId",
                                                       "Player", "Type", "Reason"]))
    # season stats of the missing players (cached for 6 days)
    cache = load_snapshot("af_players")
    need = (injuries.merge(fixtures[["FixtureId", "Season"]], on="FixtureId")
            [["PlayerId", "Season"]].drop_duplicates()) if not injuries.empty else pd.DataFrame()
    fresh = set()
    if not cache.empty:
        upd = pd.to_datetime(cache["Updated"], utc=True, errors="coerce")
        ok = cache[upd > now - pd.Timedelta(days=6)]
        fresh = set(zip(ok["PlayerId"].astype(int), ok["Season"].astype(int)))
    new_rows = []
    for _, r in need.iterrows():
        if len(new_rows) >= max_players:
            status["players_skipped"] = True
            break
        k = (int(r["PlayerId"]), int(r["Season"]))
        if k in fresh:
            continue
        js = _apifootball_get("/players", key, id=k[0], season=k[1]) or {}
        used += 1
        app_, lin, mins, g, a, pos = 0, 0, 0, 0, 0, None
        for st_ in ((js.get("response") or [{}])[0].get("statistics") or []):
            gm = st_.get("games") or {}
            app_ += gm.get("appearences") or 0
            lin += gm.get("lineups") or 0
            mins += gm.get("minutes") or 0
            g += (st_.get("goals") or {}).get("total") or 0
            a += (st_.get("goals") or {}).get("assists") or 0
            pos = pos or gm.get("position")
        new_rows.append({"PlayerId": k[0], "Season": k[1], "Position": pos, "Apps": app_,
                         "Starts": lin, "Minutes": mins, "Goals": g, "Assists": a,
                         "Updated": now.isoformat()})
    players = pd.concat([cache, pd.DataFrame(new_rows)], ignore_index=True) if new_rows else cache
    if not players.empty:
        players = players.drop_duplicates(["PlayerId", "Season"], keep="last")
    status.update(injured=int(len(injuries)), players_fetched=len(new_rows), requests_used=used)
    save_snapshot("af_fixtures", fixtures)
    if not injuries.empty:
        save_snapshot("af_injuries", injuries)
    if not players.empty:
        save_snapshot("af_players", players)
    status["result"] = f"{len(injuries)} missing players across {len(fixtures)} fixtures"
    return status


# Bookmaker prices for the exact pick markets (API-Football /odds, Pro plan).
AF_BET_MAP = {
    "Match Winner": {"Home": "H", "Draw": "D", "Away": "A"},
    "Double Chance": {"Home/Draw": "1X", "Draw/Away": "X2", "Home/Away": "12"},
    "Goals Over/Under": {"Over 1.5": "O15", "Under 1.5": "U15", "Over 2.5": "O25",
                         "Under 2.5": "U25", "Over 3.5": "O35", "Under 3.5": "U35",
                         "Under 4.5": "U45", "Under 5.5": "U55", "Over 0.5": "O05"},
    "Total - Home": {"Over 0.5": "HS"},
    "Total - Away": {"Over 0.5": "AS"},
    "Both Teams Score": {"Yes": "BTTS_Y", "No": "BTTS_N"},
}
MY_BOOKS = ("1xBet",)        # the user's bookmakers that API-Football covers


def af_odds_rows(item: dict) -> list:
    """API-Football /odds response item -> [{FixtureId, Book, Key, Price}]."""
    rows = []
    fid = (item.get("fixture") or {}).get("id")
    for bk in item.get("bookmakers") or []:
        for bet in bk.get("bets") or []:
            mp = AF_BET_MAP.get(bet.get("name"))
            if not mp:
                continue
            for v in bet.get("values") or []:
                k = mp.get(str(v.get("value")))
                try:
                    price = float(v.get("odd"))
                except (TypeError, ValueError):
                    continue
                if k and price > 1:
                    rows.append({"FixtureId": fid, "Book": bk.get("name"), "Key": k,
                                 "Price": price})
    return rows


def refresh_book_odds(key: str, now=None, days: int = 3, max_calls: int = 400) -> dict:
    """Bookmaker prices (1xBet and others) for every upcoming fixture."""
    now = now or pd.Timestamp.now(tz="UTC")
    status = {"odds_time": now.strftime("%Y-%m-%d %H:%M UTC")}
    fx = load_snapshot("af_fixtures")
    if not key or fx.empty:
        status["odds_result"] = "no key or no fixtures"
        return status
    ko = pd.to_datetime(fx["Kickoff"], utc=True, errors="coerce")
    fx = fx[(ko > now) & (ko <= now + pd.Timedelta(days=days))]
    rows, books, bets, calls = [], set(), set(), 0
    for fid in fx["FixtureId"].head(max_calls):
        js = _apifootball_get("/odds", key, fixture=int(fid)) or {}
        calls += 1
        for item in js.get("response") or []:
            for bk in item.get("bookmakers") or []:
                books.add(bk.get("name"))
                if bk.get("name") in MY_BOOKS:
                    bets.update(b.get("name") for b in bk.get("bets") or [])
            rows += af_odds_rows(item)
    df = pd.DataFrame(rows)
    status.update(odds_calls=calls, odds_rows=len(df),
                  odds_fixtures=int(df["FixtureId"].nunique()) if not df.empty else 0,
                  books_seen=sorted(b for b in books if b)[:40],
                  my_book_bets_seen=sorted(b for b in bets if b)[:60],
                  my_book_fixtures=int(df[df["Book"].isin(MY_BOOKS)]["FixtureId"].nunique())
                  if not df.empty else 0)
    if not df.empty:
        save_snapshot("af_odds", df)
        status["odds_result"] = f"prices for {status['odds_fixtures']} fixtures"
    else:
        status["odds_result"] = "no prices returned"
    return status


def load_book_odds() -> pd.DataFrame:
    """Long table FixtureId / Book / Key / Price with each fixture's kick-off and
    API-Football team names (for matching to our fixtures)."""
    od, fx = load_snapshot("af_odds"), load_snapshot("af_fixtures")
    if od.empty or fx.empty:
        return pd.DataFrame()
    t = od.merge(fx[["FixtureId", "Kickoff", "HomeTeam", "AwayTeam"]], on="FixtureId")
    t["_ko"] = pd.to_datetime(t["Kickoff"], utc=True, errors="coerce")
    return t


def book_prices(fx, book_odds: pd.DataFrame, key: str) -> tuple:
    """(1xBet price, best price of any bookmaker) for one pick; NaN if unknown."""
    if book_odds is None or book_odds.empty or not isinstance(key, str):
        return np.nan, np.nan
    row = match_fixture_row(fx, book_odds.drop_duplicates("FixtureId"))
    if row is None:
        return np.nan, np.nan
    t = book_odds[(book_odds["FixtureId"] == row["FixtureId"]) & (book_odds["Key"] == key)]
    if t.empty:
        return np.nan, np.nan
    mine = t[t["Book"].isin(MY_BOOKS)]["Price"]
    return (float(mine.max()) if not mine.empty else np.nan), float(t["Price"].max())


def load_injury_table() -> pd.DataFrame:
    """One row per missing player of an upcoming fixture, with season stats and
    which side (H/A) he plays for. Empty when no data is available."""
    fx, inj, pl = load_snapshot("af_fixtures"), load_snapshot("af_injuries"), load_snapshot("af_players")
    if fx.empty or inj.empty:
        return pd.DataFrame()
    t = inj.merge(fx, on="FixtureId", how="inner")
    t["Side"] = np.where(t["TeamId"] == t["HomeId"], "H",
                         np.where(t["TeamId"] == t["AwayId"], "A", ""))
    t = t[t["Side"] != ""]
    if not pl.empty:
        t = t.merge(pl, on=["PlayerId", "Season"], how="left")
    for c in ("Apps", "Starts", "Minutes", "Goals", "Assists"):
        t[c] = pd.to_numeric(t[c], errors="coerce").fillna(0) if c in t.columns else 0
    if "Position" not in t.columns:
        t["Position"] = None
    t["_ko"] = pd.to_datetime(t["Kickoff"], utc=True, errors="coerce")
    return t


def injury_effect(players: pd.DataFrame, team_xg: float) -> tuple:
    """(attack change, defence-leakiness change, text, key players) for the
    missing players of ONE team."""
    if players is None or players.empty:
        return 0.0, 0.0, "", 0
    att = dfn = 0.0
    parts, key = [], 0
    for _, r in players.iterrows():
        w = INJ_WEIGHT.get(str(r.get("Type")), 0.5)
        apps, mins = float(r.get("Apps") or 0), float(r.get("Minutes") or 0)
        g, a, starts = float(r.get("Goals") or 0), float(r.get("Assists") or 0), float(r.get("Starts") or 0)
        per_match = (g + 0.5 * a) / apps if apps > 0 and mins >= 180 else 0.0
        att += w * INJ_ATT_KEEP * per_match / max(team_xg, 0.5)
        regular = starts >= 3
        if regular:
            dfn += w * INJ_DEF_BUMP.get(str(r.get("Position")), 0.0)
        important = per_match >= 0.25 or (regular and r.get("Position") in ("Goalkeeper", "Defender"))
        key += int(important and w >= 1)
        tag = []
        if g or a:
            tag.append(f"{int(g)}G {int(a)}A")
        if r.get("Position") == "Goalkeeper" and regular:
            tag.append("first-choice GK")
        if w < 1:
            tag.append("doubtful")
        name = str(r.get("Player"))
        parts.append(("⚠️" if important else "") + name + (f" ({', '.join(tag)})" if tag else ""))
    parts.sort(key=lambda x: not x.startswith("⚠️"))
    text = ", ".join(parts[:5]) + (f" +{len(parts) - 5} more" if len(parts) > 5 else "")
    return -min(att, INJ_ATT_CAP), min(dfn, INJ_DEF_CAP), text, key


def fixture_injuries(fx, inj_table: pd.DataFrame, model, neutral: bool = False):
    """(adj dict for predict(), short text) for one of our fixtures."""
    adj, text, _, _ = fixture_injuries_full(fx, inj_table, model, neutral)
    return adj, text


def fixture_injuries_full(fx, inj_table: pd.DataFrame, model, neutral: bool = False):
    """(adj, text, key players missing at home, key players missing away)."""
    if inj_table is None or inj_table.empty:
        return None, "", 0, 0
    fxs = inj_table.drop_duplicates("FixtureId")
    row = match_fixture_row(fx, fxs)
    if row is None:
        return None, "", 0, 0
    lh, la = expected_goals(model, fx["HomeTeam"], fx["AwayTeam"], neutral=neutral)
    t = inj_table[inj_table["FixtureId"] == row["FixtureId"]]
    ha, hd, ht, hk = injury_effect(t[t["Side"] == "H"], lh)
    aa, ad, at, ak = injury_effect(t[t["Side"] == "A"], la)
    if not ht and not at:
        return None, "", 0, 0
    text = " | ".join(x for x in (f"{fx['HomeTeam']}: {ht}" if ht else "",
                                  f"{fx['AwayTeam']}: {at}" if at else "") if x)
    adj = {"h_att": ha, "h_def": hd, "a_att": aa, "a_def": ad}
    return (adj if any(abs(v) > 1e-9 for v in adj.values()) else None), text, hk, ak


# -----------------------------------------------------------------------------
# API-Football (free key stored as the GitHub secret APIFOOTBALL_KEY): line-ups
# and injuries. First step: a cheap probe (3 requests of the 100/day) that
# records whether the free plan covers the current season.
# -----------------------------------------------------------------------------
APIFOOTBALL = "https://v3.football.api-sports.io"
APIFOOTBALL_STATUS = "apifootball_status"


def _apifootball_get(path: str, key: str, **params):
    if OFFLINE or not key:
        return None
    try:
        r = requests.get(f"{APIFOOTBALL}{path}", params=params,
                         headers={"x-apisports-key": key, **HTTP_HEADERS}, timeout=20)
        return r.json() if r.status_code == 200 else {"errors": {"http": r.status_code}}
    except Exception as exc:
        return {"errors": {"request": type(exc).__name__}}


def probe_apifootball(key: str) -> dict:
    """Is the key valid, and does the free plan cover the current season?"""
    out = {"time": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M UTC"),
           "key_present": bool(key)}
    if not key:
        out["result"] = "no APIFOOTBALL_KEY secret"
        return out
    st_ = _apifootball_get("/status", key) or {}
    resp = st_.get("response") or {}
    out["plan"] = (resp.get("subscription") or {}).get("plan") if isinstance(resp, dict) else None
    out["requests_today"] = (resp.get("requests") or {}) if isinstance(resp, dict) else None
    out["status_errors"] = st_.get("errors") or None
    season = _season_start_year(dt.date.today())
    fx = _apifootball_get("/fixtures", key, league=39, season=season, next=3) or {}
    out["fixtures_errors"] = fx.get("errors") or None
    out["fixtures_found"] = len(fx.get("response") or [])
    inj = _apifootball_get("/injuries", key, league=39, season=season) or {}
    out["injuries_errors"] = inj.get("errors") or None
    out["injuries_found"] = len(inj.get("response") or [])
    out["season"] = season
    ok = out["fixtures_found"] > 0 and not out["fixtures_errors"]
    out["result"] = ("current season available" if ok else
                     "current season NOT available on this plan" if out["fixtures_errors"]
                     else "no data returned")
    return out


def load_cup_odds() -> pd.DataFrame:
    df = load_snapshot("cup_odds")
    if not df.empty:
        df["_ko"] = pd.to_datetime(df["Kickoff"], utc=True, errors="coerce")
    return df


def cup_predictions(fixtures: pd.DataFrame, model: dict, code_col: str = "Comp",
                    days: int = 7, min_edge: float = DEFAULTS["min_edge"],
                    cup_odds: pd.DataFrame | None = None,
                    injuries: pd.DataFrame | None = None,
                    book_odds: pd.DataFrame | None = None) -> pd.DataFrame:
    """Prediction rows (same columns as the league advice table) for cup or
    international fixtures that have no odds feed."""
    if fixtures is None or fixtures.empty or model is None:
        return pd.DataFrame()
    up = upcoming_fixtures(fixtures, days)
    rows = []
    for _, fx in up.iterrows():
        neutral = bool(fx.get("Neutral", False))
        adj, missing, key_h, key_a = fixture_injuries_full(fx, injuries, model, neutral)
        lh, la, m, p = predict(model, fx["HomeTeam"], fx["AwayTeam"], adj, neutral=neutral)
        pred, _ = predicted_score(m, p)
        best = max(("H", "D", "A"), key=lambda k: p[k])
        lab = {"H": fx["HomeTeam"], "D": "Draw", "A": fx["AwayTeam"]}[best]
        odds = match_fixture_odds(fx, cup_odds)
        has_odds = all(k in odds for k in ("H", "D", "A"))
        k_shr = model.get("pick_shrink", 1.0)
        if all(k in odds for k in ("H", "D", "A")):
            # bookmaker odds known: blend model + market; the market needs no
            # over-confidence correction, so only half of it is kept
            pb, _ = pick_probs(m, model["rho"], odds)
            q_ = shrink_probs(pb, 1 - (1 - k_shr) / 2)
        else:
            q_ = shrink_probs(p, k_shr)
        bp = best_pick(q_, fx["HomeTeam"], fx["AwayTeam"])
        rows.append({
            "League": fx[code_col],
            "Kickoff (UK)": fx["Kickoff"].tz_convert(UK_TZ).strftime("%a %d %b %H:%M"),
            "Kickoff UTC": fx["Kickoff"],
            "Match": f"{fx['HomeTeam']} v {fx['AwayTeam']}",
            "Best pick": bp["text"], "Best pick %": bp["prob"] * 100, "How to bet": bp["how"],
            "Pick key": bp["key"], "Missing players": missing,
            "Key out H": key_h, "Key out A": key_a, "Odds used": has_odds,
            **dict(zip(("1xBet odds", "Best odds"), book_prices(fx, book_odds, bp["key"]))),
            "U45 %": q_["U45"] * 100, "U55 %": q_["U55"] * 100,
            "1xBet U4.5": book_prices(fx, book_odds, "U45")[0],
            "1xBet U5.5": book_prices(fx, book_odds, "U55")[0],
            "Predicted score": pred,
            "Top 5 scores": ", ".join(f"{sc} ({pr*100:.0f}%)" for sc, pr in top_scores(m)),
            "Home %": p["H"] * 100, "Draw %": p["D"] * 100, "Away %": p["A"] * 100,
            **goal_line_cols(p), "BTTS %": p["BTTS_Y"] * 100,
            "xG": f"{lh:.2f} - {la:.2f}",
            "Advice": ((f"Bookmakers: {fx['HomeTeam']} {odds['H']:.2f} · draw {odds['D']:.2f} · "
                        f"{fx['AwayTeam']} {odds['A']:.2f}. Model: back {lab} only at "
                        f"{(1 / p[best]) * (1 + min_edge / 100):.2f}+")
                       if all(k in odds for k in ("H", "D", "A")) else
                       (f"No odds feed: back {lab} only at "
                        f"{(1 / p[best]) * (1 + min_edge / 100):.2f}+")),
            "Stake": "",
        })
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out.insert(1, "Status", STATUS_LABEL["cup"])
    out.insert(0, "Competition", out["League"].map(league_name))
    return out



def league_models(all_results: dict, half_life: int, goals_weight: float) -> dict:
    """Fitted model for every loaded league (each fit is cached)."""
    return {lg: fit_model(df, half_life, goals_weight)
            for lg, df in all_results.items() if not df.empty}


def combine_overview(*frames) -> pd.DataFrame:
    """Join league advice and cup predictions; best opportunities first."""
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["_rank"] = out["League"].map(lambda c: STATUS_RANK[league_status(c)])
    out.loc[out["Advice"].str.startswith("BET"), "_rank"] = 0
    out = out.sort_values(["_rank", "Competition", "Kickoff UTC"]).drop(columns=["_rank"])
    return out.reset_index(drop=True)


def add_local_time(adv: pd.DataFrame, tz: str) -> pd.DataFrame:
    """Add the local day and time of each kick-off, sorted by kick-off."""
    out = adv.copy()
    local = pd.to_datetime(out["Kickoff UTC"], utc=True).dt.tz_convert(tz)
    out["_day"] = local.dt.date
    out["Day"] = local.dt.strftime("%A %d %B")
    out["Time"] = local.dt.strftime("%H:%M")
    return out.sort_values(["_day", "Kickoff UTC", "Competition"]).reset_index(drop=True)


def bet_slip_lines(day_rows: pd.DataFrame) -> list[str]:
    """One line per BET for a day's bet slip."""
    bets = day_rows[day_rows["Advice"].str.startswith("BET")]
    return [f"{r['Time']} · {r['Match']} ({r['Competition']}) — "
            f"**{r['Advice'][4:]}** — stake {r['Stake']}" for _, r in bets.iterrows()]


BEFORE_YOU_BET = (
    "**Before you place any bet**\n"
    "1. Your bookmaker's odds must be **at least** the price shown (for 🏆 games: at "
    "least the 'back only at' price). Lower odds = skip the bet.\n"
    "2. Team news: the system already removes picks where a key player is missing "
    "(injury or suspension data, updated 4 times a day). Late news (rotation, "
    "line-ups an hour before kick-off) is not included - skip a bet if big news breaks.\n"
    "3. Stake no more than shown, and never chase losses.\n"
    "4. 🎯 Best pick = the option most likely to win. In testing, best picks won "
    "~88% in leagues (8,591 matches: 🔒 Banker ~92%, ⭐⭐⭐ Strong ~90%, ⭐⭐ Good "
    "~88%, ⭐ Fair ~84%), ~85% in European cups (972 games) and ~88% for national "
    "teams (1,315 games). Combining picks on one slip multiplies the risk: 4 picks "
    "at 85% all win only ~52% of the time. The "
    "catch: they pay very short odds and bookmakers almost never offer the 'bet at' "
    "price, so over time they lose a little (about -4%). One loss wipes out about "
    "ten wins - stake small, and only bet at or above the odds shown.\n"
    "5. Bets are logged automatically - check the Paper Trading tab to see if the "
    "edge is real before risking real money.")


MOBILE_CSS = """
<style>
.bfe-card {border-left: 5px solid rgba(128,128,128,.45); background: rgba(128,128,128,.08);
  border-radius: 10px; padding: .6rem .75rem; margin: 0 0 .6rem 0; line-height: 1.35;}
.bfe-card.bet {border-left-color: #1e9e45; background: rgba(30,158,69,.12);}
.bfe-card.watch {border-left-color: #c9a227; background: rgba(201,162,39,.12);}
.bfe-top {font-size: .78rem; opacity: .75;}
.bfe-rank {font-size: .95rem; opacity: 1;}
.bfe-match {font-size: 1.02rem; font-weight: 700; margin: .1rem 0 .25rem;}
.bfe-adv {font-size: .9rem; font-weight: 600; margin-bottom: .3rem;}
.bfe-card.bet .bfe-adv {color: #1e9e45;} .bfe-card.watch .bfe-adv {color: #b08d17;}
.bfe-row {font-size: .86rem; margin: .1rem 0;}
.bfe-small {font-size: .78rem; opacity: .85;}
.bfe-ou {width: 100%; border-collapse: collapse; font-size: .74rem; margin-top: .3rem;
  text-align: center; table-layout: fixed;}
.bfe-ou td, .bfe-ou th {border: 1px solid rgba(128,128,128,.25); padding: 2px 0;}
.bfe-ou th {font-weight: 600; opacity: .8;}
@media (max-width: 640px) {
  [data-testid="stMainBlockContainer"], .block-container {
    padding: 2.6rem .65rem 3rem !important;}
  h1 {font-size: 1.5rem !important; line-height: 1.25 !important; padding-top: 0 !important;}
  h2 {font-size: 1.25rem !important;} h3 {font-size: 1.1rem !important;}
  h4 {font-size: 1rem !important;}
  [data-baseweb="tab-list"] {overflow-x: auto !important; flex-wrap: nowrap !important;
    scrollbar-width: none; -webkit-overflow-scrolling: touch;}
  [data-baseweb="tab-list"]::-webkit-scrollbar {display: none;}
  [data-baseweb="tab"] {flex: 0 0 auto !important; padding: 0 .5rem !important;}
  [data-baseweb="tab"] p {font-size: .85rem !important; white-space: nowrap !important;}
  [data-testid="stMetricValue"] {font-size: 1.2rem !important;}
  [data-testid="stMetricLabel"] p {font-size: .78rem !important;}
  [data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) {
    flex-wrap: wrap !important; gap: .4rem !important;}
  [data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) > div {
    flex: 1 1 calc(50% - .4rem) !important; min-width: calc(50% - .4rem) !important;}
  [data-testid="stElementContainer"]:has([data-testid="stFormSubmitButton"]),
  [data-testid="stFormSubmitButton"], [data-testid="stFormSubmitButton"] button,
  .stButton button {width: 100% !important; min-height: 46px;}
  [data-testid="stTab"] {padding: 0 .5rem !important; flex: 0 0 auto !important;}
  [data-testid="stTab"] p {font-size: .85rem !important; white-space: nowrap !important;}
  input, textarea {font-size: 16px !important;}
  [data-testid="stCaptionContainer"] {font-size: .8rem !important;}
}
</style>
"""


def is_phone_browser() -> bool:
    """Best guess from the browser's user agent (the sidebar switch overrides it)."""
    try:
        ua = st.context.headers.get("User-Agent", "") or ""
    except Exception:
        return False
    import re
    return bool(re.search(r"Mobi|Android|iPhone|iPod|Windows Phone", ua))


def phone_mode() -> bool:
    try:
        return bool(st.session_state.get("phone_layout", False))
    except Exception:
        return False


def match_cards_html(d: pd.DataFrame) -> str:
    """One card per match - readable on a phone without sideways scrolling.
    `d` must already have the local 'Time' column (see add_local_time)."""
    e = html.escape
    cards = []
    for _, r in d.iterrows():
        adv = str(r.get("Advice", ""))
        cls = "bet" if adv.startswith("BET") else "watch" if adv.startswith("WATCH") else ""
        stake = r.get("Stake", "")
        stake_txt = f" — stake {e(str(stake))}" if isinstance(stake, str) and stake else ""
        lines = [g for g in GOAL_LINES if f"Over {g} %" in r.index and pd.notna(r[f"Over {g} %"])]
        ou = ""
        if lines:
            ou = ("<table class='bfe-ou'><tr><th>Goals</th>"
                  + "".join(f"<th>{g}</th>" for g in lines) + "</tr><tr><th>Over</th>"
                  + "".join(f"<td>{r[f'Over {g} %']:.0f}</td>" for g in lines)
                  + "</tr><tr><th>Under</th>"
                  + "".join(f"<td>{r[f'Under {g} %']:.0f}</td>" for g in lines)
                  + "</tr></table>")
        btts = (f" · BTTS {r['BTTS %']:.0f}%" if "BTTS %" in r.index and pd.notna(r["BTTS %"])
                else "")
        top = r.get("Top 5 scores", "")
        cards.append(
            f"<div class='bfe-card {cls}'>"
            f"<div class='bfe-top'>"
            + (f"<b class='bfe-rank'>#{int(r['Rank'])}</b> " if pd.notna(r.get("Rank", np.nan)) else "")
            + f"{e(str(r.get('Time', '')))} · {e(str(r.get('Competition', '')))}</div>"
            f"<div class='bfe-match'>{e(str(r['Match']))}</div>"
            f"<div class='bfe-adv'>{e(adv)}{stake_txt}</div>"
            + (f"<div class='bfe-row'>🎯 Best pick: <b>{e(str(r['Best pick']))}</b></div>"
               if isinstance(r.get("Best pick"), str) else "")
            + (f"<div class='bfe-row bfe-small'>📝 <b>How to bet:</b> {e(str(r['How to bet']))}</div>"
               if isinstance(r.get("How to bet"), str) else "")
            + (f"<div class='bfe-row bfe-small'>🚑 <b>Missing:</b> {e(str(r['Missing players']))}</div>"
               if isinstance(r.get("Missing players"), str) and r.get("Missing players") else "") +
            f"<div class='bfe-row'>Predicted <b>{e(str(r.get('Predicted score', '')))}</b> · "
            f"H {r['Home %']:.0f}% · D {r['Draw %']:.0f}% · A {r['Away %']:.0f}%{btts}</div>"
            + (f"<div class='bfe-row bfe-small'>Top 5: {e(str(top))}</div>" if top else "")
            + ou + "</div>")
    return "".join(cards)


def render_match_cards(adv: pd.DataFrame, tz: str):
    if adv is None or adv.empty:
        return
    st.markdown(match_cards_html(add_local_time(adv, tz)), unsafe_allow_html=True)


def render_safest_picks(adv: pd.DataFrame, tz: str, all_results: dict, uefa, nations_fx):
    st.markdown("#### ✅ Your bets for each day")
    st.caption("Already checked for you: only picks with 85%+ chance, bookmaker odds "
               "behind them (or 88%+ without), no key player missing for the team you "
               "back, and - where 1xBet's price is known - 1xBet pays at least the minimum. "
               "💚 = ready to place on 1xBet. Otherwise skip a bet if your odds are lower "
               "than the number shown.")
    days = ready_picks_by_day(adv, tz, 5)
    if not days:
        st.info("No bets pass all the checks yet. The list updates four times a day.")
    for k, (day, top) in enumerate(days):
        with st.expander(f"📅 {pd.Timestamp(day):%A %d %B} — {len(top)} bet"
                         f"{'s' if len(top) != 1 else ''}", expanded=(k == 0)):
            st.markdown(ready_cards_html(top), unsafe_allow_html=True)
            st.warning(ticket_line(top))
    st.markdown("#### 📋 Every match: what to bet")
    st.caption("Every match of the day from all leagues and tournaments, highest winning "
               "chance first, with the one bet the system would make - whatever the odds. "
               "✅ = passes all checks (same as "
               "'Your bets'); ⚠️ = riskier, with the reason. Past days show the bet given "
               "before kick-off and whether it won.")
    settled = settle_pick_log(read_pick_log(), result_frames(all_results, uefa, nations_fx))
    sm = pick_summary(settled)
    future = dict(all_matches_by_day(adv, tz))
    past = past_matches_by_day(settled, tz)
    days_all = sorted(set(future) | set(past))
    if not days_all:
        st.info("No matches in the feeds yet.")
    else:
        today = pd.Timestamp.now(tz=tz).date()
        default = today if today in days_all else next((d_ for d_ in days_all if d_ >= today),
                                                       days_all[-1])

        def _lab(d_):
            n_ = len(future.get(d_, [])) + len(past.get(d_, []))
            tag = " (today)" if d_ == today else " (past)" if d_ < today else ""
            return f"{pd.Timestamp(d_):%A %d %B}{tag} — {n_} matches"
        pick_day = st.selectbox("Day", days_all, index=days_all.index(default),
                                format_func=_lab, key="all_matches_day")
        gp, gf = past.get(pick_day), future.get(pick_day)
        comps = sorted((set(gp["Competition"]) if gp is not None else set())
                       | (set(gf["Competition"]) if gf is not None else set()))
        comp = st.selectbox("Competition", ["All competitions"] + comps, key="all_matches_comp")
        if comp != "All competitions":
            gp = gp[gp["Competition"] == comp] if gp is not None else None
            gf = gf[gf["Competition"] == comp] if gf is not None else None
        if gp is not None and not gp.empty:
            st.markdown("**Already played / started**")
            st.info(day_result_line(gp))
            if phone_mode():
                st.markdown(past_cards_html(gp), unsafe_allow_html=True)
            else:
                show_df(pd.DataFrame({"Time": gp["Time"], "Competition": gp["Competition"],
                                      "Match": gp["Match"], "Bet given": gp["Bet"],
                                      "Chance %": gp["Best pick %"].round(0),
                                      "Checked": gp["Checked"], "Result": gp["Result"]}),
                        hide_index=True, height=min(700, 38 + 35 * len(gp)))
        if gf is not None and not gf.empty:
            if gp is not None and not gp.empty:
                st.markdown("**Still to play**")
            g = gf
            if phone_mode():
                st.markdown(all_matches_cards_html(g), unsafe_allow_html=True)
            else:
                tab = pd.DataFrame({
                    "Time": g["Time"], "Competition": g["Competition"], "Match": g["Match"],
                    "What to bet": g["Bet"], "Chance %": g["Best pick %"].round(0),
                    "Fair min odds": g["Min odds"].round(2),
                    "1xBet odds": g.get("1xBet odds", pd.Series(np.nan, index=g.index)),
                    "Check": np.where(g["Check"] == "", "✅ passes all checks",
                                      "⚠️ " + g["Check"])})
                show_df(tab.style.format({"Chance %": "{:.0f}", "Fair min odds": "{:.2f}",
                                          "1xBet odds": "{:.2f}"}, na_rep="—"),
                        hide_index=True, height=min(900, 38 + 35 * len(tab)))
        if (gp is None or gp.empty) and (gf is None or gf.empty):
            st.caption("No matches for this filter.")
    with st.expander("📈 Live track record of best picks (real picks, saved before kick-off)",
                     expanded=bool(sm.get("n"))):
        if not sm.get("n"):
            st.caption(pick_record_lines(sm)[0])
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Best picks won", f"{sm['pct']:.0f}%", f"{sm['won']} of {sm['n']}",
                      delta_color="off")
            sw, sn = sm["strong"]
            c2.metric("🔒 + ⭐⭐⭐ picks won", f"{100*sw/sn:.0f}%" if sn else "—",
                      f"{sw} of {sn}" if sn else "", delta_color="off")
            c3.metric("Waiting for results", sm["pending"])
            st.caption(pick_record_lines(sm)[1])
            rec = sm["recent"][["Kickoff (UTC)", "HomeTeam", "AwayTeam", "Best pick",
                                "Score", "Won"]].copy()
            rec["Won"] = rec["Won"].map({True: "✅", False: "❌"})
            show_df(rec, hide_index=True)
    with st.expander("🛡️ Rules for low-odds betting", expanded=False):
        st.markdown(SAFE_RULES)


def render_day_by_day(adv: pd.DataFrame, tz: str, key: str):
    """Every match from every competition, one day at a time, with advice."""
    if adv is None or adv.empty:
        st.info("No fixtures in the next 7 days are in the feeds yet. League rounds "
                "are usually published mid-week.")
        return
    d = add_local_time(adv, tz)
    days = list(dict.fromkeys(d["_day"]))

    def _label(day):
        sub = d[d["_day"] == day]
        nb = int(sub["Advice"].str.startswith("BET").sum())
        return (f"{pd.Timestamp(day):%A %d %B} — {len(sub)} matches"
                + (f", {nb} ✅ BET" if nb else ""))

    c1, c2 = st.columns([2, 2])
    day = c1.selectbox("Day", days, format_func=_label, key=f"{key}_day")
    comps = ["All competitions"] + list(dict.fromkeys(d.loc[d["_day"] == day, "Competition"]))
    comp = c2.selectbox("Competition", comps, key=f"{key}_comp")
    only = st.toggle("Only value bets (BET / WATCH - leagues only, never cups)", value=False,
                     key=f"{key}_only",
                     help="Value bets come from the league odds model. European cups and "
                          "national teams never get BET advice, so this hides them. For "
                          "safe low-odds picks use '✅ Your bets for each day' at the top.")
    strong = st.toggle("Only 🔒 Banker and ⭐⭐⭐ Strong best picks (won ~91% in testing)", value=False,
                       key=f"{key}_strong")

    sub = d[d["_day"] == day]
    if comp != comps[0]:
        sub = sub[sub["Competition"] == comp]
    slip = bet_slip_lines(sub)
    if slip:
        st.success(f"**Bet slip — {pd.Timestamp(day):%A %d %B}**\n\n"
                   + "\n".join(f"- {line}" for line in slip))
    else:
        st.info("No value bets (✅ BET) this day - that is normal. For safe low-odds "
                "picks, see '✅ Your bets for each day' at the top of the page.")
    n_watch = int(sub["Advice"].str.startswith("WATCH").sum())
    if n_watch:
        st.caption(f"👀 {n_watch} watch-list match(es): logged for paper trading, "
                   "not advised for real money.")
    if "Best pick %" in sub.columns:
        top = sub.dropna(subset=["Best pick %"]).sort_values("Best pick %", ascending=False)
        top = top[(top["Best pick %"] >= BEST_PICK_STRONG * 100)
                  & np.array([ready_filter(r) == "" for _, r in top.iterrows()], dtype=bool)].head(5)
        if not top.empty:
            st.success("**🎯 Strongest best picks today** (same safety checks as 'Your bets')"
                       "\n\n" + "\n".join(
                f"- {r['Time']} · **{r['Match']}** — **{r['Best pick']}**"
                + (f"  \n  📝 {r['How to bet']}" if isinstance(r.get("How to bet"), str) else "")
                for _, r in top.iterrows()))
    view = sub[sub["Advice"].str.match(r"^(BET|WATCH)")] if only else sub
    if strong and "Best pick %" in view.columns:
        view = view[(view["Best pick %"] >= BEST_PICK_STRONG * 100)
                    & np.array([ready_filter(r) == "" for _, r in view.iterrows()], dtype=bool)]

    def _style(row):
        txt = row["Advice"]
        css = ("background-color: #1e7b34; color: white; font-weight: 600"
               if txt.startswith("BET") else
               "background-color: #8a6d0b; color: white" if txt.startswith("WATCH")
               else "")
        return [css if c in ("Advice", "Stake") else "" for c in row.index]

    cols = ["Time", "Competition", "Match", "Advice", "Stake", "Best pick", "How to bet",
            "Missing players", "Predicted score",
            "Top 5 scores", "Home %", "Draw %", "Away %", *GOAL_COLS, "BTTS %"]
    pct_cols = ["Home %", "Draw %", "Away %", *GOAL_COLS, "BTTS %"]
    cols = [c for c in cols if c in view.columns]
    if view.empty:
        st.caption("No matches to show with this filter.")
    elif phone_mode():
        st.markdown(match_cards_html(view), unsafe_allow_html=True)
    else:
        show_df(view[cols].style.apply(_style, axis=1)
                .format({c: "{:.0f}" for c in pct_cols}), hide_index=True,
                height=min(800, 38 + 35 * len(view)))
    st.caption(f"Times shown in {tz} (change it in the sidebar).")
    st.markdown(BEFORE_YOU_BET)


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
PAPER_COLS = ["Logged (UTC)", "League", "Kickoff (UTC)", "Date", "HomeTeam",
              "AwayTeam", "Market", "Book", "Price", "Net odds", "Model %",
              "Edge %", "Goals weight", "Min edge", "Max edge"]
PAPER_KEY = ["League", "Date", "HomeTeam", "AwayTeam", "Market", "Book"]


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
                "League": fx.get("Div", fx.get("League", DEFAULT_LEAGUE)),
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
            log = pd.read_csv(PAPER_LOG)
            if "League" not in log.columns:          # logs from the EPL-only version
                log.insert(1, "League", DEFAULT_LEAGUE)
            return log
    except Exception:
        pass
    return pd.DataFrame(columns=PAPER_COLS)


def update_paper_log(new_rows: pd.DataFrame, allow_all: bool = False):
    """
    Append new flags from 'bet' and 'watch' leagues; the FIRST price seen for
    a match/market/book is kept (that is the bet you would have placed).
    Returns (log, n_added, error).
    """
    log = read_paper_log()
    if not new_rows.empty:
        new_rows = new_rows[new_rows["League"].map(
            lambda c: league_status(c, allow_all) in ("bet", "watch"))]
    if new_rows.empty:
        return log, 0, None
    before = len(log)
    merged = (pd.concat([log, new_rows], ignore_index=True)
              .drop_duplicates(subset=PAPER_KEY, keep="first")
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


def settle_paper_log(log: pd.DataFrame, results) -> pd.DataFrame:
    """Attach final score, win/loss, profit (1 unit) and closing-line value.
    `results` is {league: DataFrame} (or one DataFrame of Premier League results)."""
    if log.empty:
        return log.assign(Score=[], Won=[], Profit=[], Close=[], **{"CLV %": []})
    if isinstance(results, pd.DataFrame):
        results = {DEFAULT_LEAGUE: results}
    frames = []
    for lg, df in results.items():
        if df is None or df.empty:
            continue
        d = df.copy()
        d["League"] = lg
        d["Date"] = d["Date"].dt.strftime("%Y-%m-%d")
        keep = ["League", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"] + [
            c for c in d.columns if c in ODDS_COLUMNS]
        frames.append(d[keep])
    if frames:
        res = pd.concat(frames, ignore_index=True).drop_duplicates(
            ["League", "Date", "HomeTeam", "AwayTeam"])
    else:
        res = pd.DataFrame(columns=["League", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    out = log.merge(res, on=["League", "Date", "HomeTeam", "AwayTeam"], how="left")

    hg = pd.to_numeric(out["FTHG"], errors="coerce")
    ag = pd.to_numeric(out["FTAG"], errors="coerce")
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


# =============================================================================
# SAFETY SWITCH + DATA HEALTH
# =============================================================================
def safety_overrides(settled: pd.DataFrame) -> dict:
    """
    Downgrade a BET league to WATCH when real paper-trading results say the
    edge is not there: 150+ bets with closing prices and an average CLV below
    zero, or 150+ settled bets with an ROI that is clearly negative.
    """
    out = {}
    if settled is None or settled.empty or "League" not in settled.columns:
        return out
    n_min = SAFETY["min_bets"]
    for lg, base in LEAGUE_STATUS.items():
        if base != "bet":
            continue
        s = settled[settled["League"] == lg]
        clv = pd.to_numeric(s["CLV %"], errors="coerce").dropna()
        done = pd.to_numeric(s["Profit"], errors="coerce").dropna()
        if len(clv) >= n_min and clv.mean() < SAFETY["min_avg_clv"]:
            out[lg] = ("watch", f"Auto-downgraded to WATCH: {len(clv)} paper bets with an "
                                f"average CLV of {clv.mean():+.2f}% - not beating the "
                                "closing odds.")
        elif len(done) >= n_min:
            roi = done.mean() * 100
            hi = roi + 1.96 * done.std(ddof=1) / np.sqrt(len(done)) * 100
            if hi < 0:
                out[lg] = ("watch", f"Auto-downgraded to WATCH: ROI {roi:+.1f}% over "
                                    f"{len(done)} paper bets is clearly negative.")
    return out


def promotion_candidates(settled: pd.DataFrame) -> list[str]:
    """Watch leagues whose paper results now look genuinely positive (never
    promoted automatically - shown for your decision)."""
    notes = []
    if settled is None or settled.empty or "League" not in settled.columns:
        return notes
    for lg, base in LEAGUE_STATUS.items():
        if base != "watch":
            continue
        s = settled[settled["League"] == lg]
        clv = pd.to_numeric(s["CLV %"], errors="coerce").dropna()
        done = pd.to_numeric(s["Profit"], errors="coerce").dropna()
        if len(clv) >= SAFETY["min_bets"] and clv.mean() > 1.0 and len(done) > 1:
            lo = done.mean() * 100 - 1.96 * done.std(ddof=1) / np.sqrt(len(done)) * 100
            if lo > 0:
                notes.append(f"{league_name(lg)}: {len(clv)} paper bets, average CLV "
                             f"{clv.mean():+.2f}% and a positive ROI range - a candidate "
                             "for BET status.")
    return notes


def data_health(all_results: dict, fixtures, uefa, intl, intl_fx) -> pd.DataFrame:
    """One row per data source: ✅ fine, ⚠️ check, ❌ problem."""
    rows = []
    today = pd.Timestamp(dt.date.today())
    off_season = today.month in (6, 7)

    def grade(age, ok, warn):
        return "✅" if age <= ok else "⚠️" if age <= warn else "❌"

    def src(df):
        return getattr(df, "attrs", {}).get("source", "live") if df is not None else "-"

    for lg in LEAGUES:
        df = all_results.get(lg)
        if df is None or df.empty:
            continue
        last = df["Date"].max()
        age = (today - last).days
        n_fill = int(df["Filled"].fillna(False).astype(bool).sum()) if "Filled" in df.columns else 0
        rows.append({"Source": f"Results: {league_name(lg)}",
                     "Status": grade(age, 90 if off_season else 10, 120 if off_season else 21),
                     "Latest data": f"{last:%d %b %Y}", "From": src(df),
                     "Details": f"{len(df):,} matches"
                                + (f" ({n_fill} newest from fixturedownload.com - main "
                                   "source behind)" if n_fill else "")})
    if fixtures is None or fixtures.empty:
        rows.append({"Source": "Fixtures & odds (all leagues)", "Status": "❌",
                     "Latest data": "-", "From": src(fixtures),
                     "Details": "No fixtures file - league advice unavailable"})
    else:
        n7 = len(upcoming_fixtures(fixtures))
        rows.append({"Source": "Fixtures (all leagues: odds feed + full schedules)",
                     "Status": "✅" if n7 else "⚠️",
                     "Latest data": f"{fixtures['Kickoff'].max():%d %b %Y}", "From": src(fixtures),
                     "Details": f"{n7} matches in the next 7 days"
                                + ("" if n7 else " (normal mid-week or in international breaks)")})
    if uefa is None or uefa.empty:
        rows.append({"Source": "UEFA cups", "Status": "❌", "Latest data": "-",
                     "From": src(uefa), "Details": "No UEFA data"})
    else:
        played = uefa.dropna(subset=["FTHG"])
        n14 = len(upcoming_fixtures(uefa, 14))
        rows.append({"Source": "UEFA cups (UCL / UEL / UECL)", "Status": "✅",
                     "Latest data": f"{played['Kickoff'].max():%d %b %Y}" if not played.empty else "-",
                     "From": src(uefa), "Details": f"{len(played):,} results, {n14} fixtures in 14 days"})
    if intl is None or intl.empty:
        rows.append({"Source": "National-team results", "Status": "❌", "Latest data": "-",
                     "From": src(intl), "Details": "No international data"})
    else:
        last = intl["Date"].max()
        rows.append({"Source": "National-team results", "Status": grade((today - last).days, 60, 120),
                     "Latest data": f"{last:%d %b %Y}", "From": src(intl),
                     "Details": f"{len(intl):,} matches"})
    n30 = len(upcoming_fixtures(intl_fx, 30)) if intl_fx is not None and not intl_fx.empty else 0
    rows.append({"Source": "National-team fixtures (Nations League, AFCON, ...)",
                 "Status": "✅" if intl_fx is not None and not intl_fx.empty else "⚠️",
                 "Latest data": "-", "From": src(intl_fx),
                 "Details": f"{n30} fixtures in the next 30 days"})
    # Paid / keyed sources: written by the scheduled GitHub run (no keys stored)
    def _status(name):
        try:
            return json.loads((SNAPSHOT_DIR / f"{name}.json").read_text())
        except Exception:
            return {}

    def _age_h(stamp):
        try:
            return (pd.Timestamp.now(tz="UTC")
                    - pd.Timestamp(str(stamp).replace(" UTC", ""), tz="UTC")).total_seconds() / 3600
        except Exception:
            return None
    so = _status(ODDS_STATUS)
    a_ = _age_h(so.get("time"))
    rows.append({"Source": "Cup / national-team odds (The Odds API)",
                 "Status": ("❌" if not so.get("key_present") else
                            "✅" if a_ is not None and a_ < 13 else "⚠️"),
                 "Latest data": so.get("time", "-"), "From": "GitHub run",
                 "Details": f"{so.get('result', 'never run')}"
                            + (f" · credits left this month: {so['credits_remaining']}"
                               if so.get("credits_remaining") else "")})
    si = _status("injuries_status")
    a_ = _age_h(si.get("time"))
    rows.append({"Source": "Missing players + 1xBet prices (API-Football)",
                 "Status": ("❌" if not si.get("key_present") else
                            "✅" if a_ is not None and a_ < 13 and not si.get("errors") else "⚠️"),
                 "Latest data": si.get("time", "-"), "From": "GitHub run",
                 "Details": f"{si.get('result', 'never run')} · {si.get('odds_result', '')}"
                            + (f" · 1xBet prices for {si['my_book_fixtures']} matches"
                               if si.get("my_book_fixtures") is not None else "")
                            + (f" · problems: {si['errors'][:2]}" if si.get("errors") else "")})
    meta = snapshot_meta()
    if not meta.get("saved_at"):
        rows.append({"Source": "Backup copy (data_snapshot)", "Status": "⚠️", "Latest data": "-",
                     "From": "-", "Details": "Not created yet - the scheduled GitHub run "
                                             "creates it on its first run"})
    else:
        age_h = (pd.Timestamp.now(tz="UTC") - pd.Timestamp(meta["saved_at"])).total_seconds() / 3600
        rows.append({"Source": "Backup copy (data_snapshot)",
                     "Status": "✅" if age_h < 13 else "⚠️" if age_h < 48 else "❌",
                     "Latest data": pd.Timestamp(meta["saved_at"]).strftime("%d %b %Y %H:%M UTC"),
                     "From": "-", "Details": f"{len(meta.get('files', {}))} sources saved, "
                                             f"{age_h:.0f} h ago"})
    return pd.DataFrame(rows)


# =============================================================================
# MATCH SEARCH: type a home and an away team, get advice for that match
# =============================================================================
# Common ways people write team names -> the name used in the data
SEARCH_ALIASES = {
    "manchester united": "Man United", "man utd": "Man United", "man u": "Man United",
    "manchester city": "Man City", "mancity": "Man City", "spurs": "Tottenham",
    "tottenham hotspur": "Tottenham", "nottingham forest": "Nott'm Forest",
    "forest": "Nott'm Forest", "wolverhampton": "Wolves", "wolverhampton wanderers": "Wolves",
    "west ham united": "West Ham", "newcastle united": "Newcastle", "leeds united": "Leeds",
    "brighton and hove albion": "Brighton", "brighton & hove albion": "Brighton",
    "sheffield wednesday": "Sheffield Weds", "sheffield utd": "Sheffield United",
    "qpr": "QPR", "queens park rangers": "QPR", "west bromwich albion": "West Brom",
    "psg": "Paris SG", "paris saint-germain": "Paris SG", "paris saint germain": "Paris SG",
    "olympique de marseille": "Marseille", "om": "Marseille", "olympique lyonnais": "Lyon",
    "ol": "Lyon", "as monaco": "Monaco", "losc": "Lille", "saint-etienne": "St Etienne",
    "saint etienne": "St Etienne", "inter milan": "Inter", "internazionale": "Inter",
    "ac milan": "Milan", "as roma": "Roma", "ss lazio": "Lazio", "juve": "Juventus",
    "ssc napoli": "Napoli", "hellas verona": "Verona",
    "atletico madrid": "Ath Madrid", "atletico de madrid": "Ath Madrid", "atleti": "Ath Madrid",
    "athletic bilbao": "Ath Bilbao", "athletic club": "Ath Bilbao", "real betis": "Betis",
    "real sociedad": "Sociedad", "rayo vallecano": "Vallecano", "espanyol": "Espanol",
    "celta vigo": "Celta", "deportivo la coruna": "La Coruna", "fc barcelona": "Barcelona",
    "barca": "Barcelona", "bayern munich": "Bayern Munich", "bayern munchen": "Bayern Munich",
    "bayern": "Bayern Munich", "borussia dortmund": "Dortmund", "bvb": "Dortmund",
    "bayer leverkusen": "Leverkusen", "eintracht frankfurt": "Ein Frankfurt",
    "borussia monchengladbach": "M'gladbach", "gladbach": "M'gladbach",
    "rb leipzig": "RB Leipzig", "leipzig": "RB Leipzig", "fc koln": "FC Koln",
    "cologne": "FC Koln", "koln": "FC Koln", "werder": "Werder Bremen",
    "sporting lisbon": "Sp Lisbon", "sporting cp": "Sp Lisbon", "sporting": "Sp Lisbon",
    "sc braga": "Sp Braga", "braga": "Sp Braga", "fc porto": "Porto", "sl benfica": "Benfica",
    "vitoria guimaraes": "Guimaraes", "psv": "PSV Eindhoven", "az": "AZ Alkmaar",
    "nec nijmegen": "Nijmegen", "nec": "Nijmegen", "union saint-gilloise": "St. Gilloise",
    "union sg": "St. Gilloise", "olympiacos": "Olympiakos", "aek athens": "AEK",
    "fenerbahce": "Fenerbahce", "besiktas": "Besiktas", "basaksehir": "Buyuksehyr",
    "istanbul basaksehir": "Buyuksehyr", "celtic glasgow": "Celtic", "glasgow rangers": "Rangers",
    "cote d'ivoire": "Ivory Coast", "cote divoire": "Ivory Coast", "cameroun": "Cameroon",
    "rd congo": "DR Congo", "congo dr": "DR Congo", "drc": "DR Congo",
    "cabo verde": "Cape Verde", "usa": "United States", "turkiye": "Turkey",
    "czechia": "Czech Republic", "korea republic": "South Korea", "ir iran": "Iran",
    "holland": "Netherlands", "republic of ireland": "Republic of Ireland",
    "ksa": "Saudi Arabia", "uae": "United Arab Emirates", "emirates": "United Arab Emirates",
    "korea": "South Korea", "china pr": "China", "zenit st petersburg": "Zenit",
}


def _norm(name: str) -> str:
    import unicodedata
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    s = s.lower().replace(".", "").replace("’", "'").strip()
    for junk in (" fc", "fc ", " cf", " afc", " sc"):
        if s.endswith(junk) or s.startswith(junk.strip() + " "):
            s = s.replace(junk.strip(), "", 1).strip()
    return " ".join(s.split())


def team_directory(all_models: dict, uefa_model, intl_model) -> dict:
    """{team name: [(kind, league code or None), ...]} for every team we can rate."""
    d = {}
    for lg, m in all_models.items():
        for t in m.get("current_teams", m.get("teams", [])):
            d.setdefault(t, []).append(("league", lg))
    if uefa_model:
        for t in uefa_model["teams"]:
            # list UEFA clubs under their league spelling ("Man Utd" -> "Man United")
            # so a match between two league clubs is always treated as a league match
            name = UEFA_CLUB_MAP[t][1] if t in UEFA_CLUB_MAP else t
            d.setdefault(name, []).append(("uefa", t))
    if intl_model:
        for t in intl_model["current_teams"]:
            d.setdefault(t, []).append(("intl", None))
    return d


def resolve_team(query: str, names: list) -> tuple:
    """(team, suggestions, exact). exact=True only for an exact or alias match -
    anything fuzzy must be confirmed by the user, never silently guessed."""
    import difflib
    q = _norm(query)
    if not q:
        return None, [], False
    by_norm = {}
    for n in names:
        by_norm.setdefault(_norm(n), n)
    if q in by_norm:
        return by_norm[q], [by_norm[q]], True
    if q in SEARCH_ALIASES:
        target = _norm(SEARCH_ALIASES[q])
        if target in by_norm:
            return by_norm[target], [by_norm[target]], True
    partial = [n for k, n in by_norm.items()
               if len(q) >= 3 and (k.startswith(q) or f" {q}" in f" {k}")]
    close = [by_norm[k] for k in difflib.get_close_matches(q, list(by_norm), n=5, cutoff=0.75)]
    sugg = list(dict.fromkeys(partial + close if len(q) <= 4 else close + partial))[:6]
    return (sugg[0] if sugg else None), sugg, False


def _cross_league_model(uefa_model: dict, all_models: dict, teams: dict) -> dict:
    """Clubs from different leagues: domestic rating + how strong each league is
    in Europe (from the UEFA model)."""
    m = dict(uefa_model)
    att, dfn = dict(uefa_model["attack"]), dict(uefa_model["defence"])
    for t, lg in teams.items():
        lm = all_models.get(lg)
        a, d = uefa_model["groups"].get(lg, (0.0, 0.0))
        if lm is not None and t in lm["attack"]:
            att[t], dfn[t] = lm["attack"][t] + a, lm["defence"][t] + d
    m["attack"], m["defence"] = att, dfn
    return m


def match_query(home: str, away: str, directory: dict, all_models: dict, uefa_model,
                intl_model, fixtures: pd.DataFrame, book: str, min_edge: float,
                max_edge: float, bankroll: float, allow_all: bool, neutral: bool) -> dict:
    """Pick the right model for the two teams and build the advice."""
    kh, ka = directory.get(home, []), directory.get(away, [])
    lh = {lg for k, lg in kh if k == "league"}
    la = {lg for k, lg in ka if k == "league"}
    common = sorted(lh & la, key=lambda c: STATUS_RANK[league_status(c, allow_all)])
    out = {"home": home, "away": away, "neutral": neutral, "mh": home, "ma": away}
    if common:
        lg = common[0]
        model = all_models[lg]
        out.update(model=model, competition=league_name(lg), league=lg,
                   status=league_status(lg, allow_all))
        fx = pd.DataFrame()
        if fixtures is not None and not fixtures.empty:
            up = upcoming_fixtures(fixtures, 14)
            fx = up[(up["Div"] == lg) & (up["HomeTeam"] == home) & (up["AwayTeam"] == away)]
        if not fx.empty and not neutral:
            fx = fx.head(1)
            flags = flagged_bets(fx, model, book, min_edge, max_edge)
            adv = match_advice(score_predictions(fx, model), flags, bankroll, allow_all)
            out.update(fixture=fx.iloc[0], advice=adv.iloc[0]["Advice"],
                       stake=adv.iloc[0]["Stake"], odds=fixture_odds(fx.iloc[0], book))
        return out
    if any(k == "intl" for k, _ in kh) and any(k == "intl" for k, _ in ka):
        out.update(model=intl_model, competition="National teams", league=None, status="cup")
        return out
    uh = next((u for k, u in kh if k == "uefa"), None)
    ua = next((u for k, u in ka if k == "uefa"), None)
    if uefa_model is not None and uh and ua:
        out.update(model=uefa_model, competition="European club match", league=None,
                   status="cup", mh=uh, ma=ua)          # UEFA-feed spellings
        return out
    if uefa_model is not None and lh and la:
        m = _cross_league_model(uefa_model, all_models,
                                {home: sorted(lh)[0], away: sorted(la)[0]})
        out.update(model=m, competition=f"{league_name(sorted(lh)[0])} v "
                                        f"{league_name(sorted(la)[0])} (cross-league)",
                   league=None, status="cup")
        return out
    out["error"] = ("These two teams can't be compared (for example a club against a "
                    "national team).")
    return out


def render_match_search(directory: dict, all_models: dict, uefa_model, intl_model,
                        fixtures, book, min_edge, max_edge, bankroll, allow_all,
                        week_adv: pd.DataFrame | None = None, tz: str = DEFAULTS["timezone"]):
    st.markdown("#### 🔎 Ask about any match")
    with st.form("match_search_form", border=False):
        q1, q2, q3 = st.columns([3, 3, 1])
        home_q = q1.text_input("Home team", key="q_home", placeholder="e.g. Arsenal")
        away_q = q2.text_input("Away team", key="q_away", placeholder="e.g. Chelsea")
        neutral = q3.toggle("Neutral venue", key="q_neutral",
                            help="Finals and tournament games at a neutral ground.")
        st.form_submit_button("🔎 Get betting advice", type="primary")
    if not (home_q and away_q):
        st.caption("Type two teams from any of the 25 leagues, the UEFA cups or "
                   "national teams (e.g. 'Man Utd' and 'PSG', or 'Cameroon' and "
                   "'Nigeria') to get advice for that match.")
        return
    names = list(directory)
    picks = []
    for label, q, k in (("Home team", home_q, "q_home_pick"), ("Away team", away_q, "q_away_pick")):
        best, sugg, exact = resolve_team(q, names)
        if best is None:
            st.warning(f"No team found for '{q}'. Check the spelling, or try the "
                       "short name (e.g. 'Man United', 'Paris SG', 'Ivory Coast').")
            return
        if not exact:                     # never guess silently - ask
            best = st.selectbox(f"{label}: no exact match for '{q}' - did you mean...",
                                sugg, key=k)
        picks.append(best)
    home, away = picks
    if home == away:
        st.warning("Pick two different teams.")
        return
    res = match_query(home, away, directory, all_models, uefa_model, intl_model, fixtures,
                      book, min_edge, max_edge, bankroll, allow_all, neutral)
    if "error" in res:
        st.warning(res["error"])
        return
    model = res["model"]
    lh_, la_, m, p = predict(model, res["mh"], res["ma"], neutral=neutral)
    ps, pp = predicted_score(m, p)
    st.markdown(f"**{home} v {away}** — {res['competition']} "
                f"({STATUS_LABEL[res['status']]})")
    pp_, used_odds = pick_probs(m, model["rho"], res.get("odds") or {})
    bp = best_pick(shrink_probs(pp_, model.get("pick_shrink", 1.0)), home, away, min_edge)
    st.success(f"🎯 **Best pick: {bp['label']}** {bp['stars']} ({bp['tier']}) — "
               f"{bp['prob']*100:.0f}% chance"
               + (" (model + bookmaker odds)" if used_odds else "")
               + f". {bp['tier']} picks won {bp['tier_hit']} of the time in testing.\n\n"
               f"📝 **How to bet:** {bp['how']}")
    # Same fixture in this week's list? Then show the system's checked verdict
    # (it also uses missing players and 1xBet's price).
    if week_adv is not None and not week_adv.empty and "Pick key" in week_adv.columns:
        names = {f"{res['mh']} v {res['ma']}", f"{home} v {away}"}
        hit = week_adv[week_adv["Match"].isin(names)]
        if not hit.empty:
            r = add_local_time(hit, tz).iloc[0]
            h_, a_ = r["Match"].split(" v ", 1)
            chk = ready_filter(r)
            price = r.get("1xBet odds", np.nan)
            msg = (f"📅 **This match is on {r['Day']} at {r['Time']}.** With missing players and "
                   f"bookmaker prices included, the system's bet is: **{short_bet(r['Pick key'], h_, a_)}** "
                   f"({r['Best pick %']:.0f}% chance"
                   + (f", 1xBet {float(price):.2f}" if pd.notna(price) else "") + "). ")
            if chk == "":
                st.success(msg + "✅ It passes all checks - it is in 'Your bets'.")
            else:
                st.warning(msg + f"⚠️ Riskier: {chk}.")
            if isinstance(r.get("Missing players"), str) and r.get("Missing players"):
                st.caption(f"🚑 Missing: {r['Missing players']}")
    adv = res.get("advice")
    if adv is not None:
        if adv.startswith("BET"):
            st.success(f"✅ **{adv}** — stake {res['stake']}")
        elif adv.startswith("WATCH"):
            st.warning(f"👀 **{adv}** (not advised for real money)")
        else:
            st.info(f"**{adv}**")
    else:
        best = max(("H", "D", "A"), key=lambda k: p[k])
        lab = {"H": f"{home} win", "D": "Draw", "A": f"{away} win"}[best]
        why = ("this match isn't in the odds feed in the next 14 days"
               if res["league"] else "there is no odds feed for this kind of match")
        st.info(f"No bookmaker odds available ({why}). Most likely: **{lab}** "
                f"({p[best]*100:.0f}%). Bet only if your bookmaker offers at least "
                f"**{(1 / p[best]) * (1 + min_edge / 100):.2f}**.")
        if res["status"] == "predict":
            st.caption("Note: the model lost money in this league in testing, so treat "
                       "these numbers as information, not betting advice.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"{home} xG", f"{lh_:.2f}")
    c2.metric(f"{away} xG", f"{la_:.2f}")
    c3.metric("Predicted score", ps, delta=f"{pp*100:.0f}% chance", delta_color="off")
    c4.metric("H / D / A", f"{p['H']*100:.0f} / {p['D']*100:.0f} / {p['A']*100:.0f}%")
    odds = res.get("odds", {})
    rows = []
    for k in ["H", "D", "A", *[k for g in GOAL_LINES for k in ("O" + g.replace(".", ""), "U" + g.replace(".", ""))],
              "BTTS_Y", "BTTS_N"]:
        pr = p[k]
        bo = odds.get(k, np.nan)
        rows.append({"Market": MARKET_LABELS[k].replace("Home Win", f"{home} win")
                                               .replace("Away Win", f"{away} win"),
                     "Model %": pr * 100, "Fair odds": 1 / pr,
                     "Bet only at odds of at least": (1 / pr) * (1 + min_edge / 100),
                     f"{book} odds": bo,
                     "Edge %": (pr * bo - 1) * 100 if pd.notna(bo) else np.nan})
    mk = pd.DataFrame(rows)
    if phone_mode():
        mk = mk.rename(columns={"Bet only at odds of at least": "Bet at ≥"})
        keep = ["Market", "Model %", "Bet at ≥"] + (
            [f"{book} odds", "Edge %"] if mk[f"{book} odds"].notna().any() else [])
        show_df(mk[keep].style.format({"Model %": "{:.0f}", "Bet at ≥": "{:.2f}",
                                       f"{book} odds": "{:.2f}", "Edge %": "{:+.1f}"},
                                      na_rep="—"), hide_index=True)
    else:
      show_df(mk.style.format(
        {"Model %": "{:.1f}", "Fair odds": "{:.2f}", "Bet only at odds of at least": "{:.2f}",
         f"{book} odds": "{:.2f}", "Edge %": "{:+.1f}"}, na_rep="—"), hide_index=True)
    st.caption("Top 5 scores: " + ", ".join(f"{sc} ({pr*100:.0f}%)" for sc, pr in top_scores(m))
               + ". Check team news before betting; adjust for injuries in the Match "
                 "Analyzer.")


@contextmanager
def safe_section(name: str):
    """Keep one failing section from taking down the whole app."""
    try:
        yield
    except Exception as exc:
        if type(exc).__name__ in ("StopException", "RerunException", "RerunData"):
            raise                                  # Streamlit control flow
        st.error(f"Something went wrong in '{name}': {exc}. The rest of the app still "
                 "works, and this is retried automatically on the next refresh.")


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
    st.set_page_config(page_title="BOUBA's Football Engine", page_icon="⚽", layout="wide",
                       initial_sidebar_state="auto")
    st.markdown(MOBILE_CSS, unsafe_allow_html=True)
    if "phone_layout" not in st.session_state:
        st.session_state["phone_layout"] = is_phone_browser()

    # ---------------- Sidebar: global settings ----------------
    st.sidebar.toggle("🟢 Under 4.5 / 5.5 mode (show only these picks)", value=True,
                      key="under_mode",
                      help="On: the app shows only matches where Under 4.5 / Under 5.5 has "
                           "an 80%+ chance. Off: the full app with every section.")
    st.sidebar.toggle("📱 Phone layout (match cards)", key="phone_layout",
                      help="Shows each match as a card instead of a wide table. "
                           "Switched on automatically on phones.")
    st.sidebar.header("⚙️ Model settings")
    codes = list(LEAGUES)
    league = st.sidebar.selectbox(
        "League for the detailed tabs", codes, index=codes.index(DEFAULT_LEAGUE),
        format_func=lambda c: f"{league_name(c)}  {STATUS_LABEL[league_status(c)].split()[0]}",
        help="✅ = BET advice, 👀 = paper-trade only, 📊 = predictions only.")
    allow_all = st.sidebar.toggle(
        "Show BET advice in every league (not recommended)", value=False,
        help="Overrides the backtest results. The model lost money in most leagues.")
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
    tz = st.sidebar.selectbox("Your time zone", TIMEZONES,
                              index=TIMEZONES.index(DEFAULTS["timezone"]))
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

    # ---------------- Data + models ----------------
    st.title("⚽ BOUBA's Football Engine")
    st.caption(f"{len(LEAGUES)} leagues (Europe, Russia, Japan, China), UEFA cups and national teams. BET advice only "
               "where testing showed a profit; everywhere else you get predictions.")
    with st.spinner("Fetching fixtures..."):
        fixtures = load_fixtures()
    window_all = upcoming_fixtures(fixtures)
    log_existing = read_paper_log()
    # Every league: the match search covers all teams, and the UEFA model uses
    # domestic ratings (fast when the backup copy is fresh)
    needed = {league} | set(UEFA_LEAGUES) | set(LEAGUES)
    if not window_all.empty:
        needed |= set(window_all["Div"])
    if not log_existing.empty:
        needed |= set(log_existing["League"].dropna().astype(str)) & set(LEAGUES)
    with st.spinner(f"Downloading results for {len(needed)} league(s) "
                    "(the first load can take about 30 seconds)..."):
        all_results = load_all_results(tuple(sorted(needed)), seasons_back)
    fixtures = add_schedule(fixtures, load_schedule(), all_results)
    window_all = upcoming_fixtures(fixtures)
    results = all_results.get(league, pd.DataFrame())

    if results.empty:
        st.error(f"Could not download {league_name(league)} results from "
                 "Football-Data.co.uk right now. The app retries automatically; "
                 "or press 'Force data refresh'.")
        st.stop()

    # Safety switch: paper-trading results can downgrade a BET league
    settled_all = (settle_paper_log(log_existing, all_results) if not log_existing.empty
                   else pd.DataFrame())
    STATUS_OVERRIDES.clear()
    STATUS_OVERRIDES.update(safety_overrides(settled_all))

    inj_table = load_injury_table()
    book_odds = load_book_odds()
    with st.spinner("Fitting models for every league..."):
        ov_adv, ov_flags, ov_models = weekly_overview(
            fixtures, all_results, half_life, goals_weight, book, min_edge,
            max_edge, bankroll, allow_all, inj_table, book_odds)
    model = ov_models.get(league) or fit_model(results, half_life, goals_weight)

    with st.spinner("Loading UEFA competitions and national teams..."):
        uefa = load_uefa()
        nations_fx = load_intl_fixtures()
        intl = load_international()
        all_models = league_models(all_results, half_life, goals_weight)
        uefa_model = None
        if not uefa.empty:
            uefa_teams = sorted(set(uefa["HomeTeam"]) | set(uefa["AwayTeam"]))
            uefa_model = fit_uefa(uefa, uefa_domestic(uefa_teams, all_models))
        intl_model = fit_international(intl) if not intl.empty else None
        cup_odds = load_cup_odds()
    week_adv = combine_overview(
        ov_adv, cup_predictions(uefa, uefa_model, "Comp", 7, min_edge, cup_odds, inj_table, book_odds),
        cup_predictions(nations_fx, intl_model, "Comp", 7, min_edge, cup_odds, inj_table, book_odds))

    age_days = (pd.Timestamp(dt.date.today()) - model["last_date"]).days
    with st.expander(f"📈 Model status - {league_name(league)} "
                     f"({STATUS_LABEL[league_status(league, allow_all)]})"):
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Matches in model", f"{model['n_matches']:,}")
        c2.metric("This-season matches", model["n_current"])
        c3.metric("Latest result", model["last_date"].strftime("%d %b %Y"))
        c4.metric("Home advantage", f"×{np.exp(model['home']):.2f}")
        c5.metric("Dixon-Coles ρ", f"{model['rho']:+.3f}")
        xg_note = (f"{model['n_xg']} matches use real xG" if model["n_xg"]
                   else "shots on target used as xG stand-in")
        sot = model["goals_per_sot"]
        st.caption(f"Rating signal: {goals_pct}% actual goals, {100 - goals_pct}% shot "
                   f"quality ({xg_note}"
                   + (f"; {sot:.2f} goals per shot on target" if np.isfinite(sot) else
                      "; no shot data in this league, goals only") + ").")
        st.caption(f"Backtest of flagged bets in this league: "
                   f"{LEAGUE_EVIDENCE.get(league, 'not tested')}.")
        if not model["converged"]:
            st.warning("Optimiser reported non-convergence; ratings may be approximate.")
    if age_days > 10:
        st.warning(f"The newest {league_name(league)} result in the feed is {age_days} "
                   "days old, so ratings miss recent matches. Football-Data.co.uk "
                   "usually updates twice a week; the app picks it up automatically.")

    # Fixtures of the selected league that have not kicked off (next 7 days).
    # Teams unknown to the model fall back to the promoted-team prior.
    window = (window_all[window_all["Div"] == league] if not window_all.empty
              else window_all)

    # Paper trading: log flags from 'bet' and 'watch' leagues
    log_note = None
    if autolog and not ov_flags.empty:
        _, added, err = update_paper_log(ov_flags, allow_all)
        log_note = (f"Could not write {PAPER_LOG}: {err}" if err
                    else f"{added} new value bet(s) logged." if added else None)

    # Data health: sidebar indicator + full table in its own tab
    health = data_health(all_results, fixtures, uefa, intl, nations_fx)
    n_bad = int((health["Status"] == "❌").sum())
    n_warn = int((health["Status"] == "⚠️").sum())
    st.sidebar.markdown("---")
    if n_bad:
        st.sidebar.error(f"Data health: {n_bad} problem(s) - see the 🩺 System Health tab")
    elif n_warn:
        st.sidebar.warning(f"Data health: {n_warn} warning(s) - see the 🩺 System Health tab")
    else:
        st.sidebar.success("Data health: all sources OK")
    for lg, (_, why) in STATUS_OVERRIDES.items():
        st.warning(f"🛡️ {league_name(lg)}: {why}")

    if st.session_state.get("under_mode", True):
        with safe_section("Under picks"):
            render_under_section(week_adv, tz)
        st.caption("Switch off '🟢 Under 4.5 / 5.5 mode' in the side menu (☰ / ›) to see "
                   "all other sections.")
        with st.expander("🛡️ Rules for low-odds betting", expanded=False):
            st.markdown(SAFE_RULES)
        st.stop()

    # Safest picks of the day, the betting rules and the live track record
    with safe_section("Safest picks"):
        render_safest_picks(week_adv, tz, all_results, uefa, nations_fx)

    # Match search at the top of the page
    with safe_section("Match search"):
        directory = team_directory(all_models, uefa_model, intl_model)
        render_match_search(directory, all_models, uefa_model, intl_model, fixtures,
                            book, min_edge, max_edge, bankroll, allow_all, week_adv, tz)
    st.markdown("---")

    if phone_mode():
        st.caption("👉 Swipe the tab bar sideways to see all 9 sections.")
    (tab_week, tab_value, tab_match, tab_ratings, tab_bt, tab_paper, tab_uefa,
     tab_intl, tab_health) = st.tabs(["🗓️ This Week: Everything", "📋 Value Table",
                                      "🔍 Match Analyzer", "📊 Team Ratings", "🧾 Backtest",
                                      "📒 Paper Trading", "🏆 UEFA Cups", "🌍 National Teams",
                                      "🩺 System Health"])

    # ================= TAB 0: This week, every league =================
    with tab_week, safe_section("This Week"):
        st.markdown("### What to do — match by match, day by day")
        st.caption("Every match in the next 7 days from all 25 leagues, the UEFA cups "
                   "and national-team tournaments, with advice.")
        if week_adv.empty:
            render_day_by_day(week_adv, tz, "week")
        else:
            n_bet = int(week_adv["Advice"].str.startswith("BET").sum())
            n_watch = int(week_adv["Advice"].str.startswith("WATCH").sum())
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Matches", len(week_adv))
            k2.metric("Leagues", week_adv["League"].nunique())
            k3.metric("✅ Bets", n_bet)
            k4.metric("👀 Watch", n_watch)
            render_day_by_day(week_adv, tz, "week")
            st.caption("✅ BET = value price in a league where the model made money in "
                       "testing (stake = 1/4 Kelly, max "
                       f"{DEFAULTS['max_stake_pct']:.0f}% of bankroll). 👀 WATCH = "
                       "about break-even in testing: logged for paper trading only. "
                       "NO BET (model not profitable) = the model lost money in that "
                       "league in testing. 🏆 cups and national teams have no odds "
                       "feed: compare the 'back only at' odds with your bookmaker.")
        with st.expander("Why only some leagues get BET advice"):
            ev = pd.DataFrame([{"Competition": league_name(c),
                                "Status": STATUS_LABEL[league_status(c)],
                                "ROI of flagged bets in testing": LEAGUE_EVIDENCE.get(c, "-")}
                               for c in LEAGUES])
            show_df(ev, hide_index=True)
            st.caption("Each season was replayed week by week using only past matches, "
                       "betting every 3-20% edge at Betfair Exchange prices. Only the "
                       "Premier League made money in both seasons tested; the same "
                       "model lost money in most other leagues, where the bookmakers' "
                       "prices were more accurate than it is.")

    # ================= TAB 1: Master value table =================
    with tab_value, safe_section("Value Table"):
        st.markdown(f"### {league_name(league)}: fixtures ranked by edge — odds: **{book}**")
        if league_status(league, allow_all) == "predict":
            st.warning("The model lost money in this league in testing, so these edges "
                       "are shown for information only and are never turned into BET "
                       "advice.")
        if window.empty:
            st.info(f"No {league_name(league)} fixtures still to kick off in the next "
                    "7 days are in the feed yet (it is usually published mid-week, and "
                    "it is empty during international breaks). Use the Match Analyzer "
                    "meanwhile.")
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
                if phone_mode():
                    table = table[["Match", "Market", "Book Odds", "Fair Odds", "Edge %"]]
                show_df(style_value(table, min_edge, max_edge), hide_index=True,
                             height=600)
                st.caption("Edge % = model probability × book odds − 1. "
                           "¼ Kelly % = suggested stake as % of bankroll (quarter-Kelly). "
                           "What-If sliders do not affect this table.")

        st.markdown("### What to do — match by match (all competitions, day by day)")
        render_day_by_day(week_adv, tz, "value")
        st.caption("Predicted score = most likely score within the most likely "
                   "result. In testing it got the winner right ~50% of the time "
                   "and the exact score ~11%; the top 5 scores covered ~50%.")

    # ================= TAB 2: Match analyzer =================
    with tab_match, safe_section("Match Analyzer"):
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
        pp_, used_odds = pick_probs(m, model["rho"], odds)
        bp = best_pick(pp_, home, away, min_edge)
        st.success(f"🎯 **Best pick: {bp['label']}** {bp['stars']} ({bp['tier']}) — "
                   f"{bp['prob']*100:.0f}% chance"
                   + (" (model + bookmaker odds)" if used_odds else "")
                   + f".\n\n📝 **How to bet:** {bp['how']}")

        st.markdown("#### Match outcome")
        o1, o2, o3 = st.columns(3)
        for col, k in zip((o1, o2, o3), ("H", "D", "A")):
            col.metric(MARKET_LABELS[k], f"{probs[k]*100:.1f}%",
                       delta=f"Fair odds {1/probs[k]:.2f}", delta_color="off")

        st.markdown("#### Goals markets")
        gcols = st.columns(4) + st.columns(4)
        for col, g in zip(gcols, GOAL_LINES):
            k = g.replace(".", "")
            col.metric(f"Over / Under {g}",
                       f"{probs['O' + k]*100:.0f}% / {probs['U' + k]*100:.0f}%",
                       delta=f"Fair {1/probs['O' + k]:.2f} / {1/probs['U' + k]:.2f}",
                       delta_color="off")
        b1, b2, _ = st.columns(3)
        b1.metric("BTTS Yes", f"{probs['BTTS_Y']*100:.1f}%",
                  delta=f"Fair {1/probs['BTTS_Y']:.2f}", delta_color="off")
        b2.metric("BTTS No", f"{probs['BTTS_N']*100:.1f}%",
                  delta=f"Fair {1/probs['BTTS_N']:.2f}", delta_color="off")

        st.markdown(f"#### Value check vs {book}")
        keys = ["H", "D", "A", *[k for g in GOAL_LINES for k in ("O" + g.replace(".", ""), "U" + g.replace(".", ""))],
                "BTTS_Y", "BTTS_N"]
        vt = pd.DataFrame(value_rows(probs, odds, keys))
        show_df(style_value(vt, min_edge, max_edge), hide_index=True)
        if not odds:
            st.caption("No bookmaker odds in the feed for this pairing — fair odds only.")
        st.caption("The Football-Data feed publishes 1X2 and O/U 2.5 odds only; "
                   "compare the fair odds for the other goal lines and BTTS against your "
                   "bookmaker yourself.")

        st.markdown("#### Most likely scorelines")
        flat = [(f"{i}-{j}", m[i, j]) for i in range(m.shape[0]) for j in range(m.shape[1])]
        flat.sort(key=lambda t: t[1], reverse=True)
        top = pd.DataFrame(flat[:8], columns=["Score", "Prob"])
        top["Prob %"] = (top["Prob"] * 100).round(1)
        top["Fair Odds"] = (1 / top["Prob"]).round(1)
        show_df(top[["Score", "Prob %", "Fair Odds"]], hide_index=True)

    # ================= TAB 3: Ratings =================
    with tab_ratings, safe_section("Team Ratings"):
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
    with tab_bt, safe_section("Backtest"):
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
                                     format_func=lambda c: c if league in NEW_FORMAT
                                     else f"20{c[:2]}/{c[2:]}")
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
    with tab_paper, safe_section("Paper Trading"):
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
            settled = settle_paper_log(log, all_results)
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
            view["League"] = view["League"].map(league_name)
            show_df(view[["Kickoff (UTC)", "League", "HomeTeam", "AwayTeam", "Market", "Book",
                          "Price", "Model %", "Edge %", "Score", "Profit", "Close",
                          "CLV %"]].style.format(
                {"Price": "{:.2f}", "Model %": "{:.1f}", "Edge %": "{:+.1f}",
                 "Profit": "{:+.2f}", "Close": "{:.2f}", "CLV %": "{:+.1f}"},
                na_rep="—"), hide_index=True)
            st.download_button("⬇️ Download log (CSV)", settled.to_csv(index=False),
                               file_name="paper_bets_settled.csv", mime="text/csv")
            st.caption(f"Log file: {PAPER_LOG}")

    # ================= TAB: UEFA club competitions =================
    with tab_uefa, safe_section("UEFA Cups"):
        st.markdown("### UEFA Champions League, Europa League & Conference League")
        if uefa_model is None:
            st.error("Could not download the UEFA fixtures right now. The app retries "
                     "automatically.")
        else:
            st.caption(f"Model fitted on {uefa_model['n_matches']:,} European matches "
                       f"(last {UEFA_DEFAULTS['seasons']} seasons) on top of the domestic "
                       "league ratings. In testing on the whole 2025/26 European season "
                       "it picked the right result 56.5% of the time. There is no free "
                       "odds feed for these games, so it never says BET: compare the "
                       "'back only at' odds with your bookmaker.")
            up = upcoming_fixtures(uefa, 14)
            st.markdown("#### Next 14 days")
            if up.empty:
                st.info("No UEFA club fixtures in the next 14 days.")
            else:
                cup_tab = cup_predictions(uefa, uefa_model, "Comp", 14, min_edge, cup_odds,
                                          inj_table, book_odds)
                if phone_mode():
                    render_match_cards(cup_tab, tz)
                else:
                  show_df(cup_tab.drop(columns=["League", "Status", "Stake", "Kickoff UTC"]).style.format(
                    {c: "{:.0f}" for c in ["Home %", "Draw %", "Away %", *GOAL_COLS, "BTTS %"]}), hide_index=True,
                    height=min(700, 38 + 35 * max(len(cup_tab), 1)))
            st.markdown("#### Any two clubs")
            clubs_u = sorted(uefa_model["teams"])
            u1, u2, u3 = st.columns([2, 2, 1])
            ch = u1.selectbox("Home club", clubs_u, index=clubs_u.index("Real Madrid")
                              if "Real Madrid" in clubs_u else 0)
            ca = u2.selectbox("Away club", clubs_u, index=clubs_u.index("Man City")
                              if "Man City" in clubs_u else 1)
            neutral_u = u3.toggle("Neutral venue (final)", value=False)
            if ch != ca:
                lh_u, la_u, m_u, p_u = predict(uefa_model, ch, ca, neutral=neutral_u)
                ps_u, pp_u = predicted_score(m_u, p_u)
                v1, v2, v3, v4 = st.columns(4)
                v1.metric(f"{ch} xG", f"{lh_u:.2f}")
                v2.metric(f"{ca} xG", f"{la_u:.2f}")
                v3.metric("Predicted score", ps_u, delta=f"{pp_u*100:.0f}% chance",
                          delta_color="off")
                v4.metric("Over 2.5 goals", f"{p_u['O25']*100:.0f}%")
                rows_u = []
                for k in ["H", "D", "A", "O25", "U25", "BTTS_Y", "BTTS_N"]:
                    pr = p_u[k]
                    label = (MARKET_LABELS[k].replace("Home Win", f"{ch} win")
                             .replace("Away Win", f"{ca} win"))
                    rows_u.append({"Market": label, "Model %": pr * 100, "Fair odds": 1 / pr,
                                   "Bet only at odds of at least": (1 / pr) * (1 + min_edge / 100)})
                show_df(pd.DataFrame(rows_u).style.format(
                    {"Model %": "{:.1f}", "Fair odds": "{:.2f}",
                     "Bet only at odds of at least": "{:.2f}"}), hide_index=True)
                st.caption("Knockout games are predicted for 90 minutes (extra time and "
                           "penalties not included).")
            with st.expander("How strong each league / country is in Europe"):
                gs = pd.DataFrame([{"League / country": league_name(g) if g in LEAGUES else g,
                                    "Attack ×": np.exp(a), "Defence ×": np.exp(d),
                                    "Net strength": np.exp(a + d)}
                                   for g, (a, d) in uefa_model["groups"].items()])
                gs = gs.sort_values("Net strength", ascending=False)
                show_df(gs.style.format({c: "{:.2f}" for c in
                                         ["Attack ×", "Defence ×", "Net strength"]}),
                        hide_index=True)
                st.caption("Relative to an average European-cup team, on top of each "
                           "club's domestic rating. Country codes are leagues the app "
                           "doesn't cover domestically (e.g. AUT = Austria).")

    # ================= TAB 7: National teams =================
    with tab_intl, safe_section("National Teams"):
        st.markdown("### National teams — AFCON, World Cup, Euros, Nations League, qualifiers")
        if intl.empty or intl_model is None:
            st.error("Could not download the international results right now. "
                     "The app retries automatically.")
        else:
            im = intl_model
            nl_up = cup_predictions(nations_fx, im, "Comp", 30, min_edge, cup_odds, inj_table, book_odds)
            st.markdown("#### Upcoming fixtures (next 30 days)")
            st.caption("Accuracy in testing (Jul 2024 - Oct 2026, model refitted on past "
                       "matches only): right result in 62.9% of 423 competitive games "
                       "between African teams and 69.2% of the 52 AFCON 2025 games; "
                       "61.6% across all 1,909 competitive internationals.")
            st.caption("Listed automatically: UEFA Nations League, plus the Africa Cup "
                       "of Nations, World Cup, Euros and Copa America finals as soon as "
                       "their fixtures are published. AFCON qualifiers and CAF club "
                       "competitions have no free fixture feed: use 'Any two national "
                       "teams' below for those.")
            if nl_up.empty:
                st.caption("None in the next 30 days.")
            elif phone_mode():
                render_match_cards(nl_up, tz)
            else:
                show_df(nl_up.drop(columns=["League", "Status", "Stake", "Kickoff UTC"])
                        .style.format({c: "{:.0f}" for c in
                                       ["Home %", "Draw %", "Away %", *GOAL_COLS, "BTTS %"]}),
                        hide_index=True)
            st.markdown("#### Any two national teams")
            st.caption(f"Ratings from {im['n_matches']:,} internationals over the last "
                       f"{INTL_DEFAULTS['years']} years (recent matches count more). "
                       f"Latest result in the data: {im['last_date']:%d %b %Y}.")
            teams_i = im["current_teams"]

            def _ix(name, fallback):
                return teams_i.index(name) if name in teams_i else fallback

            i1, i2, i3 = st.columns([2, 2, 1])
            t_home = i1.selectbox("Team 1 (home side if not neutral)", teams_i,
                                  index=_ix("England", 0))
            t_away = i2.selectbox("Team 2", teams_i, index=_ix("France", 1))
            venue = i3.radio("Venue", ["Neutral", "Team 1 at home"])
            if t_home == t_away:
                st.warning("Pick two different teams.")
            else:
                neutral = venue == "Neutral"
                lh_i, la_i, m_i, p_i = predict(im, t_home, t_away, neutral=neutral)
                ps_i, pp_i = predicted_score(m_i, p_i)
                n1, n2, n3, n4 = st.columns(4)
                n1.metric(f"{t_home} xG", f"{lh_i:.2f}")
                n2.metric(f"{t_away} xG", f"{la_i:.2f}")
                n3.metric("Predicted score", ps_i, delta=f"{pp_i*100:.0f}% chance",
                          delta_color="off")
                n4.metric("Over 2.5 goals", f"{p_i['O25']*100:.0f}%")
                o1, o2, o3 = st.columns(3)
                o1.metric(f"{t_home} win", f"{p_i['H']*100:.1f}%")
                o2.metric("Draw (90 min)", f"{p_i['D']*100:.1f}%")
                o3.metric(f"{t_away} win", f"{p_i['A']*100:.1f}%")

                st.markdown("#### Minimum odds worth taking")
                rows_i = []
                for k in ["H", "D", "A", "O15", "O25", "U25", "O35", "BTTS_Y", "BTTS_N"]:
                    pr = p_i[k]
                    label = (MARKET_LABELS[k].replace("Home Win", f"{t_home} win")
                             .replace("Away Win", f"{t_away} win"))
                    rows_i.append({"Market": label, "Model %": pr * 100,
                                   "Fair odds": 1 / pr,
                                   "Bet only at odds of at least": (1 / pr) * (1 + min_edge / 100)})
                show_df(pd.DataFrame(rows_i).style.format(
                    {"Model %": "{:.1f}", "Fair odds": "{:.2f}",
                     "Bet only at odds of at least": "{:.2f}"}), hide_index=True)
                st.caption("No free feed publishes international fixtures or their odds, "
                           "so compare these with your bookmaker yourself. In testing on "
                           "1,915 competitive internationals (Jul 2024 - Oct 2026) the "
                           "model picked the right result 61% of the time, but its "
                           "profit against bookmakers is untested - treat the minimum "
                           "odds as a filter, not a guarantee. Knockout games are "
                           "predicted for 90 minutes (extra time and penalties not included).")
                st.markdown("#### Most likely scores")
                show_df(pd.DataFrame([{"Score": sc, "Chance %": round(pr * 100, 1),
                                       "Fair odds": round(1 / pr, 1)}
                                      for sc, pr in top_scores(m_i, 6)]), hide_index=True)

            st.markdown("#### Strongest national teams right now")
            confed = nation_confederations(intl)
            region = st.selectbox("Region", ["All"] + [r for r, _ in CONFEDERATIONS]
                                  + ["Middle East"])
            teams_r = [t for t in teams_i if region == "All" or confed.get(t) == region
                       or (region == "Middle East" and t in MIDDLE_EAST)]
            rk = pd.DataFrame({
                "Team": teams_r,
                "Attack ×": [np.exp(im["attack"][t]) for t in teams_r],
                "Defence × (conceded, lower=better)": [np.exp(-im["defence"][t]) for t in teams_r],
            })
            rk["Net strength"] = rk["Attack ×"] / rk["Defence × (conceded, lower=better)"]
            rk = rk.sort_values("Net strength", ascending=False).head(30)
            rk.insert(0, "Rank", range(1, len(rk) + 1))
            show_df(rk.style.format({c: "{:.2f}" for c in rk.columns
                                     if c not in ("Team", "Rank")}), hide_index=True)
            with st.expander("Latest international results in the data"):
                last = intl.tail(20).iloc[::-1].copy()
                last["Score"] = (last["FTHG"].astype(int).astype(str) + "-"
                                 + last["FTAG"].astype(int).astype(str))
                last["Date"] = last["Date"].dt.strftime("%d %b %Y")
                show_df(last[["Date", "HomeTeam", "Score", "AwayTeam", "Tournament"]],
                        hide_index=True)

    # ================= TAB: System health =================
    with tab_health, safe_section("System Health"):
        st.markdown("### 🩺 System health")
        st.caption("Every data source the system depends on. ✅ fine · ⚠️ check · ❌ problem. "
                   "'backup copy' = read from the copy saved by the scheduled GitHub run "
                   "(fast start-up, or because the source was down).")
        show_df(health, hide_index=True, height=min(900, 38 + 35 * len(health)))
        st.markdown("#### 🛡️ Safety switch")
        if STATUS_OVERRIDES:
            for lg, (_, why) in STATUS_OVERRIDES.items():
                st.warning(f"{league_name(lg)}: {why}")
        else:
            st.success(f"No league downgraded. A BET league is automatically moved to WATCH "
                       f"once {SAFETY['min_bets']}+ paper bets show it is not beating the "
                       "closing odds (average CLV below zero) or its ROI is clearly negative.")
        for note in promotion_candidates(settled_all):
            st.info("📈 " + note)
        st.markdown("#### If something shows ❌")
        st.markdown(
            "- **A source is down:** the app keeps working from the backup copy; nothing to do.\n"
            "- **Results are weeks old in season:** the source has stopped updating that "
            "league; advice there uses older form, so be more careful.\n"
            "- **Backup copy is old:** open your GitHub repository → **Actions** → "
            "**Daily football predictions** and check the latest run (GitHub also emails "
            "you when a scheduled run fails).")

    st.markdown("---")
    st.caption("Data: Football-Data.co.uk (leagues, refreshed hourly), fixturedownload.com "
               "(UEFA competitions) and the open international_results dataset by "
               "martj42 (national teams). "
               "Statistical estimates only — no outcome is guaranteed. Bet responsibly.")


def run_headless_log(argv=None) -> int:
    """
    Headless mode for a daily scheduled task (no browser needed):
        python epl_value_dashboard.py --log
    Fetches every league, logs new value flags (bet / watch leagues), writes
    predictions.md (advice + predicted scores for all leagues) and prints a
    paper-trading summary.
    """
    ap = argparse.ArgumentParser(description="Log football value bets (paper trading).")
    ap.add_argument("--log", action="store_true")
    ap.add_argument("--book", default=DEFAULTS["book"], choices=list(BOOKMAKERS))
    ap.add_argument("--min-edge", type=float, default=DEFAULTS["min_edge"])
    ap.add_argument("--max-edge", type=float, default=DEFAULTS["max_edge"],
                    help="100 or more = no cap")
    ap.add_argument("--goals-pct", type=int, default=DEFAULTS["goals_pct"])
    ap.add_argument("--half-life", type=int, default=DEFAULTS["half_life"])
    ap.add_argument("--seasons-back", type=int, default=DEFAULTS["seasons_back"])
    ap.add_argument("--bankroll", type=float, default=DEFAULTS["bankroll"])
    ap.add_argument("--timezone", default=DEFAULTS["timezone"],
                    help="time zone for predictions.md (e.g. Africa/Douala)")
    ap.add_argument("--all-leagues-bet", action="store_true",
                    help="give BET advice in every league (not recommended)")
    ap.add_argument("--no-backup", action="store_true",
                    help="do not refresh the data_snapshot backup folder")
    args = ap.parse_args(argv)
    logging.getLogger("streamlit").setLevel(logging.ERROR)
    global PREFER_SNAPSHOT
    PREFER_SNAPSHOT = False                # the scheduled run always downloads fresh data

    fixtures = load_fixtures()
    window = upcoming_fixtures(fixtures)
    log = read_paper_log()
    # Every league (3 seasons) so the backup covers the whole app
    full = load_all_results(tuple(LEAGUES), 3)
    if not full:
        print("ERROR: could not download results from Football-Data.co.uk")
        return 1
    all_results = {lg: filter_seasons(df, args.seasons_back) for lg, df in full.items()}
    saved = []
    if not args.no_backup:
        for lg, df in full.items():
            if df.attrs.get("source") == "live" and save_snapshot(f"results_{lg}", df):
                saved.append(lg)
        if fixtures.attrs.get("source") == "live" and save_snapshot("fixtures", fixtures):
            saved.append("fixtures")
    schedule = load_schedule()
    if (not args.no_backup and schedule.attrs.get("source") == "live"
            and save_snapshot("schedule", schedule)):
        saved.append("schedule")
    fixtures = add_schedule(fixtures, schedule, all_results)
    window = upcoming_fixtures(fixtures)
    # Safety switch from paper-trading results
    STATUS_OVERRIDES.clear()
    if not log.empty:
        STATUS_OVERRIDES.update(safety_overrides(settle_paper_log(log, all_results)))
    for lg, (_, why) in STATUS_OVERRIDES.items():
        print(f"SAFETY: {league_name(lg)} - {why}")
    max_edge = np.inf if args.max_edge >= 100 else args.max_edge
    # Missing players (needs the APIFOOTBALL_KEY secret on a paid plan)
    afk = os.environ.get("APIFOOTBALL_KEY", "").strip()
    try:
        inj_status = refresh_injuries(afk)
    except Exception as exc:
        inj_status = {"result": f"failed: {type(exc).__name__}"}
    try:
        inj_status.update(refresh_book_odds(afk))
    except Exception as exc:
        inj_status["odds_result"] = f"failed: {type(exc).__name__}"
    if not args.no_backup:
        try:
            SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
            (SNAPSHOT_DIR / "injuries_status.json").write_text(
                json.dumps(inj_status, indent=1, default=str))
        except Exception:
            pass
    print(f"  missing players: {inj_status.get('result')}")
    print(f"  bookmaker prices: {inj_status.get('odds_result')}")
    inj_table = load_injury_table()
    book_odds = load_book_odds()
    adv, flags, models = weekly_overview(
        fixtures, all_results, args.half_life, args.goals_pct / 100, args.book,
        args.min_edge, max_edge, args.bankroll, args.all_leagues_bet, inj_table, book_odds)
    _, added, err = update_paper_log(flags, args.all_leagues_bet)
    cups = []
    uefa = intl_fx = pd.DataFrame()
    # Bookmaker odds for cups / national teams (needs the ODDS_API_KEY secret)
    try:
        odds_status = refresh_cup_odds(os.environ.get("ODDS_API_KEY", "").strip())
    except Exception as exc:
        odds_status = {"result": f"failed: {type(exc).__name__}"}
    if not args.no_backup:
        save_odds_status(odds_status)
    print(f"  cup odds: {odds_status.get('result')}"
          + (f" | credits left: {odds_status['credits_remaining']}"
             if odds_status.get("credits_remaining") else ""))
    cup_odds = load_cup_odds()
    # API-Football: one cheap check per day until line-ups/injuries are built in
    try:
        afk = os.environ.get("APIFOOTBALL_KEY", "").strip()
        f = SNAPSHOT_DIR / f"{APIFOOTBALL_STATUS}.json"
        prev = json.loads(f.read_text()) if f.exists() else {}
        last = pd.Timestamp(prev["time"].replace(" UTC", ""), tz="UTC") if prev.get("time") else None
        stale = last is None or (pd.Timestamp.now(tz="UTC") - last) > pd.Timedelta(hours=20)
        if (afk and (stale or not prev.get("key_present"))) or not prev:
            ap = probe_apifootball(afk)
            if not args.no_backup:
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_text(json.dumps(ap, indent=1, default=str))
            print(f"  API-Football: {ap.get('result')}")
    except Exception as exc:
        print(f"WARNING: API-Football check failed: {type(exc).__name__}")
    try:
        uefa = load_uefa()
        if not args.no_backup and uefa.attrs.get("source") == "live" and save_snapshot("uefa", uefa):
            saved.append("uefa")
        if not uefa.empty:
            all_models = league_models(all_results, args.half_life, args.goals_pct / 100)
            teams = sorted(set(uefa["HomeTeam"]) | set(uefa["AwayTeam"]))
            cups.append(cup_predictions(uefa, fit_uefa(uefa, uefa_domestic(teams, all_models)),
                                        "Comp", 7, args.min_edge, cup_odds, inj_table, book_odds))
        intl = load_international()
        intl_fx = load_intl_fixtures()
        if not args.no_backup:
            if intl.attrs.get("source") == "live" and save_snapshot("international", intl):
                saved.append("international")
            if intl_fx.attrs.get("source") == "live" and save_snapshot("intl_fixtures", intl_fx):
                saved.append("intl_fixtures")
        if not intl.empty:
            cups.append(cup_predictions(intl_fx, fit_international(intl),
                                        "Comp", 7, args.min_edge, cup_odds, inj_table, book_odds))
    except Exception as exc:
        print(f"WARNING: UEFA / national-team predictions failed: {exc}")
    adv = combine_overview(adv, *cups)
    record = ["(track record unavailable this run)"]
    settled_p = None
    try:
        _, n_picks, perr = update_pick_log(adv)
        if perr:
            print(f"WARNING: could not write {PICK_LOG}: {perr}")
        settled_p = settle_pick_log(read_pick_log(), result_frames(full, uefa, intl_fx))
        sm_p = pick_summary(settled_p)
        record = pick_record_lines(sm_p)
        print(f"  best picks: {n_picks} new logged | " + " ".join(record).replace("**", ""))
    except Exception as exc:
        print(f"WARNING: best-pick log failed: {exc}")
    try:
        write_predictions_md(adv, flags, PREDICTIONS_MD, args.bankroll,
                             args.all_leagues_bet, args.timezone, record, settled_p)
    except Exception as exc:
        print(f"WARNING: could not write {PREDICTIONS_MD}: {exc}")
    if err:
        print(f"ERROR writing {PAPER_LOG}: {err}")
        return 1
    sm = paper_summary(settle_paper_log(read_paper_log(), all_results))
    n_bet = int(adv["Advice"].str.startswith("BET").sum()) if not adv.empty else 0
    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M UTC")
    n_cup = int(adv["League"].isin(list(UEFA_COMPS) + [NATIONS_LEAGUE[0]]).sum()) \
        if not adv.empty else 0
    print(f"[{stamp}] leagues: {len(models)} | fixtures: {len(window)} | cup/intl: {n_cup} | "
          f"BET advice: {n_bet} | new flags logged: {added}")
    if AF_FILL_STATUS and not args.no_backup:
        try:
            (SNAPSHOT_DIR / "results_fill_status.json").write_text(json.dumps(
                {"time": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M UTC"),
                 "leagues": AF_FILL_STATUS}, indent=1, default=str))
        except Exception:
            pass
    print(f"  backup refreshed: {len(saved)} sources"
          + ("" if len(saved) >= len(LEAGUES) else
             " (some sources unavailable - the app falls back to the previous copy)"))
    print(f"  logged {sm['logged']} | settled {sm['settled']} | "
          f"profit {sm['profit']:+.2f}u | ROI "
          + ("n/a" if np.isnan(sm["roi"]) else f"{sm['roi']:+.1f}%")
          + " | avg CLV " + ("n/a" if np.isnan(sm["clv"]) else f"{sm['clv']:+.2f}%"))
    return 0



if __name__ == "__main__":
    if "--log" in sys.argv:
        sys.exit(run_headless_log(sys.argv[1:]))
    main()
