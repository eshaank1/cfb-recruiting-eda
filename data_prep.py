"""
Cleaning and feature-building functions for the FBS football EDA.

Every function takes plain DataFrames and returns a new DataFrame, so each
one can be tested on a small hand-built table (see test_eda.py).
Teams are matched across datasets with a normalized `team_key` column.
"""

import unicodedata

import pandas as pd

PORTAL_FIRST_SEASON = 2021
MIN_GAMES = 8

# Extend this using the unmatched-name report printed by eda.py.
TEAM_ALIASES = {
    "uconn": "connecticut",
    "umass": "massachusetts",
}


def normalize_team_name(name):
    """
    Converts a team name to a comparable lowercase key.

    Accents, periods, apostrophes, and repeated spaces are removed and "&"
    becomes "and". Known aliases are then mapped to one spelling.

    Args:
        name: a team name, e.g. "San José State".

    Returns:
        The key string (e.g. "san jose state"), or None if name is not text.
    """
    if not isinstance(name, str):
        return None
    text = unicodedata.normalize("NFKD", name)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.replace("&", " and ").replace(".", "").replace("'", "")
    text = " ".join(text.split()).casefold()
    return TEAM_ALIASES.get(text, text)


def add_team_key(df, column):
    """
    Returns a copy of df with a `team_key` column built from `column`.

    Args:
        df: any DataFrame.
        column: name of the column holding team names.

    Returns:
        A copy of df with the extra `team_key` column.
    """
    out = df.copy()
    out["team_key"] = out[column].map(normalize_team_name)
    return out


def make_fbs_table(fbs_teams):
    """
    Builds the list of FBS team-seasons with their conference.

    Args:
        fbs_teams: rows with `school`, `season`, and `conference`.

    Returns:
        DataFrame with team, team_key, season, conference (no duplicates).
    """
    out = add_team_key(fbs_teams, "school")
    out = out.rename(columns={"school": "team"})
    keep = ["team", "team_key", "season", "conference"]
    return out[keep].drop_duplicates(["team_key", "season"])


def team_game_results(games):
    """
    Turns completed regular-season games into one row per team per game.

    Games that are not completed or that have a missing score are dropped.
    A win counts 1, a tie 0.5, and a loss 0.

    Args:
        games: rows with season, homeTeam, homePoints, awayTeam, awayPoints,
            and optionally `completed` and `seasonType`.

    Returns:
        DataFrame with season, team, team_key, points_for, points_against,
        and win.
    """
    df = games.copy()
    if "completed" in df.columns:
        df = df[df["completed"].fillna(False).astype(bool)]
    if "seasonType" in df.columns:
        df = df[df["seasonType"] == "regular"]
    df = df.dropna(subset=["homeTeam", "awayTeam",
                           "homePoints", "awayPoints"])
    home = pd.DataFrame({"season": df["season"], "team": df["homeTeam"],
                         "points_for": df["homePoints"],
                         "points_against": df["awayPoints"]})
    away = pd.DataFrame({"season": df["season"], "team": df["awayTeam"],
                         "points_for": df["awayPoints"],
                         "points_against": df["homePoints"]})
    out = pd.concat([home, away], ignore_index=True)
    out["win"] = ((out["points_for"] > out["points_against"]).astype(float)
                  + 0.5 * (out["points_for"] == out["points_against"]))
    return add_team_key(out, "team")


def compute_win_pct(results, fbs=None, min_games=MIN_GAMES):
    """
    Computes each team-season's regular-season win percentage.

    Args:
        results: output of team_game_results.
        fbs: optional output of make_fbs_table. If given, only team-seasons
            that were FBS members that season are kept.
        min_games: team-seasons with fewer completed games are dropped.

    Returns:
        DataFrame with team_key, season, team, wins, games, win_pct.
    """
    grouped = results.groupby(["team_key", "season"])
    out = grouped.agg(team=("team", "first"), wins=("win", "sum"),
                      games=("win", "size")).reset_index()
    out = out[out["games"] >= min_games]
    if fbs is not None:
        out = out.merge(fbs[["team_key", "season", "conference"]],
                        on=["team_key", "season"], how="inner")
    out["win_pct"] = out["wins"] / out["games"]
    return out.reset_index(drop=True)


def add_prior_win_pct(win):
    """
    Adds last season's win percentage and the year-over-year change.

    A team-season gets NaN if the team has no row for the previous season.

    Args:
        win: output of compute_win_pct.

    Returns:
        Copy of win with prior_win_pct and delta_win columns.
    """
    prior = win[["team_key", "season", "win_pct"]].copy()
    prior["season"] = prior["season"] + 1
    prior = prior.rename(columns={"win_pct": "prior_win_pct"})
    out = win.merge(prior, on=["team_key", "season"], how="left")
    out["delta_win"] = out["win_pct"] - out["prior_win_pct"]
    return out


def compute_talent(recruiting, window=4, min_classes=4):
    """
    Computes the mean recruiting class rating over a trailing window.

    For season S the window covers classes S-window+1 through S. A missing
    class year counts as missing, not zero.

    Args:
        recruiting: rows with `year`, `team`, and `points`.
        window: number of classes to average (4 in the main analysis).
        min_classes: classes required inside the window to get a value.

    Returns:
        DataFrame with team_key, season, talent_4yr (rows with too few
        classes are omitted). The column keeps the name talent_4yr even
        when window is not 4.
    """
    df = add_team_key(recruiting, "team")
    wide = df.pivot_table(index="team_key", columns="year",
                          values="points", aggfunc="mean")
    years = range(int(wide.columns.min()), int(wide.columns.max()) + 1)
    wide = wide.reindex(columns=list(years))
    rolled = wide.T.rolling(window, min_periods=min_classes).mean().T
    rolled.columns.name = "season"
    out = rolled.reset_index().melt(id_vars="team_key", var_name="season",
                                    value_name="talent_4yr")
    return out.dropna().reset_index(drop=True)


def prepare_returning(returning):
    """
    Selects and renames the returning-production columns.

    Args:
        returning: rows with team, season, and percentPPA (plus optional
            percentPassingPPA, percentRushingPPA, percentReceivingPPA).

    Returns:
        DataFrame with team_key, season, ret_ppa and any component columns.
    """
    df = add_team_key(returning, "team")
    names = {"percentPPA": "ret_ppa", "percentPassingPPA": "ret_pass",
             "percentRushingPPA": "ret_rush",
             "percentReceivingPPA": "ret_rec"}
    present = {old: new for old, new in names.items() if old in df.columns}
    out = df.rename(columns=present)
    keep = ["team_key", "season"] + list(present.values())
    return out[keep].drop_duplicates(["team_key", "season"])


def compute_portal_net(portal):
    """
    Summarizes transfer-portal activity per team and season.

    Ratings that are missing are skipped in the sums but counted, so the
    share of rated transfers can be reported.

    Args:
        portal: rows with season, origin, destination, and rating.

    Returns:
        DataFrame with team_key, season, n_in, n_out, rated_in, rated_out,
        rating_in, rating_out, net_portal_talent, and share_rated_in.
    """
    df = portal.copy()
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    sides = []
    for column, tag in (("destination", "in"), ("origin", "out")):
        side = add_team_key(df, column).dropna(subset=["team_key"])
        grouped = side.groupby(["team_key", "season"])
        agg = grouped.agg(**{"n_" + tag: ("rating", "size"),
                             "rated_" + tag: ("rating", "count"),
                             "rating_" + tag: ("rating", "sum")})
        sides.append(agg.reset_index())
    out = sides[0].merge(sides[1], on=["team_key", "season"], how="outer")
    count_cols = [c for c in out.columns if c not in ("team_key", "season")]
    out[count_cols] = out[count_cols].fillna(0)
    out["net_portal_talent"] = out["rating_in"] - out["rating_out"]
    rated_share = out["rated_in"] / out["n_in"].where(out["n_in"] > 0)
    out["share_rated_in"] = rated_share
    return out


def build_team_season_table(win, talent, returning, portal_net):
    """
    Merges all features into one table with one row per FBS team-season.

    Portal columns are NaN before PORTAL_FIRST_SEASON (no data), and 0 for
    teams with no transfers in later seasons.

    Args:
        win: output of add_prior_win_pct.
        talent: output of compute_talent.
        returning: output of prepare_returning.
        portal_net: output of compute_portal_net.

    Returns:
        The merged DataFrame, with a boolean post_portal column.
    """
    keys = ["team_key", "season"]
    out = win.merge(talent, on=keys, how="left")
    out = out.merge(returning, on=keys, how="left")
    out = out.merge(portal_net, on=keys, how="left")
    portal_cols = [c for c in portal_net.columns if c not in keys]
    in_portal_era = out["season"] >= PORTAL_FIRST_SEASON
    for col in portal_cols:
        if col == "share_rated_in":
            continue
        out.loc[in_portal_era, col] = out.loc[in_portal_era, col].fillna(0)
    out["post_portal"] = in_portal_era
    return out


def unmatched_keys(base, other):
    """
    Lists team keys that appear in `other` but not in `base`.

    Args:
        base: DataFrame with a team_key column (e.g., the FBS table).
        other: DataFrame with a team_key column to check.

    Returns:
        Sorted list of unmatched keys.
    """
    missing = set(other["team_key"].dropna()) - set(base["team_key"])
    return sorted(missing)
