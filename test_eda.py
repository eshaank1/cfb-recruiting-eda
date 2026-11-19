"""
Tests for data_prep.py and the helpers in eda.py.

Each test builds a tiny DataFrame by hand where the right answer can be
checked by eye. Run with either `python test_eda.py` or `pytest`.
"""

import math

import numpy as np
import pandas as pd

import data_prep as prep
import eda


def make_games():
    """Four hand-made games: one tie, one incomplete, one null score."""
    return pd.DataFrame({
        "season": [2020, 2020, 2020, 2020],
        "seasonType": ["regular"] * 4,
        "completed": [True, True, False, True],
        "homeTeam": ["Utah", "Utah", "Utah", "Utah"],
        "awayTeam": ["Oregon", "Idaho", "Boise", "Cal"],
        "homePoints": [10, 7, 30, np.nan],
        "awayPoints": [3, 7, 0, 14],
    })


def test_normalize_team_name():
    assert prep.normalize_team_name("San José State") == "san jose state"
    assert prep.normalize_team_name("  Texas   A&M ") == "texas a and m"
    assert prep.normalize_team_name("Hawai'i") == "hawaii"
    assert prep.normalize_team_name("UConn") == "connecticut"
    assert prep.normalize_team_name(None) is None
    assert prep.normalize_team_name(float("nan")) is None


def test_team_game_results_drops_bad_rows_and_scores_ties():
    results = prep.team_game_results(make_games())
    # Only the win and the tie survive; each makes two team rows.
    assert len(results) == 4
    utah = results[results["team"] == "Utah"].sort_values("points_for")
    assert list(utah["win"]) == [0.5, 1.0]
    oregon = results[results["team"] == "Oregon"]
    assert list(oregon["win"]) == [0.0]


def make_results(n_games, n_wins, team="A", season=2020):
    """Builds game results for one team with a chosen record."""
    wins = [1.0] * n_wins + [0.0] * (n_games - n_wins)
    return pd.DataFrame({"season": season, "team": team,
                         "team_key": team.lower(), "win": wins})


def test_compute_win_pct_and_min_games():
    results = pd.concat([make_results(10, 7, "A"), make_results(7, 7, "B")])
    out = prep.compute_win_pct(results, min_games=8)
    assert list(out["team_key"]) == ["a"]
    assert math.isclose(out.loc[0, "win_pct"], 0.7)
    assert out.loc[0, "games"] == 10


def test_compute_win_pct_restricts_to_fbs_by_season():
    results = pd.concat([make_results(10, 5, "A", 2020),
                         make_results(10, 5, "A", 2021)])
    fbs = pd.DataFrame({"team_key": ["a"], "season": [2021],
                        "conference": ["X"]})
    out = prep.compute_win_pct(results, fbs)
    assert list(out["season"]) == [2021]
    assert list(out["conference"]) == ["X"]


def test_add_prior_win_pct_uses_previous_season_only():
    win = pd.DataFrame({"team_key": ["a", "a", "a", "b"],
                        "season": [2018, 2019, 2021, 2021],
                        "win_pct": [0.5, 0.75, 0.25, 0.9]})
    out = prep.add_prior_win_pct(win).set_index(["team_key", "season"])
    assert math.isclose(out.loc[("a", 2019), "prior_win_pct"], 0.5)
    assert math.isclose(out.loc[("a", 2019), "delta_win"], 0.25)
    # 2020 is missing for team a, so 2021 has no prior value.
    assert math.isnan(out.loc[("a", 2021), "prior_win_pct"])
    assert math.isnan(out.loc[("b", 2021), "delta_win"])


def test_compute_talent_window_edges():
    rec = pd.DataFrame({"team": ["A"] * 5, "year": [2011, 2012, 2013,
                                                    2014, 2015],
                        "points": [100.0, 200.0, 300.0, 400.0, 500.0]})
    out = prep.compute_talent(rec).set_index("season")["talent_4yr"]
    assert 2013 not in out.index          # only 3 classes available
    assert math.isclose(out[2014], 250.0)  # mean of 2011-2014
    assert math.isclose(out[2015], 350.0)  # mean of 2012-2015


def test_compute_talent_missing_class_is_not_zero():
    rec = pd.DataFrame({"team": ["A", "A", "A"], "year": [2011, 2012, 2014],
                        "points": [100.0, 100.0, 100.0]})
    out = prep.compute_talent(rec)
    assert out.empty  # 2013 class missing, so no full 4-class window
    relaxed = prep.compute_talent(rec, min_classes=3)
    assert math.isclose(relaxed.iloc[0]["talent_4yr"], 100.0)


def test_prepare_returning_renames_columns():
    ret = pd.DataFrame({"team": ["Utah"], "season": [2022],
                        "percentPPA": [0.6], "percentPassingPPA": [0.5]})
    out = prep.prepare_returning(ret)
    assert list(out.columns) == ["team_key", "season", "ret_ppa",
                                 "ret_pass"]
    assert math.isclose(out.loc[0, "ret_ppa"], 0.6)


def make_portal():
    """Three transfers: A->B (rated), A->B (unrated), B->A (rated)."""
    return pd.DataFrame({
        "season": [2022, 2022, 2022],
        "origin": ["A", "A", "B"],
        "destination": ["B", "B", "A"],
        "rating": [0.9, np.nan, 0.5],
    })


def test_compute_portal_net_signs_and_missing_ratings():
    out = prep.compute_portal_net(make_portal()).set_index("team_key")
    assert math.isclose(out.loc["b", "net_portal_talent"], 0.9 - 0.5)
    assert math.isclose(out.loc["a", "net_portal_talent"], 0.5 - 0.9)
    assert out.loc["b", "n_in"] == 2
    assert out.loc["b", "rated_in"] == 1
    assert math.isclose(out.loc["b", "share_rated_in"], 0.5)


def test_build_table_portal_nan_before_2021_zero_after():
    win = pd.DataFrame({"team_key": ["a", "a"], "season": [2020, 2022],
                        "win_pct": [0.5, 0.6],
                        "prior_win_pct": [0.4, np.nan],
                        "delta_win": [0.1, np.nan]})
    talent = pd.DataFrame({"team_key": ["a"], "season": [2020],
                           "talent_4yr": [300.0]})
    returning = pd.DataFrame({"team_key": ["a"], "season": [2022],
                              "ret_ppa": [0.7]})
    portal_net = prep.compute_portal_net(make_portal())
    out = prep.build_team_season_table(win, talent, returning, portal_net)
    out = out.set_index("season")
    assert math.isnan(out.loc[2020, "net_portal_talent"])
    assert not out.loc[2020, "post_portal"]
    assert out.loc[2022, "post_portal"]
    assert math.isclose(out.loc[2022, "ret_ppa"], 0.7)
    assert len(out) == 2  # merges never add or drop win rows


def test_unmatched_keys():
    base = pd.DataFrame({"team_key": ["a", "b"]})
    other = pd.DataFrame({"team_key": ["b", "c", None]})
    assert prep.unmatched_keys(base, other) == ["c"]


def test_missing_report_and_seven_number_summary():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, np.nan], "y": [1, 2, 3, 4]})
    report = eda.missing_report(df)
    assert report.loc["x", "n_missing"] == 1
    assert math.isclose(report.loc["x", "pct_missing"], 25.0)
    summary = eda.seven_number_summary(df, ["y"])
    assert math.isclose(summary.loc["y", "mean"], 2.5)
    assert summary.loc["y", "min"] == 1 and summary.loc["y", "max"] == 4
    assert list(summary.columns) == eda.SUMMARY_ROWS


def test_assumption_checks_runs_on_synthetic_data():
    rng = np.random.default_rng(0)
    n = 200
    data = pd.DataFrame({"team_key": rng.integers(0, 20, n).astype(str),
                         "x": rng.normal(size=n)})
    data["y"] = 2 * data["x"] + rng.normal(size=n)
    res, fit = eda.assumption_checks(data, "y ~ x")
    assert res["n_clusters"] <= 20
    assert abs(fit.params["x"] - 2) < 0.3
    assert 0 <= res["breusch_pagan_p"] <= 1
    assert "x" in res["vif"]


def main():
    """Runs every test_ function in this file without pytest."""
    tests = [(name, func) for name, func in globals().items()
             if name.startswith("test_") and callable(func)]
    for name, func in tests:
        func()
        print("passed", name)
    print("All {} tests passed.".format(len(tests)))


if __name__ == "__main__":
    main()
