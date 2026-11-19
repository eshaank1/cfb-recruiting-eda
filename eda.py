"""
Exploratory data analysis for the FBS recruiting / returning production /
transfer portal project.

Usage:
    python eda.py [--data-dir data] [--out-dir output]

Reads the CSVs written by collect_data.py, builds the merged team-season
table, and writes tables (output/tables), figures (output/figures), and
assumption checks for the three regression models (output/validity.txt).
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402
from scipy import stats  # noqa: E402
from statsmodels.graphics.gofplots import qqplot  # noqa: E402
from statsmodels.stats.diagnostic import het_breuschpagan  # noqa: E402
from statsmodels.stats.outliers_influence import (  # noqa: E402
    variance_inflation_factor)

import data_prep as prep  # noqa: E402

SUMMARY_ROWS = ["mean", "std", "min", "25%", "50%", "75%", "max"]

# name -> (formula, columns used, null hypothesis, seasons or None)
MODELS = {
    "RQ1": ("win_pct ~ prior_win_pct + talent_4yr",
            ["win_pct", "prior_win_pct", "talent_4yr"],
            "H0: talent_4yr coefficient = 0 given prior_win_pct", None),
    "RQ2": ("delta_win ~ ret_ppa + prior_win_pct + talent_4yr",
            ["delta_win", "ret_ppa", "prior_win_pct", "talent_4yr"],
            "H0: ret_ppa coefficient = 0 given prior_win_pct, talent_4yr",
            None),
    "RQ3": ("delta_win ~ net_portal_talent + ret_ppa + talent_4yr"
            " + prior_win_pct",
            ["delta_win", "net_portal_talent", "ret_ppa", "talent_4yr",
             "prior_win_pct"],
            "H0: net_portal_talent coefficient = 0 given the controls",
            (prep.PORTAL_FIRST_SEASON, 9999)),
}


# ----------------------------------------------------------------- loading
def load_data(data_dir):
    """
    Reads the five CSVs written by collect_data.py.

    Args:
        data_dir: folder containing games.csv, recruiting.csv,
            returning.csv, portal.csv, and fbs_teams.csv.

    Returns:
        Dictionary mapping dataset name to DataFrame.
    """
    names = ["games", "recruiting", "returning", "portal", "fbs_teams"]
    return {n: pd.read_csv(os.path.join(data_dir, n + ".csv"))
            for n in names}


def build_table(raw):
    """
    Runs the full cleaning pipeline on the raw datasets.

    Args:
        raw: output of load_data.

    Returns:
        Tuple (merged table, FBS table, talent table, portal table,
        returning table, win table).
    """
    fbs = prep.make_fbs_table(raw["fbs_teams"])
    results = prep.team_game_results(raw["games"])
    win = prep.add_prior_win_pct(prep.compute_win_pct(results, fbs))
    talent = prep.compute_talent(raw["recruiting"])
    returning = prep.prepare_returning(raw["returning"])
    portal = prep.compute_portal_net(raw["portal"])
    merged = prep.build_team_season_table(win, talent, returning, portal)
    return merged, fbs, talent, portal, returning, win


# --------------------------------------------------------------- summaries
def missing_report(df):
    """
    Counts missing values in every column.

    Args:
        df: any DataFrame.

    Returns:
        DataFrame with n_missing and pct_missing per column.
    """
    counts = df.isna().sum()
    return pd.DataFrame({"n_missing": counts,
                         "pct_missing": 100 * counts / max(len(df), 1)})


def seven_number_summary(df, columns):
    """
    Mean, std, min, Q1, median, Q3, and max for numeric columns.

    Args:
        df: DataFrame.
        columns: numeric column names to summarize.

    Returns:
        DataFrame with one row per column and the seven statistics.
    """
    return df[columns].describe().loc[SUMMARY_ROWS].T


def describe_dataset(name, df, numeric, categorical, out_dir):
    """
    Writes size, missingness, numeric and categorical summaries to a file.

    Args:
        name: dataset label used in the file name.
        df: the DataFrame to describe.
        numeric: numeric columns for the seven-number summary.
        categorical: columns whose unique values are counted.
        out_dir: folder for the text file.
    """
    lines = ["{}: {} rows x {} columns".format(name, *df.shape), "",
             "Missing values:", missing_report(df).to_string(), "",
             "Seven-number summary:",
             seven_number_summary(df, numeric).to_string()]
    for col in categorical:
        lines += ["", "Value counts for " + col + ":",
                  df[col].value_counts(dropna=False).to_string()]
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name + "_summary.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[:2]))


# ----------------------------------------------------------------- figures
def save_figure(fig, fig_dir, name):
    """
    Saves a figure as a PNG and closes it.

    Args:
        fig: matplotlib figure.
        fig_dir: output folder (created if needed).
        name: file name without extension.
    """
    os.makedirs(fig_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, name + ".png"), dpi=200)
    plt.close(fig)


def hist_plot(df, column, title, xlabel, fig_dir, name):
    """Saves a histogram of one column."""
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(df[column].dropna(), bins=30, ax=ax)
    ax.set(title=title, xlabel=xlabel, ylabel="Number of team-seasons")
    save_figure(fig, fig_dir, name)


def box_by_season(df, column, title, ylabel, fig_dir, name):
    """Saves a box plot of one column for each season."""
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(data=df, x="season", y=column, ax=ax)
    ax.set(title=title, xlabel="Season", ylabel=ylabel)
    ax.tick_params(axis="x", rotation=45)
    save_figure(fig, fig_dir, name)


def scatter_fit(df, x, y, title, fig_dir, name, hue=None):
    """Saves a scatterplot of y vs x with a fitted line."""
    d = df.dropna(subset=[x, y])
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(data=d, x=x, y=y, hue=hue, alpha=0.5, ax=ax)
    sns.regplot(data=d, x=x, y=y, scatter=False, color="black", ax=ax)
    ax.set_title(title)
    save_figure(fig, fig_dir, name)


def make_figures(merged, fbs, raw, fig_dir):
    """
    Creates the EDA figures (at least two per dataset).

    Args:
        merged: merged team-season table.
        fbs: FBS table.
        raw: raw datasets from load_data.
        fig_dir: folder for the PNG files.
    """
    hist_plot(merged, "win_pct", "Distribution of win percentage",
              "Regular-season win percentage", fig_dir, "win_pct_hist")
    box_by_season(merged, "win_pct", "Win percentage by season",
                  "Win percentage", fig_dir, "win_pct_by_season")

    rec = (raw["recruiting"].drop(columns="season", errors="ignore")
           .rename(columns={"year": "season"}))
    box_by_season(rec, "points", "Recruiting class rating by season",
                  "Class rating points", fig_dir, "recruiting_by_season")
    hist_plot(merged, "talent_4yr", "Four-year average recruiting rating",
              "talent_4yr (mean class points)", fig_dir, "talent_hist")

    hist_plot(merged, "ret_ppa", "Share of production returning",
              "ret_ppa", fig_dir, "ret_ppa_hist")
    scatter_fit(merged, "ret_ppa", "delta_win",
                "Returning production vs. change in win percentage",
                fig_dir, "ret_vs_delta")

    per_year = raw["portal"].groupby("season").size().reset_index(name="n")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=per_year, x="season", y="n", ax=ax)
    ax.set(title="Portal transfers per season", xlabel="Season",
           ylabel="Number of transfers")
    save_figure(fig, fig_dir, "portal_per_year")
    share = raw["portal"].assign(
        missing_rating=raw["portal"]["rating"].isna())
    share = share.groupby("season")["missing_rating"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=share, x="season", y="missing_rating", ax=ax)
    ax.set(title="Share of portal transfers with no rating",
           xlabel="Season", ylabel="Share missing")
    save_figure(fig, fig_dir, "portal_missing_rating")

    counts = fbs.groupby("conference").size().sort_values().reset_index()
    counts.columns = ["conference", "team_seasons"]
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.barplot(data=counts, y="conference", x="team_seasons", ax=ax)
    ax.set(title="FBS team-seasons by conference", xlabel="Team-seasons",
           ylabel="")
    save_figure(fig, fig_dir, "fbs_conferences")
    top = fbs.groupby("season").size().reset_index(name="n_fbs")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.lineplot(data=top, x="season", y="n_fbs", marker="o", ax=ax)
    ax.set(title="Number of FBS teams per season", xlabel="Season",
           ylabel="FBS teams")
    save_figure(fig, fig_dir, "fbs_count")

    scatter_fit(merged, "talent_4yr", "win_pct",
                "Recruiting rating vs. win percentage", fig_dir,
                "talent_vs_win")
    scatter_fit(merged, "prior_win_pct", "delta_win",
                "Prior win percentage vs. change in win percentage",
                fig_dir, "prior_vs_delta")
    scatter_fit(merged[merged["post_portal"]], "net_portal_talent",
                "delta_win", "Net portal talent vs. change in win pct.",
                fig_dir, "portal_vs_delta")
    cols = ["win_pct", "prior_win_pct", "delta_win", "talent_4yr",
            "ret_ppa", "net_portal_talent"]
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(merged[cols].corr(), annot=True, fmt=".2f",
                cmap="vlag", center=0, ax=ax)
    ax.set_title("Correlations among variables of interest")
    save_figure(fig, fig_dir, "correlation_heatmap")


# ------------------------------------------------------------ validity
def model_data(merged, columns, seasons):
    """
    Selects complete rows (and a season range) for one regression.

    Args:
        merged: merged team-season table.
        columns: columns that must be non-missing.
        seasons: None or (first, last) season, inclusive.

    Returns:
        The filtered DataFrame.
    """
    d = merged
    if seasons is not None:
        d = d[(d["season"] >= seasons[0]) & (d["season"] <= seasons[1])]
    return d.dropna(subset=columns).reset_index(drop=True)


def assumption_checks(data, formula):
    """
    Checks the assumptions behind an OLS fit with team-clustered errors.

    Args:
        data: complete-case DataFrame with a team_key column.
        formula: statsmodels formula.

    Returns:
        Tuple (results dictionary, fitted OLS model).
    """
    fit = smf.ols(formula, data=data).fit()
    exog = fit.model.exog
    names = fit.model.exog_names
    vif = {n: variance_inflation_factor(exog, i)
           for i, n in enumerate(names) if n != "Intercept"}
    bp_p = het_breuschpagan(fit.resid, exog)[1]
    jb_p = stats.jarque_bera(fit.resid).pvalue
    prev = fit.resid.groupby(data["team_key"]).shift(1)
    lag_corr = fit.resid.corr(prev)
    clustered = smf.ols(formula, data=data).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(
            data["team_key"])[0]})
    results = {"n_obs": int(fit.nobs),
               "n_clusters": data["team_key"].nunique(),
               "breusch_pagan_p": bp_p, "jarque_bera_p": jb_p,
               "resid_lag1_corr_within_team": lag_corr,
               "vif": vif, "r_squared": fit.rsquared,
               "clustered_summary": clustered.summary().as_text()}
    return results, fit


def run_validity(merged, out_dir):
    """
    Runs assumption checks and saves residual and QQ plots per model.

    Args:
        merged: merged team-season table.
        out_dir: output folder; validity.txt and figures are written here.
    """
    fig_dir = os.path.join(out_dir, "figures")
    lines = []
    for label, (formula, cols, null, seasons) in MODELS.items():
        data = model_data(merged, cols + ["team_key"], seasons)
        res, fit = assumption_checks(data, formula)
        lines += ["=" * 60, label + ": " + formula, null,
                  "Observations: {n_obs}, teams (clusters): "
                  "{n_clusters}".format(**res),
                  "Breusch-Pagan p (small => heteroskedastic): "
                  "{:.4f}".format(res["breusch_pagan_p"]),
                  "Jarque-Bera p (small => non-normal residuals): "
                  "{:.4f}".format(res["jarque_bera_p"]),
                  "Residual lag-1 correlation within team: "
                  "{:.3f}".format(res["resid_lag1_corr_within_team"]),
                  "VIF: " + ", ".join("{}={:.2f}".format(k, v)
                                      for k, v in res["vif"].items()),
                  "", "Team-clustered OLS:", res["clustered_summary"], ""]
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(fit.fittedvalues, fit.resid, alpha=0.4)
        ax.axhline(0, color="black")
        ax.set(title=label + ": residuals vs. fitted",
               xlabel="Fitted value", ylabel="Residual")
        save_figure(fig, fig_dir, label.lower() + "_resid_vs_fitted")
        fig = qqplot(fit.resid, line="s")
        fig.axes[0].set_title(label + ": residual QQ plot")
        save_figure(fig, fig_dir, label.lower() + "_qq")
    with open(os.path.join(out_dir, "validity.txt"), "w") as f:
        f.write("\n".join(lines))


# -------------------------------------------------------------------- main
def main():
    """Runs the full EDA and writes all outputs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="output")
    args = parser.parse_args()
    table_dir = os.path.join(args.out_dir, "tables")
    fig_dir = os.path.join(args.out_dir, "figures")

    raw = load_data(args.data_dir)
    merged, fbs, talent, portal, returning, win = build_table(raw)

    describe_dataset("games_raw", raw["games"],
                     ["homePoints", "awayPoints"], ["seasonType"],
                     table_dir)
    describe_dataset("win_pct", win, ["wins", "games", "win_pct"],
                     ["conference"], table_dir)
    describe_dataset("recruiting", raw["recruiting"], ["points"], [],
                     table_dir)
    describe_dataset("returning", returning,
                     [c for c in returning.columns if c.startswith("ret_")],
                     [], table_dir)
    describe_dataset("portal_raw", raw["portal"], ["rating"],
                     ["season"], table_dir)
    describe_dataset("fbs_teams", fbs, ["season"], ["conference"],
                     table_dir)
    describe_dataset("merged", merged,
                     ["win_pct", "prior_win_pct", "delta_win", "talent_4yr",
                      "ret_ppa", "net_portal_talent"],
                     ["post_portal"], table_dir)

    for name, df in (("recruiting", talent), ("returning", returning),
                     ("portal", portal)):
        extra = prep.unmatched_keys(fbs, df)
        print("{}: {} team keys not in FBS list".format(name, len(extra)))
    merged.to_csv(os.path.join(args.out_dir, "merged_team_seasons.csv"),
                  index=False)

    make_figures(merged, fbs, raw, fig_dir)
    run_validity(merged, args.out_dir)


if __name__ == "__main__":
    main()
