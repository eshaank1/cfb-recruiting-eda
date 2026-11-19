# cfb-recruiting-eda

Exploratory data analysis of FBS college football: how much recruiting
talent, returning production, and the transfer portal explain a team's
winning percentage from season to season.

**Authors:** Eshaan Kumar and Monte Horwitz

**CSE 163 Final Project, Fall 2026**

## Research questions

| | Model | Null hypothesis |
|---|---|---|
| RQ1 | `win_pct ~ prior_win_pct + talent_4yr` | 4-year recruiting talent has no effect once prior win % is known |
| RQ2 | `delta_win ~ ret_ppa + prior_win_pct + talent_4yr` | Returning production (PPA) has no effect on the change in win % |
| RQ3 | `delta_win ~ net_portal_talent + ret_ppa + talent_4yr + prior_win_pct` | Net transfer-portal talent has no effect (2021 and later seasons) |

## Data

All data comes from the [College Football Data API](https://collegefootballdata.com):

| Dataset | Endpoint | Seasons |
|---|---|---|
| Games (regular season) | `/games` | 2013–2025 |
| Team recruiting rankings | `/recruiting/teams` | 2011–2025 |
| Returning production | `/player/returning` | 2014–2025 |
| Transfer portal | `/player/portal` | 2021–2025 |
| FBS teams | `/teams/fbs` | 2013–2025 |

## Files

| File | Purpose |
|---|---|
| `collect_data.py` | Downloads each dataset, caches raw JSON in `data/raw/`, and saves flattened CSVs to `data/` |
| `data_prep.py` | Cleaning and feature building: team-name normalization, win %, 4-year talent, returning production, net portal talent, and the merged team-season table |
| `eda.py` | Builds the merged table, writes summary tables and figures, and runs regression assumption checks |
| `test_eda.py` | Tests for `data_prep.py` and the `eda.py` helpers on small hand-built tables |
| `requirements.txt` | Python dependencies |

## Setup

```bash
pip install -r requirements.txt
```

Get a free API key from <https://collegefootballdata.com/key>, then:

```bash
export CFBD_API_KEY="your-key"
```

## Usage

```bash
python collect_data.py          # download data into data/
python eda.py                   # write results into output/
```

Both scripts accept `--data-dir`, and `eda.py` also accepts `--out-dir`.
Re-running `collect_data.py` reuses cached raw files, so it only requests
what is missing.

### Output

- `output/merged_team_seasons.csv` — the cleaned team-season table
- `output/tables/` — dataset summaries (size, missingness, numeric and categorical statistics)
- `output/figures/` — histograms, by-season box plots, and scatter plots with fit lines
- `output/validity.txt` — assumption checks for the three models (normality, Breusch–Pagan heteroscedasticity, VIF)

## Tests

```bash
pytest            # or: python test_eda.py
flake8 .
```
