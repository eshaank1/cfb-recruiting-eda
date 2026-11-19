"""
Downloads the College Football Data (CFBD) datasets used in the EDA.

Usage:
    export CFBD_API_KEY="your-key"
    python collect_data.py [--data-dir data]

For every season, each endpoint is requested, the raw JSON is cached in
<data-dir>/raw/, and the flattened result is saved as a CSV in <data-dir>/.
Existing raw files are reused, so re-running the script is cheap.
"""

import argparse
import json
import os
import sys
import time

import pandas as pd
import requests

BASE_URL = "https://api.collegefootballdata.com"

GAMES_SEASONS = range(2013, 2026)
RECRUITING_SEASONS = range(2011, 2026)
RETURNING_SEASONS = range(2014, 2026)
PORTAL_SEASONS = range(2021, 2026)
FBS_SEASONS = range(2013, 2026)

# name -> (path, seasons, extra query parameters)
ENDPOINTS = {
    "games": ("/games", GAMES_SEASONS, {"seasonType": "regular"}),
    "recruiting": ("/recruiting/teams", RECRUITING_SEASONS, {}),
    "returning": ("/player/returning", RETURNING_SEASONS, {}),
    "portal": ("/player/portal", PORTAL_SEASONS, {}),
    "fbs_teams": ("/teams/fbs", FBS_SEASONS, {}),
}

MAX_TRIES = 4


def get_api_key():
    """
    Reads the API key from the CFBD_API_KEY environment variable.

    Returns:
        The key as a string. Exits with a message if it is not set.
    """
    key = os.environ.get("CFBD_API_KEY")
    if not key:
        sys.exit("Set the CFBD_API_KEY environment variable first.")
    return key


def fetch_json(path, params, key):
    """
    Makes one authenticated GET request, retrying on transient errors.

    Args:
        path: endpoint path such as "/games".
        params: dictionary of query parameters.
        key: the CFBD API key.

    Returns:
        The decoded JSON (a list of records).
    """
    headers = {"Authorization": "Bearer " + key}
    for attempt in range(MAX_TRIES):
        response = requests.get(BASE_URL + path, params=params,
                                headers=headers, timeout=60)
        if response.status_code == 200:
            return response.json()
        if response.status_code in (429, 500, 502, 503, 504):
            time.sleep(2 ** attempt)
            continue
        response.raise_for_status()
    response.raise_for_status()
    return []


def collect_endpoint(name, data_dir, key):
    """
    Downloads one endpoint for all of its seasons and saves a CSV.

    Args:
        name: a key of ENDPOINTS.
        data_dir: folder that receives <name>.csv and raw/ JSON files.
        key: the CFBD API key.

    Returns:
        The combined DataFrame (also written to <data_dir>/<name>.csv).
    """
    path, seasons, extra = ENDPOINTS[name]
    raw_dir = os.path.join(data_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    frames = []
    for season in seasons:
        raw_path = os.path.join(raw_dir, "{}_{}.json".format(name, season))
        if os.path.exists(raw_path):
            with open(raw_path) as f:
                records = json.load(f)
        else:
            params = dict(extra, year=season)
            records = fetch_json(path, params, key)
            with open(raw_path, "w") as f:
                json.dump(records, f)
        frame = pd.json_normalize(records)
        if "season" not in frame.columns:
            frame["season"] = season
        frames.append(frame)
        print("{} {}: {} rows".format(name, season, len(frame)))
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(os.path.join(data_dir, name + ".csv"), index=False)
    return combined


def main():
    """Downloads every endpoint listed in ENDPOINTS."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()
    key = get_api_key()
    for name in ENDPOINTS:
        collect_endpoint(name, args.data_dir, key)


if __name__ == "__main__":
    main()
