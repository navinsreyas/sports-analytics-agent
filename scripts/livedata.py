import os
import requests
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

# dirname twice -> project root (one level above scripts/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))

NEON_DB_URL = os.getenv("NEON_DB_URL")
API_KEY     = os.getenv("FOOTBALL_API_KEY")


def fetch_premier_league_standings():
    print("Connecting to Football-Data.org API...")
    url     = "https://api.football-data.org/v4/competitions/PL/standings"
    headers = {"X-Auth-Token": API_KEY}
    try:
        response  = requests.get(url, headers=headers)
        response.raise_for_status()
        data      = response.json()
        standings = data["standings"][0]["table"]
        df        = pd.json_normalize(standings)
        df_clean  = df[[
            "position", "team.name", "playedGames", "won", "draw", "lost",
            "points", "goalsFor", "goalsAgainst", "goalDifference"
        ]].copy()
        df_clean.columns = [
            "position", "team_name", "played", "won", "drawn", "lost",
            "points", "goals_for", "goals_against", "goal_diff"
        ]
        top = df_clean.iloc[0]["team_name"]
        print(f"Data fetched! Top team: {top}")
        return df_clean
    except Exception as e:
        print(f"API Error: {e}")
        return None


def upload_to_neon(df):
    try:
        engine = create_engine(NEON_DB_URL)
        df.to_sql("live_table", engine, if_exists="replace", index=False)
        print(f"live_table updated with {len(df)} rows.")
    except Exception as e:
        print(f"Database Error: {e}")


if __name__ == "__main__":
    df = fetch_premier_league_standings()
    if df is not None:
        upload_to_neon(df)
