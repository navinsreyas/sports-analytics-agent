"""
scripts/datapush.py
-------------------
ONE-TIME SETUP SCRIPT — run this before anything else.

Reads IPL CSVs and Premier League season CSVs from sportsdata/ and
uploads them to the Neon PostgreSQL database.

Run from the project root:
    python scripts/datapush.py
"""

import os
import glob
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# CONFIG — resolve paths relative to the PROJECT ROOT, not this file's folder
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))

NEON_DB_URL   = os.getenv("NEON_DB_URL", "").strip()
SOURCE_FOLDER = os.path.join(BASE_DIR, "sportsdata")


def fix_working_directory():
    print(f" Targeting folder: {SOURCE_FOLDER}")

    if os.path.exists(SOURCE_FOLDER):
        os.chdir(SOURCE_FOLDER)
        print(f"Files found here: {os.listdir()}")
    else:
        print("ERROR: Python cannot find that folder.")
        exit()


def connect_to_db():
    try:
        engine = create_engine(NEON_DB_URL)
        print("Connected to Neon Database successfully.")
        return engine
    except Exception as e:
        print(f"Failed to connect to DB: {e}")
        return None


def push_ipl_data(engine):
    target_files = {
        'ipl_matches':    'ipl_matches.csv',
        'ipl_deliveries': 'ipl_deliveries.csv',
    }

    for table_name, expected_name in target_files.items():
        if os.path.exists(expected_name):
            print(f"   Reading {expected_name}...")
            try:
                df = pd.read_csv(expected_name)
                df.columns = [c.lower().replace(' ', '_') for c in df.columns]
                print(f"Uploading {len(df)} rows to table '{table_name}'...")
                df.to_sql(table_name, engine, if_exists='replace', index=False, chunksize=1000)
                print(f"Upload complete for {table_name}.")
            except Exception as e:
                print(f"Error uploading {expected_name}: {e}")
        else:
            print(f"Could not find '{expected_name}' in this folder.")


def push_football_data(engine):
    folder_name = 'football_matches'
    if not os.path.exists(folder_name):
        print("Error: football_matches folder not found.")
        return

    search_path = os.path.join(folder_name, "*.csv")
    all_files   = glob.glob(search_path)

    if not all_files:
        all_files = glob.glob(os.path.join(folder_name, "*E0*"))

    if not all_files:
        print("Error: No football CSV files found.")
        return

    print(f"Found {len(all_files)} football files. Merging...")

    dataframes = []
    for filename in all_files:
        try:
            df = pd.read_csv(filename, encoding='latin1')
            dataframes.append(df)
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    if dataframes:
        combined_df = pd.concat(dataframes, ignore_index=True)
        combined_df.columns = [c.lower().replace(' ', '_') for c in combined_df.columns]
        print(f"Uploading {len(combined_df)} combined rows to 'football_matches'...")
        combined_df.to_sql('football_matches', engine, if_exists='replace', index=False, chunksize=1000)
        print("Upload complete for football_matches.")


if __name__ == "__main__":
    fix_working_directory()
    db_engine = connect_to_db()
    if db_engine:
        push_ipl_data(db_engine)
        push_football_data(db_engine)
        print("Successful")
