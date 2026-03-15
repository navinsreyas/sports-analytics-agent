import os
import sys
import pandas as pd
import sqlalchemy
from neo4j import GraphDatabase
from dotenv import load_dotenv

# dirname twice -> project root (one level above scripts/)
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FOLDER = os.path.join(BASE_DIR, "sportsdata")
ENV_PATH    = os.path.join(BASE_DIR, ".env")
BATCH_SIZE  = 500

load_dotenv(ENV_PATH)

NEO4J_URI      = "bolt://127.0.0.1:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "").strip()
NEON_DB_URL    = os.getenv("NEON_DB_URL", "").strip()


def run_batches(session, query, rows, label="rows"):
    total = len(rows)
    for start in range(0, total, BATCH_SIZE):
        chunk = rows[start : start + BATCH_SIZE]
        session.run(query, rows=chunk)
        done = min(start + BATCH_SIZE, total)
        print(f"      ...{done:,}/{total:,} {label}")


def create_constraints(session):
    print("\n[1/6] Creating uniqueness constraints...")
    session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Player) REQUIRE p.name IS UNIQUE")
    session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (t:Team)   REQUIRE t.name IS UNIQUE")
    print("      Done.")


def load_ipl_csvs():
    print("\n[2/6] Reading IPL CSV files...")
    deliveries_path = os.path.join(DATA_FOLDER, "ipl_deliveries.csv")
    matches_path    = os.path.join(DATA_FOLDER, "ipl_matches.csv")
    for path in (deliveries_path, matches_path):
        if not os.path.exists(path):
            print(f"      ERROR: file not found -- {path}")
            sys.exit(1)
    deliveries = pd.read_csv(deliveries_path)
    matches    = pd.read_csv(matches_path)
    print(f"      ipl_deliveries : {len(deliveries):,} rows")
    print(f"      ipl_matches    : {len(matches):,} rows")
    return deliveries, matches


def compute_player_roles(deliveries):
    print("\n[3/6] Computing player roles...")
    batters = set(deliveries["batter"].dropna().unique())
    bowlers = set(deliveries["bowler"].dropna().unique())
    roles = {}
    for name in batters | bowlers:
        if name in batters and name in bowlers:
            roles[name] = "ALLROUNDER"
        elif name in batters:
            roles[name] = "BATTER"
        else:
            roles[name] = "BOWLER"
    counts = {"BATTER": 0, "BOWLER": 0, "ALLROUNDER": 0}
    for r in roles.values():
        counts[r] += 1
    print(f"      Total players  : {len(roles):,}")
    for k, v in counts.items():
        print(f"        {k:<12}: {v:,}")
    return roles


def merge_players_and_ipl_teams(session, deliveries, player_roles):
    print("\n[4/6] Merging Player nodes and IPL Team nodes...")
    player_rows = [{"name": name, "role": role} for name, role in player_roles.items()]
    run_batches(session,
        '''
        UNWIND $rows AS row
        MERGE (p:Player {name: row.name})
        SET   p.role = row.role
        ''',
        player_rows, label="players")
    ipl_teams = (set(deliveries["batting_team"].dropna().unique())
                 | set(deliveries["bowling_team"].dropna().unique()))
    run_batches(session,
        '''
        UNWIND $rows AS row
        MERGE (t:Team {name: row.name})
        SET   t.league = 'IPL'
        ''',
        [{"name": t} for t in sorted(ipl_teams)], label="IPL teams")


def merge_represents(session, deliveries, matches):
    print("\n[5/6] Merging REPRESENTS relationships...")
    season_map = matches.set_index("id")["season"].to_dict()
    deliveries = deliveries.copy()
    deliveries["season"] = deliveries["match_id"].map(season_map).astype(str)
    batter_rels = (deliveries[["batter", "batting_team", "season"]]
                   .rename(columns={"batter": "player", "batting_team": "team"}))
    bowler_rels = (deliveries[["bowler", "bowling_team", "season"]]
                   .rename(columns={"bowler": "player", "bowling_team": "team"}))
    all_rels = (pd.concat([batter_rels, bowler_rels])
                .drop_duplicates()
                .dropna(subset=["player", "team", "season"])
                .reset_index(drop=True))
    run_batches(session,
        '''
        UNWIND $rows AS row
        MERGE (p:Player {name: row.player})
        MERGE (t:Team   {name: row.team})
        MERGE (p)-[:REPRESENTS {season: row.season}]->(t)
        ''',
        all_rels.to_dict("records"), label="REPRESENTS rels")


def merge_ipl_played_against(session, matches):
    print("\n[6a/6] Merging PLAYED_AGAINST for IPL...")
    pairs = matches[["team1", "team2"]].dropna().copy()
    pairs["a"] = pairs[["team1", "team2"]].min(axis=1)
    pairs["b"] = pairs[["team1", "team2"]].max(axis=1)
    pair_counts = pairs.groupby(["a", "b"]).size().reset_index(name="count")
    run_batches(session,
        '''
        UNWIND $rows AS row
        MERGE (t1:Team {name: row.a})
        MERGE (t2:Team {name: row.b})
        MERGE (t1)-[r:PLAYED_AGAINST]->(t2)
        SET   r.count = row.count
        ''',
        pair_counts.to_dict("records"), label="IPL PLAYED_AGAINST rels")


def merge_football_data(session, neon_url):
    print("\n[6b/6] Loading Premier League data from Neon DB...")
    try:
        engine = sqlalchemy.create_engine(neon_url)
        with engine.connect() as conn:
            football_df = pd.read_sql("SELECT hometeam, awayteam FROM football_matches", conn)
        print(f"      Loaded {len(football_df):,} football match rows.")
    except Exception as e:
        print(f"      ERROR: {e}")
        return
    pl_teams = (set(football_df["hometeam"].dropna().unique())
                | set(football_df["awayteam"].dropna().unique()))
    run_batches(session,
        '''
        UNWIND $rows AS row
        MERGE (t:Team {name: row.name})
        SET   t.league = 'Premier League'
        ''',
        [{"name": t} for t in sorted(pl_teams)], label="PL teams")
    pairs = football_df[["hometeam", "awayteam"]].dropna().copy()
    pairs["a"] = pairs[["hometeam", "awayteam"]].min(axis=1)
    pairs["b"] = pairs[["hometeam", "awayteam"]].max(axis=1)
    pair_counts = pairs.groupby(["a", "b"]).size().reset_index(name="count")
    run_batches(session,
        '''
        UNWIND $rows AS row
        MERGE (t1:Team {name: row.a})
        MERGE (t2:Team {name: row.b})
        MERGE (t1)-[r:PLAYED_AGAINST]->(t2)
        SET   r.count = row.count
        ''',
        pair_counts.to_dict("records"), label="PL PLAYED_AGAINST rels")


def print_summary(session):
    def count(q):
        return session.run(q).single()["c"]
    print("\n" + "=" * 60)
    print("  GRAPH BUILD COMPLETE")
    print("=" * 60)
    print(f"  Players        : {count('MATCH (p:Player) RETURN count(p) AS c'):>8,}")
    print(f"  Teams          : {count('MATCH (t:Team) RETURN count(t) AS c'):>8,}")
    print(f"  REPRESENTS     : {count('MATCH ()-[r:REPRESENTS]->() RETURN count(r) AS c'):>8,}")
    print(f"  PLAYED_AGAINST : {count('MATCH ()-[r:PLAYED_AGAINST]->() RETURN count(r) AS c'):>8,}")
    print("=" * 60)


def build_graph():
    if not NEO4J_PASSWORD:
        print("ERROR: NEO4J_PASSWORD not set in .env"); sys.exit(1)
    if not NEON_DB_URL:
        print("ERROR: NEON_DB_URL not set in .env");   sys.exit(1)
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
        driver.verify_connectivity()
        print("Connected to Neo4j.")
    except Exception as e:
        print(f"Connection failed: {e}"); sys.exit(1)
    deliveries, matches = load_ipl_csvs()
    player_roles = compute_player_roles(deliveries)
    with driver.session() as session:
        create_constraints(session)
        merge_players_and_ipl_teams(session, deliveries, player_roles)
        merge_represents(session, deliveries, matches)
        merge_ipl_played_against(session, matches)
        merge_football_data(session, NEON_DB_URL)
        print_summary(session)
    driver.close()
    print("\nGraph build finished.")


if __name__ == "__main__":
    build_graph()
