# Sports Analytics Agent

> A conversational AI agent that answers detailed questions about **IPL Cricket (2008-2024)**
> and **English Premier League Football** using real databases -- not LLM memory.

---

## What Is This?

Most AI chatbots answer sports questions from training memory -- which means they can
confidently hallucinate wrong stats, outdated scores, or players who never existed.

This agent is different. Every question is **routed to a real data source first**: a
PostgreSQL database of 800K+ ball-by-ball delivery records (260K batting deliveries after filtering), a vector store built from
official rulebooks, or a Neo4j knowledge graph of player-team relationships. The LLM is
only used to write queries and translate the results into plain English.

**The databases are the brain. The LLM is just the voice.**

---

## Architecture

```
User Question
      |
      v
  [app.py]  <-- image uploaded? --> [vision_tool.py: Gemini Flash]
      |                                    |
      |<-- extracts structured sports data-/
      |
      v
  [agent.py: senior_agent()]
      |
      |-- Step 1 - Router LLM:   REFUSE / SQL / RAG / GRAPH / CHAT
      |-- Step 2 - Refiner LLM:  resolve pronouns from chat history
      |
      |-- SQL    --> Neon PostgreSQL   (player stats, match results, standings)
      |-- RAG    --> ChromaDB          (cricket and football rulebooks)
      |-- GRAPH  --> Neo4j             (player-team-season relationships)
      |-- REFUSE --> polite decline    (out-of-scope questions)
      |
      v
  Synthesiser LLM --> plain English answer
```

### The Four Specialists

| Mode | Nickname | What It Does | Data Source |
|------|----------|--------------|-------------|
| SQL | The Calculator | Counts, averages, rankings, live standings | Neon PostgreSQL |
| RAG | The Librarian | Rules, definitions, LBW, offside explained | ChromaDB + PDFs |
| GRAPH | The Detective | Who played for whom, player-team links | Neo4j Graph DB |
| VISION | The Eyes | Reads scoreboards and stat-card screenshots | Gemini Flash |

---

## Features

- Natural language Q&A -- ask in plain English, get plain English back
- IPL Cricket (2008-2024) -- ball-by-ball stats, batting, bowling, match summaries
- Premier League Football -- historical results and live standings
- Rules and definitions -- LBW, DRS, offside, VAR from official rulebooks
- Player-team graph -- 'Which teams did Rohit Sharma play for?' answered via Neo4j
- Image understanding -- upload a scoreboard screenshot, ask questions about it
- Conversation memory -- 'his average' resolves correctly across chat turns
- Scope enforcement -- Test cricket, La Liga, Champions League politely refused

---

## Tech Stack

| Tool | Role |
|------|------|
| Streamlit | Chat UI frontend |
| LangChain | LLM orchestration framework |
| Groq + LLaMA 3.3 70B | Fast inference for router, refiner, and synthesiser LLMs |
| Neon PostgreSQL | Cloud SQL database for IPL and EPL match data |
| ChromaDB | Local vector store for semantic search over PDF rulebooks |
| Neo4j | Graph database for player-team-season relationship queries |
| Gemini Flash | Multimodal vision -- extracts data from sports images |
| HuggingFace all-MiniLM-L6-v2 | Sentence embeddings for RAG vector search |
| football-data.org API | Live EPL standings feed |

---

## Project Structure

```
sports-analytics-agent/
|
|-- agent.py               # Core: router LLM, tools, synthesiser LLM
|-- app.py                 # Streamlit chat UI + sidebar image uploader
|-- vision_tool.py         # Gemini Flash vision pre-processor
|
|-- scripts/               # One-time setup + maintenance scripts
|   |-- datapush.py        # Load IPL and EPL CSVs into Neon PostgreSQL
|   |-- build_knowledge.py # Build ChromaDB vector store from PDF rulebooks
|   |-- buildgraph.py      # Populate Neo4j with players, teams, seasons
|   `-- livedata.py        # Refresh live EPL standings (run periodically)
|
|-- sportsdata/            # Raw data files (gitignored -- too large)
|   |-- ipl_deliveries.csv     (~800K rows raw, 260K batting deliveries after filtering)
|   |-- ipl_matches.csv        (~1K rows, match summaries)
|   |-- football_matches/      (season CSVs in E0.csv format)
|   |-- cricket.pdf            (ICC Cricket rulebook)
|   |-- football.pdf           (FA / Premier League rulebook)
|   `-- chroma_db/             (auto-generated vector store -- do not commit)
|
|-- .env.example           # Credential template (safe to commit)
|-- .gitignore
|-- requirements.txt
`-- CLAUDE.md              # AI assistant project instructions
```

---

## Prerequisites

- Python 3.10 or later
- [Neo4j Desktop](https://neo4j.com/download/) installed and running locally on `bolt://localhost:7687`
- A [Neon](https://neon.tech) PostgreSQL project (free tier is sufficient)
- API keys for Groq, Gemini, and football-data.org (all have free tiers)

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/navinsreyas/sports-analytics-agent.git
cd sports-analytics-agent
pip install -r requirements.txt
```

### 2. Configure secrets

```bash
cp .env.example .env
```

Open `.env` and fill in your real values:

```
GROQ_API_KEY=your_groq_api_key_here
NEON_DB_URL=postgresql://user:password@host/dbname?sslmode=require
NEO4J_PASSWORD=your_neo4j_password_here
GEMINI_API_KEY=your_gemini_api_key_here
FOOTBALL_API_KEY=your_football_data_org_key_here
```

| Key | Where to get it |
|-----|-----------------|
| GROQ_API_KEY | console.groq.com -- free |
| NEON_DB_URL | neon.tech -- create a project, copy the connection string |
| NEO4J_PASSWORD | Neo4j Desktop -- set when creating a local database |
| GEMINI_API_KEY | aistudio.google.com/apikey -- free |
| FOOTBALL_API_KEY | football-data.org -- free registration |

### 3. Add your data files

Place the following files inside the `sportsdata/` folder:

```
sportsdata/
|-- ipl_deliveries.csv       # Source: Kaggle IPL Complete Dataset
|-- ipl_matches.csv          # Source: Kaggle IPL Complete Dataset
|-- football_matches/        # Source: football-data.co.uk (E0.csv per season)
|-- cricket.pdf              # ICC Playing Conditions rulebook
`-- football.pdf             # FA / Premier League rulebook
```

### 4. Run setup scripts (one time only)

Run these in order. Each step must complete successfully before moving to the next.

```bash
# Step 1: Load all CSV data into Neon PostgreSQL
python scripts/datapush.py

# Step 2: Build the ChromaDB vector store from PDF rulebooks
python scripts/build_knowledge.py

# Step 3: Populate the Neo4j knowledge graph
#         (Neo4j Desktop must be running before you run this)
python scripts/buildgraph.py
```

### 5. Refresh live EPL standings (optional, run anytime)

```bash
python scripts/livedata.py
```

### 6. Launch the app

```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

---

## Example Questions

**IPL Batting**
```
Who has the most centuries in IPL history?
What is Virat Kohli's average in the powerplay overs?
List the top 5 run scorers from IPL 2023.
```

**IPL Bowling**
```
Who has taken the most wickets in IPL history?
What is Jasprit Bumrah's economy rate in death overs?
```

**Premier League**
```
Which team has won the most Premier League titles?
What are the current EPL standings?
How many goals did Erling Haaland score in 2022-23?
```

**Rules (RAG)**
```
What is the DRS rule in cricket?
Explain the offside rule in football.
```

**Graph**
```
Which IPL teams has MS Dhoni played for?
Which players have represented both MI and CSK?
```

**Vision**
```
[Upload a scoreboard screenshot]
Who won this match and by how many runs?
```

---

## How the Routing Works

Every question goes through two LLM calls before any data is fetched:

1. **Router** -- classifies the intent as `SQL`, `RAG`, `GRAPH`, `CHAT`, or `REFUSE`
2. **Refiner** -- rewrites the question, replacing pronouns (`his`, `that team`) with
   actual names resolved from the last 3 chat turns

The agent then calls the appropriate tool, retrieves structured data, and passes it to
a **Synthesiser** that produces the final readable answer. The LLM never guesses facts;
it only interprets data that was explicitly retrieved.

---

## Limitations

- **Neo4j must run locally.** The graph tool connects to `bolt://localhost:7687`. If
  Neo4j Desktop is not running, graph queries fail gracefully with an error message.
- **Scope is intentionally narrow.** Test cricket, ODIs, La Liga, Bundesliga, and
  Champions League questions are refused by design to prevent hallucination.
- **No automatic standings refresh.** Run `scripts/livedata.py` manually to pull the
  latest EPL table from the API.
- **Vision requires a Gemini key.** If `GEMINI_API_KEY` is absent, the app still
  works fully -- image upload is simply disabled.

---

## License

MIT
