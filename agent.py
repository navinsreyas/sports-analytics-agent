import os
import sys
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_neo4j import Neo4jGraph

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ENV_PATH = os.path.join(BASE_DIR, ".env")

DATA_FOLDER = os.path.join(BASE_DIR, "sportsdata")
CHROMA_PATH = os.path.join(DATA_FOLDER, "chroma_db")

if os.path.exists(ENV_PATH):
    load_dotenv(ENV_PATH)
else:
    print(f"❌ Error: .env file not found at {ENV_PATH}")
    sys.exit(1)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
NEON_DB_URL = os.getenv("NEON_DB_URL")
NEO4J_URI = "bolt://127.0.0.1:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

try:
    llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY, temperature=0)
except Exception as e:
    print(f"Error initializing LLM: {e}")
    sys.exit(1)

CHAT_HISTORY = []

SYSTEM_INSTRUCTIONS = """
You are a Senior Sports Data Analyst specialized in **T20 Cricket (IPL)** and **English Football (Premier League)**.
Your goal is to generate executable, accurate SQL queries for a PostgreSQL database.

###  SCOPE GUARDS:
1. **Refusal Policy — return the literal text OUT_OF_SCOPE for any of these:**
   - International cricket of ANY format: Test matches, ODIs, T20Is, bilateral series
     (e.g. India vs South Africa, Australia vs England, World Cup, Champions Trophy).
   - Any cricket competition that is NOT the IPL.
   - La Liga, Bundesliga, Serie A, Champions League, or any football outside the Premier League.
   - Generic sports trivia unrelated to IPL or Premier League.
   Do NOT generate SQL for these. Do NOT answer conversationally. Return: OUT_OF_SCOPE
2. **Ambiguity:** "Who won the league?" -> Assume Premier League. "Who won the match?" -> Check context.

###  DATABASE MAP (Use Views First!):
**1. VIEW: 'player_batting_stats' (Pre-Calculated Batting)**
   - **Use for:** Average, Strike Rate, Centuries, Ducks, Top Run Scorers.
   - **Columns:** player_name, matches_played, total_runs, balls_faced, batting_avg, strike_rate, centuries.
   - *Example:* "Best strike rate?" -> `SELECT player_name, strike_rate FROM player_batting_stats WHERE balls_faced > 50 ORDER BY strike_rate DESC LIMIT 5;`

**2. VIEW: 'player_bowling_stats' (Pre-Calculated Bowling)**
   - **Use for:** Economy, Wickets, Best Bowlers.
   - **Columns:** player_name, matches, runs_conceded, wickets, economy.

**3. VIEW: 'football_season_stats' (Pre-Calculated League Tables)**
   - **Use for:** Winners, points, relegation, clean sheets.
   - **Columns:** team, season, played, won, drawn, lost, points, clean_sheets.

**4. TABLE: 'live_table' (Real-Time Data)**
   - **Use for:** "Current", "Live", "Right Now" questions regarding Football/EPL.
   - *Example:* "Who is top?" -> `SELECT team_name, points FROM live_table ORDER BY points DESC LIMIT 1;`

**5. TABLE: 'ipl_deliveries' (Granular Data)**
   - **Use ONLY for:** Phases like "Death Overs" (`over > 15`) or "Powerplay" (`over <= 6`).

###  SQL BEST PRACTICES:
- Always use `ILIKE` for names: `player_name ILIKE '%Kohli%'`.
- Always `LIMIT 5` unless specified.
- **Superlative / overall-best queries** ("most", "highest", "best" with no qualifier):
  NEVER add a WHERE clause. Use ORDER BY + LIMIT only.
  Example: "Who has the most runs in IPL history?"
    → SELECT player_name, total_runs FROM player_batting_stats ORDER BY total_runs DESC LIMIT 1;
  Example: "Best economy rate overall?"
    → SELECT player_name, economy FROM player_bowling_stats ORDER BY economy ASC LIMIT 1;
- Return ONLY the SQL query. No markdown.
"""


def run_sql_tool(question):
    """Calculates Stats using Neon Postgres"""
    print("   ⚙️ Calculator (SQL) Activated...")
    try:
        db = SQLDatabase.from_uri(NEON_DB_URL)
        
        prompt = f"""
        {SYSTEM_INSTRUCTIONS}
        
        User Question: "{question}"
        
        Write the SQL query now:
        """
        response = llm.invoke(prompt)
        sql_query = response.content.replace("```sql", "").replace("```", "").strip()
        
        if not sql_query.lower().startswith("select"):
            return "Error: Unsafe query. Only SELECT allowed."
            
        print(f"Executing: {sql_query}")
        result = db.run(sql_query)
        return f"Database Data: {result}" if result else "No results found."
    except Exception as e:
        return f"SQL Error: {e}"

def run_rag_tool(question):
    """Searches Rules in PDF Vector Store"""
    print(f"   ⚙️ Librarian (RAG) Activated... (Checking {CHROMA_PATH})")
    
    if not os.path.exists(CHROMA_PATH):
        return f"Error: Knowledge Base not found at {CHROMA_PATH}. Please run build_knowledge.py."
    
    try:
        embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
        
        docs = vector_db.similarity_search(question, k=3)
        context = "\n".join([doc.page_content for doc in docs])
        return f"Rulebook Excerpt: {context[:2000]}"
    except Exception as e:
        return f"RAG Error: {e}"

def run_graph_tool(question):
    """Finds Relationships in Neo4j"""
    print("   ⚙️ Detective (Graph) Activated...")
    try:
        graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)
        
        prompt = f"""
        You are a Neo4j Cypher expert.

        ### GRAPH SCHEMA (STRICT):

        **Node types:**
        - (:Player {{name: "V Kohli", role: "BATTER | BOWLER | ALLROUNDER"}})
        - (:Team   {{name: "Mumbai Indians", league: "IPL | Premier League"}})

        **Relationship types:**
        - (:Player)-[:REPRESENTS {{season: "2023"}}]->(:Team)
          — links a player to the team they played for in a specific IPL season.
        - (:Team)-[:PLAYED_AGAINST {{count: 12}}]-(:Team)
          — links two teams that have faced each other; count = number of matches.
          — covers both IPL teams and Premier League teams (never cross-league).
          — always query this UNDIRECTED: MATCH (t1)-[:PLAYED_AGAINST]-(t2)

        **What does NOT exist in the graph:**
        - No player stats (runs, wickets, economy, centuries).
        - No football/Premier League players — only Team nodes for the PL.
        - No 'captain', 'date', 'venue', or 'result' properties.

        ### QUERY RULES:
        1. **Season filter** — if the user mentions a specific year (e.g. "in 2019"),
           filter on the relationship property: MATCH (p)-[r:REPRESENTS]->(t) WHERE r.season = '2019'
        2. **Head-to-head / rivalry questions** — use PLAYED_AGAINST (undirected):
           MATCH (t1:Team {{name: 'X'}})-[r:PLAYED_AGAINST]-(t2:Team {{name: 'Y'}}) RETURN r.count
        3. **Common players between two teams** — MATCH via REPRESENTS, not PLAYED_AGAINST.
        4. **League filter** — use t.league = 'IPL' or t.league = 'Premier League' when relevant.
        5. If the question asks for stats or 'captain', return players for that team instead.
        6. Use case-insensitive matching where possible: WHERE toLower(p.name) CONTAINS 'kohli'

        User Question: "{question}"

        Write a single Cypher query to answer this.
        Return ONLY the query. No markdown, no explanation.
        """
        
        response = llm.invoke(prompt)
        cypher_query = response.content.replace("```cypher", "").replace("```", "").strip()
        print(f"      🕸️ Executing: {cypher_query}")
        
        result = graph.query(cypher_query)
        return f"Graph Connections: {result}" if result else "No connections found."
    except Exception as e:
        return f"Graph Error: {e}"


def senior_agent(question):
    memory_context = "\n".join([f"User: {q}\nAI: {a}" for q, a in CHAT_HISTORY[-3:]])
  
    print(f"\n🧠 Analyzing: '{question}'")
    
    router_prompt = f"""
    You are a Senior Sports Analyst Router.
    
    ### CHAT HISTORY:
    {memory_context}
    
    ### CURRENT QUESTION:
    "{question}"
    
    ### TASK:
    1. Check the Chat History. Does "he", "it", or "they" refer to a previous entity?
    2. Classify the question into ONE tool category.
    
    CATEGORIES:
    - REFUSE: International cricket of any format (Test, ODI, T20I, World Cup, bilateral series
              such as India vs South Africa, Australia vs England, etc.), any cricket outside
              the IPL, any football outside the Premier League (La Liga, Champions League, etc.).
              ALWAYS use REFUSE for these — never SQL or CHAT.
    - SQL: IPL or Premier League stats, numbers, scores, rankings, live tables,
           "Who won", "How many goals", "Best strike rate", "Most runs".
    - RAG: Rules, laws, definitions, "What is a wide?", "Offside rule".
    - GRAPH: Relationships, team membership, season-specific team queries, "Who played for X?",
             "Teammates", "Common players between two teams", "Head-to-head", "Rivalry",
             "How many times did X face Y?", "Which teams did player X represent?",
             "Who are the allrounders in team X?", "Which Premier League teams exist?".
    - CHAT: Greetings, generic talk, or if the answer is already in history.

    Return ONLY one word: REFUSE, SQL, RAG, GRAPH, or CHAT.
    """
    choice = llm.invoke(router_prompt).content.strip().upper()
    
    tool_name = "CHAT"
    if "REFUSE" in choice: tool_name = "REFUSE"
    elif "SQL" in choice: tool_name = "SQL"
    elif "RAG" in choice: tool_name = "RAG"
    elif "GRAPH" in choice: tool_name = "GRAPH"
    
    print(f"Router decided: {tool_name}")

    refined_question = question
    if tool_name != "CHAT":
        refine_prompt = f"""
        Chat History: {memory_context}
        Current Question: "{question}"

        Your ONLY job is to replace pronouns (he/she/it/they/his/her/their) with the
        actual names they refer to in the Chat History.

        STRICT RULES:
        - If the question contains NO pronouns and is already clear and standalone,
          return it EXACTLY as written — do not add player names, assumptions, or
          any other changes whatsoever.
        - Only substitute a pronoun when you are 100% certain what it refers to
          from the Chat History.
        - Never add names, stats, teams, or context that are not explicitly in the
          Chat History.
        - Do not rephrase, reorder, expand, or make the question "more specific".

        Examples:
          Pronoun present → "How many runs did he score?" (history: Kohli)
            becomes: "How many runs did Virat Kohli score?"
          No pronoun → "Who has the most runs in IPL history?"
            stays:    "Who has the most runs in IPL history?"
          No pronoun → "What is the best economy rate in IPL?"
            stays:    "What is the best economy rate in IPL?"

        Return ONLY the (possibly unchanged) question. No explanation.
        """
        refined_question = llm.invoke(refine_prompt).content.strip()
        if refined_question != question:
            print(f"   🔄 Contextualized: '{refined_question}'")

    data = ""
    if tool_name == "REFUSE":
        data = "OUT_OF_SCOPE"
    elif tool_name == "SQL":
        data = run_sql_tool(refined_question)
    elif tool_name == "RAG":
        data = run_rag_tool(refined_question)
    elif tool_name == "GRAPH":
        data = run_graph_tool(refined_question)
    else:
        data = "No external tool needed. Answer conversationally."

    final_prompt = f"""
    You are a friendly Senior Sports Analyst.

    User Question: "{question}"
    Context/Tool Data: "{data}"

    Instructions:
    1. If Context/Tool Data is "OUT_OF_SCOPE": politely refuse. Explain that you only
       cover IPL cricket (2008-2024) and English Premier League football. Do not answer
       the question or speculate. Suggest rephrasing for IPL or Premier League instead.
    2. Otherwise, answer the user professionally based on the data provided.
    3. If the data is an error message, apologize and explain what went wrong.
    4. Do not mention "SQL", "Cypher", or "Vectors" in the final answer. Just give the sports info.
    """
    final_response = llm.invoke(final_prompt).content
    

    CHAT_HISTORY.append((question, final_response))
    
    return final_response


if __name__ == "__main__":
    print("\n💬 Senior Agent Online! Data Folder: sportsdata")
    print("-" * 60)
    
    while True:
        try:
            q = input("You: ")
            if q.lower() in ["exit", "quit"]: break
            
            response = senior_agent(q)
            print(f"🤖 AI: {response}\n")
        except KeyboardInterrupt:
            print("\n👋 Exiting...")
            break
        except Exception as e:
            print(f"❌ Critical Error: {e}")