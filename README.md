# InsightPulse: AI-Powered Sales Analysis Agent

A conversational CLI tool that lets you query sales data using natural language, powered by a LlamaIndex ReAct agent with Google Gemini.

---

## How It Works

`sales_analysis_agent.py` builds a two-tool ReAct agent:

1. **`sales_data_tool`** — A LlamaIndex `VectorStoreIndex` built from `sales_data.csv`. Each row is embedded using Google's Gemini embedding model and stored locally in `index_storage/`. Used for open-ended semantic questions like "Which region had the best performance?".

2. **`analytics_tool`** — A Python function that runs direct statistical computations (sum, average) on the CSV with optional filters. Used for precise numeric queries like "Total sales for Laptops in South in 2024".

The agent (ReAct loop) decides which tool to call based on your query, reasons step-by-step, and returns a natural language response.

---

## Agent Strategy: ReActAgent

### What is ReAct?

ReAct (**Re**asoning + **Act**ing) is an agent loop pattern where the LLM alternates between:

```
Thought  → what do I need to do?
Action   → call a tool with arguments
Observation → read the tool result
Thought  → is this enough or do I need another tool?
... (repeat up to max_iterations)
Answer   → return final response to user
```

In [sales_analysis_agent.py:83](sales_analysis_agent.py#L83):
```python
agent = ReActAgent(tools=[sales_data_tool, analytics_tool], llm=llm, verbose=True)
```

With `verbose=True` you can see each Thought/Action/Observation step printed in the terminal.

### Why ReActAgent for this project?

- The query could need **semantic search** (sales_data_tool) OR **exact computation** (analytics_tool) — ReAct lets the LLM decide at runtime
- Some queries need **multiple tool calls** in sequence (e.g., fetch region data, then compute average)
- Simple and transparent — easy to debug with verbose output

---

## Other Agent Options in LlamaIndex

| Agent | How it works | Best for |
|---|---|---|
| **ReActAgent** ✅ (this project) | Think → Act → Observe loop | Multi-tool, multi-step reasoning |
| **FunctionCallingAgent** | Uses native LLM function-calling API (OpenAI/Gemini style) | When your LLM supports structured tool calls natively — faster, less verbose |
| **StructuredPlannerAgent** | Plans all steps upfront, then executes | Complex workflows where you want a full plan before any execution |
| **LLMCompilerAgent** | Parallelizes independent tool calls | When multiple tools can run simultaneously to save latency |
| **CustomAgent** | You define the loop logic entirely | Specialized workflows not covered by above |

### When to switch away from ReAct

- **Use `FunctionCallingAgent`** if Gemini's native function-calling gives better structured outputs and you don't need the verbose reasoning trace
- **Use `StructuredPlannerAgent`** if queries require many sequential steps and you want the plan reviewable before execution
- **Use `LLMCompilerAgent`** if you have many independent tools and want parallel execution for speed

---

### Index Storage
On first run, the agent embeds all sales records and saves the vector index to `index_storage/` (auto-created). On subsequent runs it loads from disk, skipping re-embedding.

---

## Prerequisites

- Python 3.8+
- A Google API key with Gemini API enabled ([Google AI Studio](https://aistudio.google.com/))

---

## Setup

```bash
# 1. Create and activate virtual environment
python -m venv sales_analyzer_env
source sales_analyzer_env/bin/activate  # Windows: sales_analyzer_env\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your API key
echo "GOOGLE_API_KEY=your_key_here" > .env

# 4. Generate dummy sales data
python create_dummy_data.py

# 5. Run the agent
python sales_analysis_agent.py
```

---

## Usage

```
Your query: What is the total sales for Laptops in South in 2024?
Your query: What is the average unit price of Monitors?
Your query: Which region had the highest sales in 2023?
Your query: history   # view last 5 queries
Your query: exit      # quit
```

---

## Project Structure

```
├── sales_analysis_agent.py   # Main agent: indexing, tools, ReAct loop
├── create_dummy_data.py       # Generates synthetic sales_data.csv
├── sales_data.csv             # Dataset (generated, not committed if empty)
├── requirements.txt           # Python dependencies
├── setup.txt                  # Step-by-step setup reference
├── .env                       # API key (never commit)
├── .env_copy                  # Safe template (no real key)
└── index_storage/             # Auto-generated vector index (gitignored)
```

---

## Sales Data Schema

| Column | Description |
|---|---|
| OrderID | Unique order identifier |
| Date | Order date |
| Region | North / South / East / West / Central |
| Product | Laptop / Keyboard / Mouse / Monitor / Webcam / Headphones |
| Category | Product category |
| Quantity | Units sold |
| UnitPrice | Price per unit |
| TotalSale | Quantity × UnitPrice |

---

## Troubleshooting

- **Missing API key** — ensure `GOOGLE_API_KEY` is set in `.env`
- **Empty results** — run `create_dummy_data.py` to populate `sales_data.csv`
- **Slow first run** — normal; the agent is embedding all rows and building the index
- **Dependency errors** — run `pip install -r requirements.txt` inside the virtual environment
