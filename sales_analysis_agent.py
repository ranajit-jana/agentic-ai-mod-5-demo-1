import os
import asyncio
import pandas as pd
from dotenv import load_dotenv
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.core import VectorStoreIndex, Document, StorageContext, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.tools import QueryEngineTool, FunctionTool
from llama_index.core.agent.workflow import ReActAgent
import logging
from pathlib import Path

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

llm = GoogleGenAI(model="gemini-2.5-flash", api_key=GOOGLE_API_KEY)
embed_model = GoogleGenAIEmbedding(model_name="gemini-embedding-2-preview", api_key=GOOGLE_API_KEY, embed_batch_size=1)

INDEX_STORAGE_DIR = "index_storage"

def compute_analytics(metric: str, column: str, filter_condition: str = None) -> float:
    """Compute statistical metrics (e.g., sum, average) on sales data with optional filtering."""
    df = pd.read_csv("sales_data.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    if filter_condition:
        df = df.query(filter_condition)
    if metric == "sum":
        return float(df[column].sum())
    elif metric == "average":
        return float(df[column].mean())
    return 0.0

analytics_tool = FunctionTool.from_defaults(
    fn=compute_analytics,
    name="analytics_tool",
    description=(
        "Computes statistical metrics (sum or average) on sales data with optional filters. "
        "Columns: OrderID, Date, Region (North/South/East/West/Central), "
        "Product (Laptop/Keyboard/Mouse/Monitor/Webcam/Headphones), Category, "
        "Quantity, UnitPrice, TotalSale, Year (int), Month (int). "
        "Example: metric='sum', column='TotalSale', filter_condition=\"Product == 'Laptop' and Year == 2024\""
    )
)

# STEP 4: Build or load vector index
index_storage_path = Path(INDEX_STORAGE_DIR)
if index_storage_path.exists():
    print("Loading existing index from disk...")
    storage_context = StorageContext.from_defaults(persist_dir=INDEX_STORAGE_DIR)
    index = load_index_from_storage(storage_context, embed_model=embed_model)
else:
    print("Building index from sales data (first run, please wait)...")
    sales_df = pd.read_csv("sales_data.csv")
    documents = []
    for _, row in sales_df.iterrows():
        text = (f"OrderID: {row['OrderID']}, Date: {row['Date']}, Region: {row['Region']}, "
                f"Product: {row['Product']}, Category: {row['Category']}, Quantity: {row['Quantity']}, "
                f"UnitPrice: {row['UnitPrice']}, TotalSale: {row['TotalSale']}")
        documents.append(Document(text=text))

    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    index = VectorStoreIndex.from_documents(
        documents,
        embed_model=embed_model,
        transformations=[splitter]
    )
    index.storage_context.persist(persist_dir=INDEX_STORAGE_DIR)
    print("Index built and saved.")

query_engine = index.as_query_engine(llm=llm, similarity_top_k=5)
sales_tool = QueryEngineTool.from_defaults(
    query_engine=query_engine,
    name="sales_data_tool",
    description="Provides insights from sales data including total sales, regional performance, and product analysis."
)

agent = ReActAgent(tools=[sales_tool, analytics_tool], llm=llm, verbose=True)

async def analyze_sales(query, query_history):
    handler = agent.run(user_msg=query, max_iterations=15)
    response = await handler
    query_history.append((query, str(response)))
    return response

async def main():
    query_history = []
    print("Welcome to InsightPulse: Your AI-Powered Sales Report Analysis Tool!")
    print("Enter your query (e.g., 'What is the total sales for Laptops in South in 2024?')")
    print("Type 'history' to view recent queries, 'exit' to quit.")

    while True:
        user_query = input("\nYour query: ").strip()
        if user_query.lower() == "exit":
            print("Exiting InsightPulse. Goodbye!")
            break
        if user_query.lower() == "history":
            if query_history:
                print("\nRecent Queries:")
                for i, (q, r) in enumerate(query_history[-5:], 1):
                    print(f"{i}. Query: {q}\n   Response: {r[:100]}...")
            else:
                print("No query history yet.")
            continue
        if not user_query:
            print("Please enter a valid query.")
            continue

        print(f"\nProcessing query: {user_query}")
        try:
            response = await analyze_sales(user_query, query_history)
            print(f"Response: {response}")
        except Exception as e:
            print(f"Error processing query: {e}")

if __name__ == "__main__":
    asyncio.run(main())
