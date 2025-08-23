import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.concurrency import asynccontextmanager
from pydantic import BaseModel
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.readers.file import PDFReader
import chromadb
import pdfplumber
from llama_index.core.schema import Document
import pandas as pd
from llama_index.core.indices.struct_store import JSONQueryEngine
from trainschema import train_info_schema
import json
from llama_index.core.tools import QueryEngineTool
import logging
import os
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import PydanticSingleSelector
from fastapi.staticfiles import StaticFiles

# Basic configuration for logging to the console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info('Starting the application...')

if not os.path.exists("data_set/train_info.json"):
    logging.info('Converting train_info.csv to train_info.json...')
    data_frames = pd.read_csv("data_set/train_info.csv")
    data_frames.to_json("data_set/train_info.json", orient="records")

def initialize_rag_system():
    # Step 1: Set your OpenAI API Key
    load_dotenv()
    os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

    logging.info('Loading PDF and initializing Chroma vector store...')
    reader = PDFReader()
    docs = reader.load_data("data/Indian_Railways_Annual_Report _23_24.pdf")

    # 2. Create a text splitter with chunk size & overlap
    splitter = SentenceSplitter(chunk_size=500, chunk_overlap=50)

    # Convert docs into chunks (nodes)
    nodes = splitter.get_nodes_from_documents(docs)

    # 3. Setup Chroma client for persistence
    chroma_client = chromadb.PersistentClient(path="./chroma_store")
    chroma_collection = chroma_client.get_or_create_collection("my_documents")

    # 4. Create Chroma vector store
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # 5. Build storage context with Chroma backend
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 6. Store the chunked nodes in Chroma
    index = VectorStoreIndex(nodes, storage_context=storage_context)

    logging.info('Index created and nodes stored in Chroma vector store for Tables.')
    #7. Extract tables from the PDF and convert them into LlamaIndex Document objects
    table_documents = extract_tables_to_documents("data/Indian_Railways_Annual_Report _23_24.pdf")

    nodes_table = splitter.get_nodes_from_documents(table_documents)

    index.insert_nodes(nodes_table, storage_context=storage_context)

    return index


def extract_tables_to_documents(pdf_path):
    """
    Extracts tables from a PDF and converts them into a list of LlamaIndex Document objects.
    Each document represents a single table.
    """
    documents = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            page_tables = page.extract_tables()
            if page_tables:
                for table_num, table in enumerate(page_tables):
                    # Convert the list of lists into a more readable string format
                    # This helps the LLM understand the table structure
                    table_str = f"Table on page {page_num + 1}, table {table_num + 1}:\n"
                    # Add a header row
                    header = table
                    header_row = [str(cell) if cell is not None else "" for cell in header]
                    table_str += "| " + " | ".join(header_row) + " |\n"
                    table_str += "| " + " | ".join(["---"] * len(header_row)) + " |\n"
                    # Add data rows
                    for row in table[1:]:
                        # Filter out None values to prevent errors
                        clean_row = [str(cell) if cell is not None else "" for cell in row]
                        table_str += "| " + " | ".join(clean_row) + " |\n"
                    
                    # Create a Document object
                    doc = Document(text=table_str, metadata={"page_number": page_num + 1})
                    documents.append(doc)
    return documents


def getQueryEngines():
    """
    Initializes the query engine with the train info JSON and schema.
    """
    logging.info('Initializing query engines...')
    query_engine = initialize_rag_system().as_query_engine()
    train_info_json_obj = None
    # Load the train info JSON file
    with open("data_set/train_info.json", "r", encoding="utf-8") as f:
        train_info_json_obj = json.load(f)

    # Initialize the query engine with the JSON data and schema
    nl_query_engine = JSONQueryEngine(
        json_value=train_info_json_obj,
        json_schema=train_info_schema
    )
    
    return nl_query_engine, query_engine


def getAgents():
    """
    Initializes the agents with the query engines.
    """
    logging.info('Setting up agents with query engines...')
    nl_query_engine, query_engine = getQueryEngines()

    query_engine_tool_1 = QueryEngineTool.from_defaults(
    query_engine=nl_query_engine,
    name="train_info_query_tool",
    description=(
        "This tool can answer questions related to Train details such as train journey from which place it starts and where it ends, and operations and it accept train no and other train details. "
        "Use it when the user asks about specific information that this query engine can handle."
        "Use this tool to get specific train information based on the train number or other details such as starting point and end point or destination.Do not use this tool for general queries or unrelated information."
    )
)
    
    query_engine_tool_2 = QueryEngineTool.from_defaults(
    query_engine=query_engine,
    name="train_annual_report_tool",
    description=(
        "This tool can answer questions related to Train Annual Report such as financial highlights, project details, and other operational information, Also" \
        "it can handle queries about the annual report's content and structure.Differnt set of question related to status and all" \
        "Use it when the user asks about specific information that this query engine can handle."
        "Do not use this tool for general queries or unrelated information which is outside of this context."
    )
)
    return query_engine_tool_1, query_engine_tool_2



async def runQuery(query, ag, ct):
    """
    Runs a query using the agent and returns the result.
    """
    logging.info(f'Running query: {query}')
    try:
        result = await ag.run(query, ctx=ct)
        return result
    except Exception as e:
        logging.error(f"Error running query: {e}")
        return str(e)

# Initialize the agents and query engines
query_engine_tool_1,query_engine_tool_2 = getAgents()
router = RouterQueryEngine(
    selector=PydanticSingleSelector.from_defaults(),  # Rule-based
    query_engine_tools=[query_engine_tool_1,query_engine_tool_2]
    )
app = FastAPI()

app.mount("/webapp", StaticFiles(directory="webapp"), name="static")

@app.post("/ask")
async def ask_question(query:str):
    try:
        response = router.query(query)
        return {"answer": str(response)}
    except Exception as e:
        return {"error": str(e)}

