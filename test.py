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
from llama_index.llms.openai import OpenAI
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.workflow import Context
import logging
import os
import asyncio
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import PydanticSingleSelector

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info('Starting the application...')

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

train_info_json_obj = None
    # Load the train info JSON file
with open("data_set/train_info.json", "r", encoding="utf-8") as f:
        train_info_json_obj = json.load(f)



nl_query_engine = JSONQueryEngine(
        json_value=train_info_json_obj,
        json_schema=train_info_schema
)

query_engine_tool_1 = QueryEngineTool.from_defaults(
    query_engine=nl_query_engine,
    name="train_info_query_tool",
    description=(
        "You are an expert on Train details and operations. You extract train no from query or source or destination and use it to fetch the train details from the train info data."
        "When you extract train no from the user query, ensure that you accurately identify the number and use it to retrieve the relevant information.Do not check for substring matches or partial matches.when checking with train no, ensure that you match the complete train number exactly as it appears in the data."
        "This tool can answer questions related to Train details such as train journey from which place it starts and where it ends, and operations and it can accept train no and other train details as input. "
        "Example questions: Give me details about train no 1011, What is the destination of train number 12345, Where does train 5678 start its journey?"
        "Give me the source and destination of train number 54321."
        "Provide details about train no:107"
        "provide details about train no:12345"
        "What is the destination of train number 12345?"
        "provide details about train no 1011"
        "Where does train 5678 start its journey?"
        "Give me the source and destination of train number 54321."
        "Use it when the user asks about specific information that this query engine can handle."
        "Use this tool to get specific train information based on the train number or other details such as starting point and end point or destination.Do not use this tool for general queries or unrelated information."
    )
)

# async def main():
#     # rr = query_engine_tool_1.query_engine.query("Provide details about train no:107")
#     # print('rr...',rr)
#     agent = FunctionAgent(tools=[query_engine_tool_1],llm=OpenAI(model="gpt-4o"))
#     context = Context(agent)
#     #context = Context(agent)
#     result = await agent.run("provide details about train no 107?", ctx=context)
#     print('result...',result)

# if __name__ == "__main__":
#     import sys
#     if sys.platform.startswith("win"):
#         asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
#     asyncio.run(main())

router = RouterQueryEngine(
    selector=PydanticSingleSelector.from_defaults(),  # Rule-based
    query_engine_tools=[query_engine_tool_1]
)

response = router.query("provide details about train no 107?")
print(response)