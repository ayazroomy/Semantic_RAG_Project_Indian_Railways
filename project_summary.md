# Project Summary: RAG System for Indian Railways Annual Report 2023-2024 & Train Details

## Overview
This project implements a Retrieval-Augmented Generation (RAG) system to answer queries about the Indian Railways Annual Report 2023-2024 and general train information. It combines unstructured (PDF) and structured (CSV/JSON) data using LlamaIndex, OpenAI LLMs, ChromaDB, and FastAPI.

---

## 1. Data Ingestion & Preprocessing

### PDF Data Loading
- Used `PDFReader` to load the annual report PDF:
  ```python
  reader = PDFReader()
  docs = reader.load_data("data/Indian_Railways_Annual_Report _23_24.pdf")
  ```

### Text Chunking
- Split PDF content into chunks for vectorization:
  ```python
  splitter = SentenceSplitter(chunk_size=500, chunk_overlap=50)
  nodes = splitter.get_nodes_from_documents(docs)
  ```

### Vector Database Setup
- Used ChromaDB for persistent vector storage:
  ```python
  chroma_client = chromadb.PersistentClient(path="./chroma_store")
  chroma_collection = chroma_client.get_or_create_collection("my_documents")
  vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
  storage_context = StorageContext.from_defaults(vector_store=vector_store)
  index = VectorStoreIndex(nodes, storage_context=storage_context)
  ```

### Table Extraction from PDF
- Used `pdfplumber` to extract tables and convert them to LlamaIndex Document objects:
  ```python
  documents = extract_tables_to_documents("data/Indian_Railways_Annual_Report _23_24.pdf")
  nodes_table = splitter.get_nodes_from_documents(documents)
  index.insert_nodes(nodes_table, storage_context=storage_context)
  ```

---

## 2. Query Engine Construction

### PDF Query Engine (Unstructured Data)
- Created a query engine for the PDF content:
  ```python
  query_engine = index.as_query_engine()
  ```

### Train Info Query Engine (Structured Data)
- Loaded train info from CSV and saved as JSON:
  ```python
  data_frames = pd.read_csv("data_set/train_info.csv")
  data_frames.to_json("data_set/train_info.json", orient="records")
  ```
- Defined a JSON schema for train info and used `JSONQueryEngine`:
  ```python
  train_info_schema = { ... }
  nl_query_engine = JSONQueryEngine(json_value=train_info_json_obj, json_schema=train_info_schema)
  ```

---

## 3. Querying and Evaluation

### Querying the Engines
- Example PDF query:
  ```python
  response = query_engine.query("At What Year Indian Railways have conducted full scale disaster management exercise?")
  ```
- Example train info query:
  ```python
  nl_response = nl_query_engine.query("Give the details for the train no 107?")
  ```
- Table and name-based queries are also supported.

---

## 4. Agent and Tool Construction

### QueryEngineTool
- Created tools for both query engines:
  ```python
  query_engine_tool_1 = QueryEngineTool.from_defaults(query_engine=nl_query_engine, ...)
  query_engine_tool_2 = QueryEngineTool.from_defaults(query_engine=query_engine, ...)
  ```

### Agent Setup
- Used `FunctionAgent` to combine tools and LLM:
  ```python
  agent = FunctionAgent(tools=[query_engine_tool_1,query_engine_tool_2], llm=OpenAI(model="gpt-4o"))
  ctx = Context(agent)
  ```

### Running Queries via Agent
- Example of running a query and streaming results:
  ```python
  handler = agent.run("Give the details for the train no 108?", ctx=ctx)
  async for ev in handler.stream_events():
      # ... handle events ...
  response = await handler
  ```

---

## 5. Evaluation
- Used `RelevancyEvaluator` to evaluate responses from the annual report query engine:
  ```python
  from llama_index.core.evaluation import RelevancyEvaluator
  eval_result = evaluator.evaluate_response(query=query, response=response)
  ```

---

## 6. Summary
- Demonstrates a full RAG pipeline: ingesting structured/unstructured data, building vector and JSON-based query engines, constructing agents, and evaluating results.
- Modular and extensible for other domains or data sources.

---

## 7. References
- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [OpenAI API](https://platform.openai.com/docs/)
- [Project GitHub Repo](https://github.com/ayazroomy/Semantic_RAG_Project_Indian_Railways)
