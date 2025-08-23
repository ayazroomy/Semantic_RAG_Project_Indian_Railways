import pytest
from fastapi.testclient import TestClient
from main import app, extract_tables_to_documents, getQueryEngines, getAgents, runQuery
import os
import tempfile
import json
import asyncio

client = TestClient(app)

def test_ask_train_info(monkeypatch):
    # Mock the router.query method to return a known value
    class DummyRouter:
        def query(self, query):
            return "Train info for 107"
    monkeypatch.setattr("main.RouterQueryEngine", lambda **kwargs: DummyRouter())
    response = client.post("/ask", params={"query": "provide details of train 107"})
    assert response.status_code == 200
    assert "Train info for 107" in response.json().get("answer", "")

def test_ask_annual_report(monkeypatch):
    class DummyRouter:
        def query(self, query):
            return "Annual report summary"
    monkeypatch.setattr("main.RouterQueryEngine", lambda **kwargs: DummyRouter())
    response = client.post("/ask", params={"query": "Summarize the key achievements in the 2023-24 annual report"})
    assert response.status_code == 200
    assert "Annual report summary" in response.json().get("answer", "")

def test_extract_tables_to_documents_empty():
    # Should return an empty list for a non-existent or empty PDF
    docs = extract_tables_to_documents("non_existent.pdf")
    assert isinstance(docs, list)

def test_get_query_engines():
    # Should return two engines
    engines = getQueryEngines()
    assert isinstance(engines, tuple)
    assert len(engines) == 2

def test_get_agents():
    # Should return two tools
    tools = getAgents()
    assert isinstance(tools, tuple)
    assert len(tools) == 2

def test_run_query_success(monkeypatch):
    class DummyAgent:
        async def run(self, query, ctx=None):
            return "dummy result"
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(runQuery("test query", DummyAgent(), None))
    assert result == "dummy result"

def test_run_query_error(monkeypatch):
    class DummyAgent:
        async def run(self, query, ctx=None):
            raise Exception("fail")
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(runQuery("test query", DummyAgent(), None))
    assert result == "fail"
