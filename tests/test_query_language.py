"""The language structured_query() runs: GQL by default, Cypher by store option or per call."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

import grafeo
import pytest
from llama_index.core.graph_stores.types import EntityNode, Relation
from llama_index.core.indices.property_graph import TextToCypherRetriever
from llama_index.core.llms import CompletionResponse, CompletionResponseGen, CustomLLM, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback
from pydantic import Field

from grafeo_llamaindex import GrafeoPropertyGraphStore
from grafeo_llamaindex.property_graph_store import DEFAULT_QUERY_LANGUAGE

# Cypher only: GQL has no pattern predicates in WHERE.
CYPHER_ONLY = "MATCH (p:person) WHERE (p)-[:KNOWS]->() RETURN p.name AS name"


class _CountingDB:
    """Delegates to a GrafeoDB and counts the attributes the store uses."""

    def __init__(self, db: grafeo.GrafeoDB) -> None:
        self._db = db
        self.calls: Counter[str] = Counter()

    def __getattr__(self, name: str) -> Any:
        self.calls[name] += 1
        return getattr(self._db, name)


def _store(**kwargs: Any) -> GrafeoPropertyGraphStore:
    store = GrafeoPropertyGraphStore(**kwargs)
    store.upsert_nodes([EntityNode(name="Alice", label="person"), EntityNode(name="Bob", label="person")])
    store.upsert_relations([Relation(label="KNOWS", source_id="Alice", target_id="Bob")])
    return store


def _executed_with(
    monkeypatch: pytest.MonkeyPatch, store: GrafeoPropertyGraphStore, query: str, **kwargs: Any
) -> list[str]:
    """Run ``query`` and return the GrafeoDB execute methods it went through."""
    counting = _CountingDB(store.client)
    with monkeypatch.context() as patch:
        patch.setattr(store, "_db", counting)
        store.structured_query(query, **kwargs)
    return sorted(name for name in counting.calls if name.startswith("execute"))


class TestLanguageChoice:
    def test_default_is_gql(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert DEFAULT_QUERY_LANGUAGE == "gql"  # 0.3.0 changes it to "cypher"
        store = _store()
        assert store.query_language == DEFAULT_QUERY_LANGUAGE
        assert _executed_with(monkeypatch, store, "MATCH (n) RETURN n.name AS name") == ["execute"]
        with pytest.raises(grafeo.GrafeoError):
            store.structured_query(CYPHER_ONLY)

    def test_store_option(self, monkeypatch: pytest.MonkeyPatch) -> None:
        store = _store(query_language="cypher")
        assert store.query_language == "cypher"
        assert store.structured_query(CYPHER_ONLY) == [{"name": "Alice"}]
        assert _executed_with(monkeypatch, store, CYPHER_ONLY) == ["execute_cypher"]

    def test_per_call_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        gql_store = _store()
        assert gql_store.structured_query(CYPHER_ONLY, language="cypher") == [{"name": "Alice"}]
        assert _executed_with(monkeypatch, gql_store, CYPHER_ONLY, language="cypher") == ["execute_cypher"]

        cypher_store = _store(query_language="cypher")
        assert _executed_with(monkeypatch, cypher_store, "MATCH (n) RETURN n.name AS name", language="gql") == [
            "execute"
        ]

    def test_params_in_cypher(self) -> None:
        store = _store(query_language="cypher")
        rows = store.structured_query("MATCH (p:person {name: $name}) RETURN p.name AS name", {"name": "Bob"})
        assert rows == [{"name": "Bob"}]

    @pytest.mark.parametrize("language", ["gql", "cypher"])
    def test_gremlin_still_detected(self, monkeypatch: pytest.MonkeyPatch, language: str) -> None:
        store = _store(query_language=language)
        assert store.structured_query("g.V().count()") == [{"count": 2}]
        assert _executed_with(monkeypatch, store, "g.V().count()") == ["execute_gremlin"]

    def test_case_insensitive(self) -> None:
        assert _store(query_language="Cypher").query_language == "cypher"

    @pytest.mark.parametrize("language", ["sql", "gremlin", "", None])
    def test_invalid_language(self, language: Any) -> None:
        with pytest.raises(ValueError, match="query language"):
            GrafeoPropertyGraphStore(query_language=language)
        if language is not None:  # None means "the store's language" in a call
            with pytest.raises(ValueError, match="query language"):
                _store().structured_query("MATCH (n) RETURN n", language=language)

    async def test_async_override(self) -> None:
        store = _store(query_language="gql")
        assert await store.astructured_query(CYPHER_ONLY, language="cypher") == [{"name": "Alice"}]
        with pytest.raises(grafeo.GrafeoError):
            await store.astructured_query(CYPHER_ONLY)


class _FixedLLM(CustomLLM):
    """Answers every prompt with the same text, standing in for a text-to-Cypher model."""

    answer: str = ""
    prompts: list[str] = Field(default_factory=list)

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata()

    @llm_completion_callback()
    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        self.prompts.append(prompt)
        return CompletionResponse(text=self.answer)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponseGen:
        yield self.complete(prompt, formatted=formatted, **kwargs)


class TestTextToCypherRetriever:
    def _retrieve(self, store: GrafeoPropertyGraphStore) -> Sequence[Any]:
        llm = _FixedLLM(answer=CYPHER_ONLY)
        retriever = TextToCypherRetriever(store, llm=llm)
        results = retriever.retrieve("Who knows someone?")
        assert "Node labels: person" in llm.prompts[0]  # the schema reached the prompt
        return results

    def test_cypher_store(self) -> None:
        [result] = self._retrieve(_store(query_language="cypher"))
        assert CYPHER_ONLY in result.text
        assert "'name': 'Alice'" in result.text

    def test_gql_store_rejects_cypher(self) -> None:
        with pytest.raises(grafeo.GrafeoError):
            self._retrieve(_store(query_language="gql"))

    async def test_async(self) -> None:
        retriever = TextToCypherRetriever(_store(query_language="cypher"), llm=_FixedLLM(answer=CYPHER_ONLY))
        [result] = await retriever.aretrieve("Who knows someone?")
        assert "'name': 'Alice'" in result.text
