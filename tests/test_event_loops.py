"""PropertyGraphIndex and API clients whose async connections outlive an event loop.

LlamaIndex runs entity extraction in its own ``asyncio.run()`` on every build and insert. API
clients built on ``httpx`` (OpenAI, Mistral, Anthropic, Ollama) keep a pooled connection from the
first event loop, so a later ``index.insert()`` with the same LLM object fails with
``RuntimeError: Event loop is closed``. The README's workaround is to insert through an index
from ``PropertyGraphIndex.from_existing()`` with a new LLM object. These tests use a fake LLM that
calls a local HTTP server through a pooled ``httpx.AsyncClient``, as a real client does. If a
LlamaIndex release stops failing, ``test_insert_with_the_same_llm_fails`` says so, and the README
note can go.
"""

from __future__ import annotations

import gc
import http.server
import threading
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
from llama_index.core import Document, PropertyGraphIndex
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.indices.property_graph import SimpleLLMPathExtractor
from llama_index.core.llms import CompletionResponse, CompletionResponseGen, CustomLLM, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback
from pydantic import PrivateAttr

from grafeo_llamaindex import GrafeoPropertyGraphStore

DIMS = 8
TRIPLETS = {
    "Alice works at Acme.": "(Alice, works at, Acme)",
    "Bob likes tea.": "(Bob, likes, Tea)",
    "Carol knows Bob.": "(Carol, knows, Bob)",
}


class _Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"  # keep-alive, so the client pools the connection

    def do_GET(self) -> None:
        body = b"ok"
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        pass


# A client whose connection belongs to a closed event loop cannot close it; its sockets are
# reclaimed by the garbage collector with a ResourceWarning, which is the behavior under test.
pytestmark = pytest.mark.filterwarnings("ignore::ResourceWarning", "ignore::pytest.PytestUnraisableExceptionWarning")


@pytest.fixture(autouse=True)
def _collect_stranded_connections() -> Iterator[None]:
    """Reclaim the stranded sockets inside these tests, so their warnings do not land on later tests."""
    yield
    gc.collect()


@pytest.fixture(scope="module")
def server_url() -> Iterator[str]:
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/"
    server.shutdown()


class _PooledHTTPLLM(CustomLLM):
    """Answers with fixed triplets, after an async request through one pooled httpx client."""

    url: str
    _client: httpx.AsyncClient = PrivateAttr(default_factory=httpx.AsyncClient)

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata()

    def _answer(self, prompt: str) -> str:
        return "\n".join(triplet for text, triplet in TRIPLETS.items() if text in prompt)

    @llm_completion_callback()
    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        return CompletionResponse(text=self._answer(prompt))

    @llm_completion_callback()
    async def acomplete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        await self._client.get(self.url)
        return CompletionResponse(text=self._answer(prompt))

    @llm_completion_callback()
    def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponseGen:
        yield self.complete(prompt, formatted=formatted, **kwargs)


def _build(store: GrafeoPropertyGraphStore, llm: _PooledHTTPLLM) -> PropertyGraphIndex:
    return PropertyGraphIndex.from_documents(
        [Document(text="Alice works at Acme.")],
        property_graph_store=store,
        kg_extractors=[SimpleLLMPathExtractor(llm=llm)],
        embed_model=MockEmbedding(embed_dim=DIMS),
        use_async=False,
        show_progress=False,
    )


def _inserting_index(store: GrafeoPropertyGraphStore, llm: _PooledHTTPLLM) -> PropertyGraphIndex:
    """The README's pattern: an index over the same store, with a new LLM object for extraction."""
    return PropertyGraphIndex.from_existing(
        property_graph_store=store,
        kg_extractors=[SimpleLLMPathExtractor(llm=llm)],
        embed_model=MockEmbedding(embed_dim=DIMS),
        embed_kg_nodes=True,
        use_async=False,
    )


def _triplets(store: GrafeoPropertyGraphStore) -> set[tuple[str, str, str]]:
    return {(t[0].id, t[1].label, t[2].id) for t in store.get_triplets()}


class TestInsertAfterBuild:
    def test_insert_with_the_same_llm_fails(self, server_url: str) -> None:
        """The LlamaIndex behavior the README works around (llama-index-core 0.14)."""
        llm = _PooledHTTPLLM(url=server_url)
        index = _build(GrafeoPropertyGraphStore(embedding_dimensions=DIMS), llm)
        with pytest.raises(RuntimeError, match="Event loop is closed"):
            index.insert(Document(text="Bob likes tea."))

    def test_insert_through_from_existing_with_a_new_llm(self, server_url: str) -> None:
        store = GrafeoPropertyGraphStore(embedding_dimensions=DIMS)
        _build(store, _PooledHTTPLLM(url=server_url))

        _inserting_index(store, _PooledHTTPLLM(url=server_url)).insert(Document(text="Bob likes tea."))
        _inserting_index(store, _PooledHTTPLLM(url=server_url)).insert(Document(text="Carol knows Bob."))

        assert _triplets(store) == {("Alice", "WORKS_AT", "Acme"), ("Bob", "LIKES", "Tea"), ("Carol", "KNOWS", "Bob")}
