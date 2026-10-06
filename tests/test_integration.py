"""End to end through LlamaIndex's PropertyGraphIndex, with mock models (no API keys)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from llama_index.core import Document, PropertyGraphIndex
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.graph_stores.types import KG_NODES_KEY, KG_RELATIONS_KEY, EntityNode, Relation
from llama_index.core.llms import MockLLM
from llama_index.core.schema import BaseNode, TransformComponent

from grafeo_llamaindex import GrafeoPGRetriever, GrafeoPropertyGraphStore

DIMS = 8


class _FixedExtractor(TransformComponent):
    """Stands in for an LLM path extractor: one fixed triplet per document."""

    def __call__(self, nodes: Sequence[BaseNode], **kwargs: Any) -> Sequence[BaseNode]:
        for node in nodes:
            if "Alice" in node.get_content():
                entities = [EntityNode(name="Alice", label="person"), EntityNode(name="Acme", label="company")]
                relations = [Relation(label="works at", source_id="Alice", target_id="Acme")]
            else:
                entities = [EntityNode(name="Bob", label="person"), EntityNode(name="Tea", label="drink")]
                relations = [Relation(label="likes", source_id="Bob", target_id="Tea")]
            node.metadata[KG_NODES_KEY] = entities
            node.metadata[KG_RELATIONS_KEY] = relations
        return nodes


def _build() -> tuple[PropertyGraphIndex, GrafeoPropertyGraphStore, MockEmbedding]:
    store = GrafeoPropertyGraphStore(embedding_dimensions=DIMS)
    embed_model = MockEmbedding(embed_dim=DIMS)
    index = PropertyGraphIndex.from_documents(
        [Document(text="Alice works at Acme.", doc_id="doc-1"), Document(text="Bob likes tea.", doc_id="doc-2")],
        property_graph_store=store,
        kg_extractors=[_FixedExtractor()],
        embed_model=embed_model,
        llm=MockLLM(),
        embed_kg_nodes=True,
        show_progress=False,
    )
    return index, store, embed_model


def _triplets(store: GrafeoPropertyGraphStore) -> set[tuple[str, str, str]]:
    return {(t[0].id, t[1].label, t[2].id) for t in store.get_triplets()}


class TestPropertyGraphIndex:
    def test_build_and_retrieve(self) -> None:
        index, store, embed_model = _build()
        assert _triplets(store) == {("Alice", "WORKS_AT", "Acme"), ("Bob", "LIKES", "Tea")}
        assert store.node_count == 6  # 2 chunks + 4 entities

        retriever = index.as_retriever(sub_retrievers=[GrafeoPGRetriever(store, embed_model=embed_model)])
        texts = [r.text for r in retriever.retrieve("Who works at Acme?")]
        assert any("Alice -> WORKS_AT -> Acme" in t for t in texts)

    def test_delete_document_and_its_entities(self) -> None:
        """delete_llama_nodes() deletes chunks and the entities extracted from them, which have
        relations: on grafeo 0.5.44 this raised GrafeoError before delete() detached them."""
        index, store, _ = _build()
        [chunk] = store.get(properties={"ref_doc_id": "doc-1"})

        index.delete_ref_doc("doc-1")
        assert store.get(ids=[chunk.id]) == []

        store.delete_llama_nodes(node_ids=[chunk.id])  # entities whose triplet_source_id is the chunk
        assert _triplets(store) == {("Bob", "LIKES", "Tea")}
        assert {n.id for n in store.get()} == {"Bob", "Tea", store.get(properties={"ref_doc_id": "doc-2"})[0].id}
