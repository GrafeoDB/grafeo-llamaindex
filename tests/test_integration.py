"""End to end through LlamaIndex's PropertyGraphIndex, with mock models (no API keys)."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from llama_index.core import Document, PropertyGraphIndex
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.graph_stores.types import KG_NODES_KEY, KG_RELATIONS_KEY, EntityNode, Relation
from llama_index.core.indices.property_graph import ImplicitPathExtractor
from llama_index.core.llms import MockLLM
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.schema import BaseNode, TransformComponent
from llama_index.core.vector_stores.types import MetadataFilter, MetadataFilters

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

    def test_build_with_implicit_paths_is_quiet(self) -> None:
        """ImplicitPathExtractor links each chunk to its document by a SOURCE relation, and
        documents are not graph nodes: those relations are skipped without a warning, while the
        chunk-to-chunk NEXT and PREVIOUS relations are stored."""
        store = GrafeoPropertyGraphStore(embedding_dimensions=DIMS)
        text = " ".join(f"Sentence number {i} about Alice and Acme." for i in range(400))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            PropertyGraphIndex.from_documents(
                [Document(text=text, doc_id="doc-1")],
                property_graph_store=store,
                # A token splitter makes the chunks: the default sentence splitter reads NLTK data,
                # which NLTK 3.10 refuses to open when uv installed it as hard links (Linux CI).
                transformations=[TokenTextSplitter(chunk_size=256, chunk_overlap=0)],
                kg_extractors=[ImplicitPathExtractor()],
                embed_model=MockEmbedding(embed_dim=DIMS),
                llm=MockLLM(),
                show_progress=False,
            )
        labels = {t[1].label for t in store.get_triplets()}
        assert labels == {"NEXT", "PREVIOUS"}

    def test_dangling_relation_still_warns(self) -> None:
        store = GrafeoPropertyGraphStore()
        store.upsert_nodes([EntityNode(name="Alice", label="person")])
        with pytest.warns(UserWarning, match="missing target_id='Nobody'"):
            store.upsert_relations([Relation(label="KNOWS", source_id="Alice", target_id="Nobody")])
        with pytest.warns(UserWarning, match="missing source_id='Nobody'"):
            store.upsert_relations([Relation(label="SOURCE", source_id="Nobody", target_id="Alice")])

    def test_delete_ref_doc_after_reopen(self, tmp_path: Path) -> None:
        """index.delete_ref_doc() reads the document's chunks from the index's docstore, which
        from_existing() starts empty; delete_llama_nodes(ref_doc_ids=...) finds them in the graph."""
        path = str(tmp_path / "kg.grafeo")
        with GrafeoPropertyGraphStore(db_path=path, embedding_dimensions=DIMS) as store:
            PropertyGraphIndex.from_documents(
                [
                    Document(text="Alice works at Acme.", doc_id="doc-1"),
                    Document(text="Bob likes tea.", doc_id="doc-2"),
                ],
                property_graph_store=store,
                kg_extractors=[_FixedExtractor()],
                embed_model=MockEmbedding(embed_dim=DIMS),
                llm=MockLLM(),
                show_progress=False,
            )
        with GrafeoPropertyGraphStore(db_path=path, embedding_dimensions=DIMS) as store:
            index = PropertyGraphIndex.from_existing(
                property_graph_store=store, embed_model=MockEmbedding(embed_dim=DIMS), llm=MockLLM()
            )
            index.delete_ref_doc("doc-1")  # the fresh docstore does not know doc-1
            assert store.node_count == 6

            store.delete_llama_nodes(ref_doc_ids=["doc-1"])
            assert _triplets(store) == {("Bob", "LIKES", "Tea")}
            assert store.node_count == 3  # Bob, Tea and doc-2's chunk

    def test_default_retriever_applies_filters(self) -> None:
        """as_retriever(filters=...) reaches VectorContextRetriever, and vector_query applies them.

        MockEmbedding gives every node the same embedding, so only the filter can pick Alice.
        """
        index, _store, _ = _build()
        filters = MetadataFilters(filters=[MetadataFilter(key="name", value="Alice")])
        texts = [r.text for r in index.as_retriever(filters=filters, include_text=False).retrieve("Alice")]
        assert texts == ["Alice -> WORKS_AT -> Acme"]

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
