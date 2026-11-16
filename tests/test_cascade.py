"""Deleting a chunk deletes what was extracted only from it.

Entities and relations record the chunks they were extracted from (``li_sources``, from LlamaIndex's
``triplet_source_id``). Deleting a chunk deletes the relations no other chunk supports, then the
entities left with no source and no relation, so ``index.delete_ref_doc()`` removes a document's
knowledge without touching what other documents also mention.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from llama_index.core import Document, PropertyGraphIndex
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.graph_stores.types import (
    KG_NODES_KEY,
    KG_RELATIONS_KEY,
    TRIPLET_SOURCE_KEY,
    ChunkNode,
    EntityNode,
    Relation,
)
from llama_index.core.llms import MockLLM
from llama_index.core.schema import BaseNode, TransformComponent

from grafeo_llamaindex import GrafeoPropertyGraphStore

# What each document's text is extracted to: (source, relation, target) triplets.
EXTRACTIONS = {
    "Scrooge and Marley were partners.": [("Scrooge", "PARTNER_OF", "Marley")],
    "Scrooge employs Cratchit.": [("Scrooge", "EMPLOYS", "Cratchit")],
}


class _Extractor(TransformComponent):
    """Stands in for an LLM path extractor, with fixed triplets per text."""

    def __call__(self, nodes: Sequence[BaseNode], **kwargs: Any) -> Sequence[BaseNode]:
        for node in nodes:
            triplets = EXTRACTIONS[node.get_content()]
            names = {name for s, _, t in triplets for name in (s, t)}
            node.metadata[KG_NODES_KEY] = [EntityNode(name=name, label="person") for name in sorted(names)]
            node.metadata[KG_RELATIONS_KEY] = [Relation(label=r, source_id=s, target_id=t) for s, r, t in triplets]
        return nodes


def _triplets(store: GrafeoPropertyGraphStore) -> set[tuple[str, str, str]]:
    return {(t[0].id, t[1].label, t[2].id) for t in store.get_triplets()}


def _entities(store: GrafeoPropertyGraphStore) -> set[str]:
    return {n.id for n in store.get() if isinstance(n, EntityNode)}


def _chunk(store: GrafeoPropertyGraphStore, chunk_id: str) -> None:
    store.upsert_nodes([ChunkNode(text=chunk_id, id_=chunk_id)])


def _extract(store: GrafeoPropertyGraphStore, chunk_id: str, *triplets: tuple[str, str, str]) -> None:
    """Upsert triplets as PropertyGraphIndex does: entities it already holds are not upserted again."""
    source = {TRIPLET_SOURCE_KEY: chunk_id}
    names = sorted({name for s, _, t in triplets for name in (s, t)})
    existing = {n.id for n in store.get(ids=names)}
    store.upsert_nodes(
        [EntityNode(name=n, label="person", properties=dict(source)) for n in names if n not in existing]
    )
    store.upsert_relations(
        [Relation(label=r, source_id=s, target_id=t, properties=dict(source)) for s, r, t in triplets]
    )


class TestCascade:
    def test_delete_chunk_keeps_what_another_chunk_supports(self, store: GrafeoPropertyGraphStore) -> None:
        _chunk(store, "c1")
        _chunk(store, "c2")
        _extract(store, "c1", ("Scrooge", "PARTNER_OF", "Marley"))
        _extract(store, "c2", ("Scrooge", "EMPLOYS", "Cratchit"))

        store.delete(ids=["c1"])
        assert _triplets(store) == {("Scrooge", "EMPLOYS", "Cratchit")}
        assert _entities(store) == {"Scrooge", "Cratchit"}  # Scrooge came from c1 but c2 relates him

        store.delete(ids=["c2"])
        assert _triplets(store) == set()
        assert _entities(store) == set()

    def test_relation_from_two_chunks_survives_one(self, store: GrafeoPropertyGraphStore) -> None:
        _chunk(store, "c1")
        _chunk(store, "c2")
        _extract(store, "c1", ("Scrooge", "EMPLOYS", "Cratchit"))
        _extract(store, "c2", ("Scrooge", "EMPLOYS", "Cratchit"))

        store.delete(ids=["c1"])
        [(_src, relation, _tgt)] = store.get_triplets()
        # LlamaIndex reads a triplet's source text through triplet_source_id: it moves to a live chunk.
        assert relation.properties[TRIPLET_SOURCE_KEY] == "c2"
        assert "li_sources" not in relation.properties
        # Scrooge came from c1 only (PropertyGraphIndex does not upsert a known entity again): no
        # pointer to the deleted chunk is left, and the relation from c2 keeps him.
        assert TRIPLET_SOURCE_KEY not in store.get(ids=["Scrooge"])[0].properties

    def test_deleting_the_latest_source_moves_triplet_source_back(self, store: GrafeoPropertyGraphStore) -> None:
        _chunk(store, "c1")
        _chunk(store, "c2")
        for chunk_id in ("c1", "c2"):  # upserted from both chunks, so triplet_source_id is c2
            source = {TRIPLET_SOURCE_KEY: chunk_id}
            store.upsert_nodes([EntityNode(name=n, label="person", properties=dict(source)) for n in ("A", "B")])
            store.upsert_relations([Relation(label="KNOWS", source_id="A", target_id="B", properties=dict(source))])

        store.delete(ids=["c2"])
        [(src, relation, _tgt)] = store.get_triplets()
        assert relation.properties[TRIPLET_SOURCE_KEY] == "c1"
        assert src.properties[TRIPLET_SOURCE_KEY] == "c1"
        assert _entities(store) == {"A", "B"}

    def test_new_source_on_data_written_without_sources(self, store: GrafeoPropertyGraphStore) -> None:
        """A relation written before li_sources existed keeps its triplet_source_id as a source."""
        _chunk(store, "c1")
        _chunk(store, "c2")
        _extract(store, "c1", ("Scrooge", "EMPLOYS", "Cratchit"))
        store.structured_query("MATCH ()-[r]->() REMOVE r.li_sources")
        _extract(store, "c2", ("Scrooge", "EMPLOYS", "Cratchit"))  # sources become [c1, c2]

        store.delete(ids=["c1"])
        assert _triplets(store) == {("Scrooge", "EMPLOYS", "Cratchit")}

    def test_manual_nodes_and_relations_are_kept(self, store: GrafeoPropertyGraphStore) -> None:
        _chunk(store, "c1")
        store.upsert_nodes([EntityNode(name="Scrooge", label="person"), EntityNode(name="London", label="place")])
        _extract(store, "c1", ("Scrooge", "PARTNER_OF", "Marley"))
        store.upsert_relations([Relation(label="LIVES_IN", source_id="Scrooge", target_id="London")])

        store.delete(ids=["c1"])
        assert _triplets(store) == {("Scrooge", "LIVES_IN", "London")}
        assert _entities(store) == {"Scrooge", "London"}  # Marley had only c1

    def test_only_chunk_deletes_cascade(self, store: GrafeoPropertyGraphStore) -> None:
        _chunk(store, "c1")
        _extract(store, "c1", ("Scrooge", "PARTNER_OF", "Marley"))
        store.delete(entity_names=["Scrooge"])
        assert _entities(store) == {"Marley"}

    def test_delete_by_properties_cascades(self, store: GrafeoPropertyGraphStore) -> None:
        store.upsert_nodes([ChunkNode(text="t", id_="c1", properties={"ref_doc_id": "d1"})])
        _extract(store, "c1", ("Scrooge", "PARTNER_OF", "Marley"))
        store.delete(properties={"ref_doc_id": "d1"})
        assert (store.node_count, store.edge_count) == (0, 0)

    def test_data_written_without_sources(self, store: GrafeoPropertyGraphStore) -> None:
        """Data written before li_sources existed falls back to triplet_source_id."""
        _chunk(store, "c1")
        _chunk(store, "c2")
        _extract(store, "c1", ("Scrooge", "PARTNER_OF", "Marley"))
        _extract(store, "c2", ("Scrooge", "EMPLOYS", "Cratchit"))
        store.structured_query("MATCH (n) REMOVE n.li_sources")
        store.structured_query("MATCH ()-[r]->() REMOVE r.li_sources")

        store.delete(ids=["c1"])
        assert _triplets(store) == {("Scrooge", "EMPLOYS", "Cratchit")}
        assert _entities(store) == {"Scrooge", "Cratchit"}


class TestDeleteRefDoc:
    def test_delete_ref_doc_removes_what_only_it_mentions(self) -> None:
        store = GrafeoPropertyGraphStore(embedding_dimensions=8)
        index = PropertyGraphIndex.from_documents(
            [Document(text=text, doc_id=f"doc-{i}") for i, text in enumerate(EXTRACTIONS, start=1)],
            property_graph_store=store,
            kg_extractors=[_Extractor()],
            embed_model=MockEmbedding(embed_dim=8),
            llm=MockLLM(),
            show_progress=False,
        )
        assert _entities(store) == {"Scrooge", "Marley", "Cratchit"}

        index.delete_ref_doc("doc-1")
        assert _triplets(store) == {("Scrooge", "EMPLOYS", "Cratchit")}
        assert _entities(store) == {"Scrooge", "Cratchit"}

        index.delete_ref_doc("doc-2")
        assert store.node_count == 0
