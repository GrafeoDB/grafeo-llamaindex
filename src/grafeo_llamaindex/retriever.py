"""Grafeo-optimized Property Graph retriever with PageRank reranking."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Sequence
from typing import Any

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.indices.property_graph.sub_retrievers.base import BasePGRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.settings import Settings
from llama_index.core.vector_stores.types import VectorStoreQuery

from grafeo_llamaindex.property_graph_store import GrafeoPropertyGraphStore

# Candidate triplets fetched per triplet returned: they are ranked first, then cut to ``limit``.
_CANDIDATES_PER_RESULT = 4


def _cosine(a: Sequence[float] | None, b: Sequence[float] | None) -> float:
    """Cosine similarity of two vectors; 0.0 when either is missing or zero."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


class GrafeoPGRetriever(BasePGRetriever):
    """Property Graph retriever that leverages Grafeo's native vector search and graph algorithms.

    Finds seed entities by vector search, expands them through the graph, and ranks each triplet
    by how well it matches the query: the mean cosine similarity of the query to the source, the
    relation label and the target (``relevance``), blended with PageRank from Grafeo as a smaller
    weight: ``final_score = alpha * relevance + (1 - alpha) * pagerank``. Scoring the whole triplet,
    relation included, keeps well-connected hubs from outranking the triplet that answers.

    ``limit`` triplets are returned, ranked among ``4 * limit`` candidates from the expansion.
    Relation labels are embedded once each (``"WORKS_AT"`` as ``"works at"``) and cached.
    """

    def __init__(
        self,
        graph_store: GrafeoPropertyGraphStore,
        *,
        embed_model: BaseEmbedding | None = None,
        similarity_top_k: int = 10,
        path_depth: int = 1,
        limit: int = 30,
        alpha: float = 0.9,
        use_pagerank: bool = True,
        include_text: bool = True,
        include_properties: bool = False,
        **kwargs: Any,
    ) -> None:
        self._embed_model = embed_model or Settings.embed_model
        self._similarity_top_k = similarity_top_k
        self._path_depth = path_depth
        self._limit = limit
        self._alpha = alpha
        self._use_pagerank = use_pagerank
        self._grafeo_store = graph_store
        self._label_embeddings: dict[str, list[float]] = {}

        super().__init__(
            graph_store=graph_store,
            include_text=include_text,
            include_properties=include_properties,
            **kwargs,
        )

    def _pagerank_scores(self, node_ids: set[str]) -> dict[str, float]:
        """Normalised PageRank (highest is 1.0) of the given LlamaIndex node IDs.

        Reads the store's cached PageRank and resolves only these nodes, instead of reading every
        node of the graph on every query. Empty when the graph has no nodes.
        """
        store = self._grafeo_store
        scores = store._pagerank()
        if not scores:
            return {}
        result: dict[str, float] = {}
        for li_id in node_ids:
            gid = store._find_node_id(li_id)
            if gid is None and li_id.isdigit():
                gid = int(li_id)  # a node written without a LlamaIndex ID is named by its Grafeo ID
            result[li_id] = scores.get(gid, 0.0) if gid is not None else 0.0
        return result

    def retrieve_from_graph(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        # 1. Embed the query
        if query_bundle.embedding is None:
            query_bundle.embedding = self._embed_model.get_agg_embedding_from_queries(query_bundle.embedding_strs)

        # 2. Vector search → seed nodes + similarity scores
        vsq = VectorStoreQuery(
            query_embedding=query_bundle.embedding,
            similarity_top_k=self._similarity_top_k,
        )
        kg_nodes, sim_scores = self._graph_store.vector_query(vsq)
        if not kg_nodes:
            return []

        sim_map: dict[str, float] = {}
        for node, score in zip(kg_nodes, sim_scores, strict=True):
            sim_map[node.id] = score

        # 3. Expand via graph traversal: more candidates than returned, ranked below
        triplets = self._graph_store.get_rel_map(
            kg_nodes,
            depth=self._path_depth,
            limit=self._limit * _CANDIDATES_PER_RESULT,
        )
        if not triplets:
            return []

        # 4. Score each triplet: its relevance to the query, blended with PageRank
        query_embedding = query_bundle.embedding
        label_embeddings = self._embed_labels({t[1].label for t in triplets})

        def node_similarity(node: Any) -> float:
            # A node without an embedding counts with its seed score, if it is a seed.
            if node.embedding:
                return _cosine(query_embedding, node.embedding)
            return sim_map.get(node.id, 0.0)

        pagerank_map: dict[str, float] = {}
        if self._use_pagerank:
            pagerank_map = self._pagerank_scores({t[0].id for t in triplets} | {t[2].id for t in triplets})

        scores: list[float] = []
        for src, rel, tgt in triplets:
            relevance = (
                node_similarity(src) + _cosine(query_embedding, label_embeddings[rel.label]) + node_similarity(tgt)
            ) / 3
            if self._use_pagerank and pagerank_map:
                pr = max(pagerank_map.get(src.id, 0.0), pagerank_map.get(tgt.id, 0.0))
                scores.append(self._alpha * relevance + (1.0 - self._alpha) * pr)
            else:
                scores.append(relevance)

        # 5. Sort by score descending, then keep ``limit``
        ranked = sorted(zip(triplets, scores, strict=True), key=lambda x: x[1], reverse=True)[: self._limit]

        return self._get_nodes_with_score(
            [t for t, _s in ranked],
            [s for _t, s in ranked],
        )

    def _embed_labels(self, labels: set[str]) -> dict[str, list[float]]:
        """Embeddings of relation labels (``"WORKS_AT"`` as ``"works at"``), each computed once."""
        missing = sorted(labels - self._label_embeddings.keys())
        if missing:
            texts = [label.replace("_", " ").lower() for label in missing]
            for label, embedding in zip(missing, self._embed_model.get_text_embedding_batch(texts), strict=True):
                self._label_embeddings[label] = embedding
        return self._label_embeddings

    async def aretrieve_from_graph(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        return await asyncio.to_thread(self.retrieve_from_graph, query_bundle)
