"""Grafeo-optimized Property Graph retriever with PageRank reranking."""

from __future__ import annotations

import asyncio
from typing import Any

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.indices.property_graph.sub_retrievers.base import BasePGRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.settings import Settings
from llama_index.core.vector_stores.types import VectorStoreQuery

from grafeo_llamaindex.property_graph_store import GrafeoPropertyGraphStore


class GrafeoPGRetriever(BasePGRetriever):
    """Property Graph retriever that leverages Grafeo's native vector search and graph algorithms.

    Combines vector similarity search with multi-hop graph expansion and optional
    PageRank-weighted reranking — all within a single embedded Grafeo engine.

    Compared to LlamaIndex's built-in ``VectorContextRetriever``, this retriever
    adds PageRank blending: ``final_score = alpha * similarity + (1 - alpha) * pagerank``.
    """

    def __init__(
        self,
        graph_store: GrafeoPropertyGraphStore,
        *,
        embed_model: BaseEmbedding | None = None,
        similarity_top_k: int = 10,
        path_depth: int = 1,
        limit: int = 30,
        alpha: float = 0.7,
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

        super().__init__(
            graph_store=graph_store,
            include_text=include_text,
            include_properties=include_properties,
            **kwargs,
        )

    def _compute_pagerank_scores(self) -> dict[str, float]:
        """Run PageRank on the underlying Grafeo DB and map node IDs to normalised scores."""
        db = self._graph_store.client
        raw = db.algorithms.pagerank()
        if not raw:
            return {}

        # raw is dict[int, float] mapping Grafeo internal ID → score
        max_score = max(raw.values()) or 1.0
        result: dict[str, float] = {}
        for gid, score in raw.items():
            node = db.get_node(gid)
            if node is None:
                continue
            li_id = node.properties().get("li_id", str(gid))
            result[li_id] = score / max_score  # normalise to [0, 1]
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

        # 3. Expand via graph traversal
        triplets = self._graph_store.get_rel_map(
            kg_nodes,
            depth=self._path_depth,
            limit=self._limit,
        )
        if not triplets:
            return []

        # 4. Score each triplet
        pagerank_map = self._compute_pagerank_scores() if self._use_pagerank else {}

        scores: list[float] = []
        for triplet in triplets:
            src_sim = sim_map.get(triplet[0].id, 0.0)
            tgt_sim = sim_map.get(triplet[2].id, 0.0)
            similarity = max(src_sim, tgt_sim)

            if self._use_pagerank and pagerank_map:
                src_pr = pagerank_map.get(triplet[0].id, 0.0)
                tgt_pr = pagerank_map.get(triplet[2].id, 0.0)
                pr = max(src_pr, tgt_pr)
                final = self._alpha * similarity + (1.0 - self._alpha) * pr
            else:
                final = similarity

            scores.append(final)

        # 5. Sort by score descending
        ranked = sorted(
            zip(triplets, scores, strict=True),
            key=lambda x: x[1],
            reverse=True,
        )

        return self._get_nodes_with_score(
            [t for t, _s in ranked],
            [s for _t, s in ranked],
        )

    async def aretrieve_from_graph(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        return await asyncio.to_thread(self.retrieve_from_graph, query_bundle)
