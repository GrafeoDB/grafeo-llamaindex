"""GrafeoDB-backed PropertyGraphStore for LlamaIndex."""

from __future__ import annotations

import contextlib
from collections.abc import Sequence
from typing import Any

import grafeo
from llama_index.core.graph_stores.types import (
    ChunkNode,
    EntityNode,
    LabelledNode,
    PropertyGraphStore,
    Relation,
    Triplet,
)
from llama_index.core.vector_stores.types import VectorStoreQuery

_ID_KEY = "li_id"
_NAME_KEY = "name"
_TEXT_KEY = "text"
_LABEL_KEY = "li_label"
_EMBEDDING_KEY = "embedding"


def _escape(value: str) -> str:
    """Escape single quotes and backslashes for safe GQL string interpolation."""
    return value.replace("\\", "\\\\").replace("'", "\\'")


def _validate_key(key: str) -> str:
    """Validate that a property key is a legal identifier."""
    if not key.isidentifier():
        raise ValueError(f"Invalid property key: {key!r}")
    return key


class GrafeoPropertyGraphStore(PropertyGraphStore):
    """PropertyGraphStore backed by GrafeoDB.

    Stores LlamaIndex's LabelledNode/Relation objects as native Grafeo nodes and edges.
    Supports both structured queries (GQL/Cypher) and vector similarity search via
    Grafeo's built-in HNSW index.
    """

    supports_structured_queries: bool = True
    supports_vector_queries: bool = True

    def __init__(
        self,
        *,
        db_path: str | None = None,
        embedding_dimensions: int = 1536,
        embedding_metric: str = "cosine",
    ) -> None:
        self._db_path = db_path
        self._embedding_dimensions = embedding_dimensions
        self._embedding_metric = embedding_metric

        self._db = grafeo.GrafeoDB(db_path) if db_path else grafeo.GrafeoDB()

        if not self._db.has_property_index(_ID_KEY):
            self._db.create_property_index(_ID_KEY)
        if not self._db.has_property_index(_NAME_KEY):
            self._db.create_property_index(_NAME_KEY)

        self._vector_indexed_labels: set[str] = set()
        self._schema_cache: dict | None = None

        # Write-through caches: LlamaIndex ID/name → Grafeo node ID.
        # Populated on upsert, lazy-filled on lookup miss, evicted on delete.
        self._id_cache: dict[str, int] = {}
        self._name_cache: dict[str, int] = {}

    @property
    def client(self) -> grafeo.GrafeoDB:
        """Return the underlying GrafeoDB instance."""
        return self._db

    # ─── Internal helpers ─────────────────────────────────────────────────────────────────────────

    def _find_node_id(self, li_id: str) -> int | None:
        if li_id in self._id_cache:
            return self._id_cache[li_id]
        matches = self._db.find_nodes_by_property(_ID_KEY, li_id)
        if matches:
            self._id_cache[li_id] = matches[0]
            return matches[0]
        return None

    def _find_node_by_name(self, name: str) -> int | None:
        if name in self._name_cache:
            return self._name_cache[name]
        matches = self._db.find_nodes_by_property(_NAME_KEY, name)
        if matches:
            self._name_cache[name] = matches[0]
            return matches[0]
        return None

    def _grafeo_node_to_labelled(self, node_id: int) -> LabelledNode | None:
        node = self._db.get_node(node_id)
        if node is None:
            return None

        props = node.properties()
        li_label = props.pop(_LABEL_KEY, "entity")
        li_id = props.pop(_ID_KEY, str(node_id))
        embedding = props.pop(_EMBEDDING_KEY, None)

        if li_label == "text_chunk":
            text = props.pop(_TEXT_KEY, "")
            return ChunkNode(id_=li_id, text=text, label=li_label, properties=props, embedding=embedding)

        name = props.pop(_NAME_KEY, li_id)
        return EntityNode(name=name, label=li_label, properties=props, embedding=embedding)

    def _ensure_vector_index(self, labels: set[str]) -> None:
        new_labels = labels - self._vector_indexed_labels
        if not new_labels:
            return
        for label in new_labels:
            with contextlib.suppress(RuntimeError):  # index may already exist
                self._db.create_vector_index(
                    label=label,
                    property=_EMBEDDING_KEY,
                    dimensions=self._embedding_dimensions,
                    metric=self._embedding_metric,
                )
            self._vector_indexed_labels.add(label)

    def _distance_to_score(self, distance: float) -> float:
        match self._embedding_metric:
            case "cosine":
                return 1.0 - distance
            case "euclidean" | "manhattan":
                return 1.0 / (1.0 + distance)
            case "dot_product":
                return -distance
            case _:
                return 1.0 - distance

    # ─── PropertyGraphStore abstract methods ──────────────────────────────────────────────────────

    def get(self, properties: dict | None = None, ids: list[str] | None = None) -> list[LabelledNode]:
        """Get nodes matching properties or IDs."""
        results: list[LabelledNode] = []

        if ids:
            for li_id in ids:
                gid = self._find_node_id(li_id)
                if gid is not None:
                    node = self._grafeo_node_to_labelled(gid)
                    if node:
                        results.append(node)
            return results

        if properties:
            conditions = []
            for key, value in properties.items():
                if isinstance(value, str):
                    conditions.append(f"n.{_validate_key(key)} = '{_escape(value)}'")
                else:
                    conditions.append(f"n.{_validate_key(key)} = {value}")
            where_clause = " AND ".join(conditions)
            query = f"MATCH (n) WHERE {where_clause} RETURN n"
            for node in self._db.execute(query).nodes():
                labelled = self._grafeo_node_to_labelled(node.id)
                if labelled:
                    results.append(labelled)
            return results

        for node in self._db.execute("MATCH (n) RETURN n").nodes():
            labelled = self._grafeo_node_to_labelled(node.id)
            if labelled:
                results.append(labelled)
        return results

    def get_triplets(
        self,
        entity_names: list[str] | None = None,
        relation_names: list[str] | None = None,
        properties: dict | None = None,
        ids: list[str] | None = None,
        limit: int = 500,
    ) -> list[Triplet]:
        """Get triplets (source, relation, target) matching filters."""
        where_parts: list[str] = []
        if entity_names:
            names_list = ", ".join(f"'{_escape(n)}'" for n in entity_names)
            where_parts.append(f"(src.name IN [{names_list}] OR tgt.name IN [{names_list}])")
        if ids:
            ids_list = ", ".join(f"'{_escape(i)}'" for i in ids)
            where_parts.append(f"(src.{_ID_KEY} IN [{ids_list}] OR tgt.{_ID_KEY} IN [{ids_list}])")

        where_clause = f" WHERE {' AND '.join(where_parts)}" if where_parts else ""

        # Use edge type in MATCH pattern for efficient filtering; fall back to untyped match.
        edge_types = relation_names if relation_names else [None]
        triplets: list[Triplet] = []
        for edge_type in edge_types:
            rel_pattern = f":{edge_type}" if edge_type else ""
            query = f"MATCH (src)-[rel{rel_pattern}]->(tgt){where_clause} RETURN src, rel, tgt LIMIT {limit}"
            for row in self._db.execute(query):
                edge = self._db.get_edge(row["rel"])
                src_node = self._grafeo_node_to_labelled(row["src"])
                tgt_node = self._grafeo_node_to_labelled(row["tgt"])
                if src_node and tgt_node:
                    triplets.append(
                        (
                            src_node,
                            Relation(
                                label=edge.edge_type,
                                source_id=src_node.id,
                                target_id=tgt_node.id,
                                properties=edge.properties(),
                            ),
                            tgt_node,
                        )
                    )
        return triplets

    def get_rel_map(
        self,
        graph_nodes: list[LabelledNode],
        depth: int = 2,
        limit: int = 30,
        ignore_rels: list[str] | None = None,
    ) -> list[Triplet]:
        """Get depth-aware relationship map around the given nodes via batched BFS."""
        if not graph_nodes:
            return []

        triplets: list[Triplet] = []
        seen_edges: set[str] = set()

        # Build initial frontier: Grafeo node ID → LlamaIndex ID
        frontier: dict[int, str] = {}
        for node in graph_nodes:
            gid = self._find_node_id(node.id)
            if gid is not None:
                frontier[gid] = node.id
        visited = set(frontier)

        for _ in range(depth):
            if not frontier or len(triplets) >= limit:
                break

            # Batch all frontier nodes into a single IN query
            ids_list = ", ".join(f"'{_escape(li_id)}'" for li_id in frontier.values())
            query = f"MATCH (src)-[rel]->(tgt) WHERE src.{_ID_KEY} IN [{ids_list}] RETURN src, rel, tgt"

            next_frontier: dict[int, str] = {}
            for row in self._db.execute(query):
                edge = self._db.get_edge(row["rel"])
                if ignore_rels and edge.edge_type in ignore_rels:
                    continue

                src_id, tgt_id = row["src"], row["tgt"]
                edge_key = f"{src_id}-{edge.edge_type}-{tgt_id}"
                if edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)

                src_node = self._grafeo_node_to_labelled(src_id)
                tgt_node = self._grafeo_node_to_labelled(tgt_id)
                if src_node and tgt_node:
                    triplets.append(
                        (
                            src_node,
                            Relation(
                                label=edge.edge_type,
                                source_id=src_node.id,
                                target_id=tgt_node.id,
                                properties=edge.properties(),
                            ),
                            tgt_node,
                        )
                    )

                if tgt_id not in visited:
                    visited.add(tgt_id)
                    tgt_obj = self._db.get_node(tgt_id)
                    if tgt_obj:
                        tgt_li_id = tgt_obj.properties().get(_ID_KEY, "")
                        if tgt_li_id:
                            next_frontier[tgt_id] = tgt_li_id

            frontier = next_frontier

        return triplets[:limit]

    def upsert_nodes(self, nodes: Sequence[LabelledNode]) -> None:
        """Insert or update nodes in the graph."""
        embedded_labels: set[str] = set()

        for node in nodes:
            existing_gid = self._find_node_id(node.id)

            props: dict[str, Any] = {_ID_KEY: node.id, _LABEL_KEY: node.label, **node.properties}

            if isinstance(node, EntityNode):
                props[_NAME_KEY] = node.name
            elif isinstance(node, ChunkNode):
                props[_TEXT_KEY] = node.text

            grafeo_label = node.label.replace(" ", "_").replace("-", "_")
            if not grafeo_label.isidentifier():
                grafeo_label = "node"

            if node.embedding:
                props[_EMBEDDING_KEY] = node.embedding
                embedded_labels.add(grafeo_label)

            if existing_gid is not None:
                for key, value in props.items():
                    self._db.set_node_property(existing_gid, key, value)
                gid = existing_gid
            else:
                created = self._db.create_node(labels=[grafeo_label], properties=props)
                gid = created.id

            self._id_cache[node.id] = gid
            if isinstance(node, EntityNode):
                self._name_cache[node.name] = gid

        if embedded_labels:
            self._ensure_vector_index(embedded_labels)

        self._schema_cache = None

    def upsert_relations(self, relations: list[Relation]) -> None:
        """Insert or update relations (edges) in the graph."""
        for rel in relations:
            source_gid = self._find_node_by_name(rel.source_id) or self._find_node_id(rel.source_id)
            target_gid = self._find_node_by_name(rel.target_id) or self._find_node_id(rel.target_id)

            if source_gid is None or target_gid is None:
                continue

            edge_type = rel.label.replace(" ", "_").replace("-", "_").upper()
            self._db.create_edge(
                source_id=source_gid,
                target_id=target_gid,
                edge_type=edge_type,
                properties=rel.properties,
            )

        self._schema_cache = None

    def delete(
        self,
        entity_names: list[str] | None = None,
        relation_names: list[str] | None = None,
        properties: dict | None = None,
        ids: list[str] | None = None,
    ) -> None:
        """Delete matching nodes and/or relations."""
        if ids:
            for li_id in ids:
                gid = self._find_node_id(li_id)
                if gid is not None:
                    self._db.delete_node(gid)
                    self._id_cache.pop(li_id, None)

        if entity_names:
            for name in entity_names:
                gid = self._find_node_by_name(name)
                if gid is not None:
                    self._db.delete_node(gid)
                    self._name_cache.pop(name, None)

        if relation_names:
            for rel_name in relation_names:
                edge_type = rel_name.replace(" ", "_").replace("-", "_").upper()
                self._db.execute(f"MATCH (a)-[r:{edge_type}]->(b) DELETE r")

        self._schema_cache = None

    def structured_query(self, query: str, param_map: dict[str, Any] | None = None) -> Any:
        """Execute a raw GQL/Cypher query against the graph."""
        if query.strip().lower().startswith("g."):
            return list(self._db.execute_gremlin(query))
        return list(self._db.execute(query))

    def vector_query(self, query: VectorStoreQuery, **kwargs: Any) -> tuple[list[LabelledNode], list[float]]:
        """Perform vector similarity search using Grafeo's HNSW index."""
        if query.query_embedding is None:
            return [], []

        k = query.similarity_top_k

        all_results: list[tuple[int, float]] = []
        for label in self._vector_indexed_labels:
            try:
                results = self._db.vector_search(
                    label=label,
                    property=_EMBEDDING_KEY,
                    query=query.query_embedding,
                    k=k,
                )
                all_results.extend(results)
            except RuntimeError:
                continue  # no vectors indexed for this label yet

        all_results.sort(key=lambda x: x[1])
        top_results = all_results[:k]

        nodes: list[LabelledNode] = []
        scores: list[float] = []
        for node_id, distance in top_results:
            labelled = self._grafeo_node_to_labelled(node_id)
            if labelled:
                nodes.append(labelled)
                scores.append(self._distance_to_score(distance))

        return nodes, scores

    # ─── Overridden concrete methods ──────────────────────────────────────────────────────────────

    def get_schema(self, refresh: bool = False) -> dict:
        """Return the graph schema from Grafeo."""
        if self._schema_cache is None or refresh:
            self._schema_cache = self._db.schema()
        return self._schema_cache

    def get_schema_str(self, refresh: bool = False) -> str:
        """Return graph schema as a formatted string for LLM prompts."""
        schema = self.get_schema(refresh)
        labels = [entry["name"] for entry in schema.get("labels", [])]
        edge_types = [entry["name"] for entry in schema.get("edge_types", [])]
        property_keys = schema.get("property_keys", [])
        return "\n".join(
            [
                f"Node labels: {', '.join(labels)}",
                f"Edge types: {', '.join(edge_types)}",
                f"Property keys: {', '.join(property_keys)}",
            ]
        )

    def persist(self, persist_path: str, fs: Any = None) -> None:
        """Save the database to disk."""
        if not self._db.is_persistent:
            self._db.save(persist_path)

    def close(self) -> None:
        """Close the database connection."""
        self._db.close()

    def __enter__(self) -> GrafeoPropertyGraphStore:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
