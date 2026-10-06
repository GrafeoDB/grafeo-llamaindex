"""GrafeoDB-backed PropertyGraphStore for LlamaIndex."""

from __future__ import annotations

import threading
import warnings
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
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
_CREATED_AT_KEY = "created_at"
_UPDATED_AT_KEY = "updated_at"

_INTERNAL_EDGE_KEYS = frozenset({_CREATED_AT_KEY, _UPDATED_AT_KEY})

QUERY_LANGUAGES = ("gql", "cypher")
"""Languages ``structured_query`` accepts; a query starting with ``g.`` always runs as Gremlin."""

DEFAULT_QUERY_LANGUAGE = "gql"
"""The language ``structured_query`` uses unless the store or the call names one.

The single place this default is defined. grafeo-llamaindex 0.3.0 changes it to ``"cypher"``, the
language LlamaIndex's ``TextToCypherRetriever`` and ``CypherTemplateRetriever`` generate.
"""

# Every query below passes values as Grafeo query parameters ($name), never as text in the query.
# Only property keys (validated identifiers, quoted) and the integer LIMIT are written into the query.


def _validate_key(key: str) -> str:
    """Validate that a property key is a legal identifier."""
    if not key.isidentifier():
        raise ValueError(f"Invalid property key: {key!r}")
    return key


def _property_ref(var: str, key: str) -> str:
    """Reference property ``key`` of ``var``, quoted so that keywords such as ``group`` work as keys."""
    return f"{var}.`{_validate_key(key)}`"


def _check_query_language(language: str) -> str:
    """Return ``language`` in lower case, or raise ``ValueError`` if it is not a supported language."""
    normalized = language.lower() if isinstance(language, str) else language
    if normalized not in QUERY_LANGUAGES:
        allowed = ", ".join(repr(name) for name in QUERY_LANGUAGES)
        raise ValueError(f"Unsupported query language {language!r}: use one of {allowed}")
    return normalized


def _node_label(label: str) -> str:
    """Return the Grafeo node label a LlamaIndex label is stored under."""
    grafeo_label = label.replace(" ", "_").replace("-", "_")
    return grafeo_label if grafeo_label.isidentifier() else "node"


def _edge_type(label: str) -> str:
    """Return the Grafeo edge type a LlamaIndex relation label is stored under."""
    return label.replace(" ", "_").replace("-", "_").upper()


def _edge_types(labels: Iterable[str]) -> list[str]:
    """Return the edge types to match for relation labels: each label as given and as normalized."""
    return sorted({edge_type for label in labels for edge_type in (label, _edge_type(label))})


def _property_conditions(var: str, properties: dict, params: dict[str, Any]) -> list[str]:
    """Build ``var.key = $param`` conditions, adding the values to ``params``."""
    conditions = []
    for key, value in properties.items():
        param = f"p{len(params)}"
        params[param] = value
        conditions.append(f"{_property_ref(var, key)} = ${param}")
    return conditions


class GrafeoPropertyGraphStore(PropertyGraphStore):
    """PropertyGraphStore backed by GrafeoDB.

    Stores LlamaIndex's LabelledNode/Relation objects as native Grafeo nodes and edges.
    Supports both structured queries (GQL, Cypher or Gremlin) and vector similarity search via
    Grafeo's built-in HNSW index.

    ``query_language`` (``"gql"`` or ``"cypher"``, default ``DEFAULT_QUERY_LANGUAGE``) is the
    language ``structured_query`` runs queries in; use ``"cypher"`` with LlamaIndex's
    text-to-Cypher retrievers.
    """

    supports_structured_queries: bool = True
    supports_vector_queries: bool = True

    def __init__(
        self,
        *,
        db_path: str | None = None,
        embedding_dimensions: int = 1536,
        embedding_metric: str = "cosine",
        dedup_threshold: float | None = None,
        query_language: str = DEFAULT_QUERY_LANGUAGE,
    ) -> None:
        self._query_language = _check_query_language(query_language)
        self._db_path = db_path
        self._embedding_dimensions = embedding_dimensions
        self._embedding_metric = embedding_metric
        self.dedup_threshold = dedup_threshold

        self._db = grafeo.GrafeoDB(db_path) if db_path else grafeo.GrafeoDB()

        if not self._db.has_property_index(_ID_KEY):
            self._db.create_property_index(_ID_KEY)
        if not self._db.has_property_index(_NAME_KEY):
            self._db.create_property_index(_NAME_KEY)

        self._vector_indexed_labels: set[str] = set()
        self._schema_cache: dict | None = None
        # (database epoch, PageRank per Grafeo node ID scaled to a maximum of 1.0)
        self._pagerank_cache: tuple[int, dict[int, float]] | None = None

        # Serializes the store's writes: an upsert looks a node up and then creates it, which must
        # not interleave with another thread doing the same for the same node.
        self._lock = threading.RLock()

        # Write-through caches: LlamaIndex ID/name → Grafeo node ID.
        # Populated on upsert, lazy-filled on lookup miss, evicted on delete. A hit is checked
        # against the database before use (see _lookup).
        self._id_cache: dict[str, int] = {}
        self._name_cache: dict[str, int] = {}

        if db_path:
            self._rediscover_vector_indexes()

    @property
    def client(self) -> grafeo.GrafeoDB:
        """Return the underlying GrafeoDB instance."""
        return self._db

    @property
    def query_language(self) -> str:
        """The language ``structured_query`` uses when a call does not name one."""
        return self._query_language

    # ─── Internal helpers ─────────────────────────────────────────────────────────────────────────

    def _rediscover_vector_indexes(self) -> None:
        """Rebuild vector indexes for the labels of a reopened persistent database.

        A WAL-directory database and a file written by ``save()`` come back without their vector
        indexes, so every label is indexed again with this store's dimensions and metric. A label
        whose stored embeddings have other dimensions cannot be indexed: it is reported with a
        ``UserWarning`` and ``vector_query`` skips it.
        """
        schema = self._db.schema()
        if _EMBEDDING_KEY not in schema.get("property_keys", []):
            return
        for entry in schema.get("labels", []):
            label = entry["name"]
            try:
                self._db.create_vector_index(
                    label=label,
                    property=_EMBEDDING_KEY,
                    dimensions=self._embedding_dimensions,
                    metric=self._embedding_metric,
                )
            except RuntimeError as exc:  # grafeo.GrafeoError is a RuntimeError subclass
                warnings.warn(
                    f"Could not rebuild the vector index for label {label!r}: {exc}. "
                    f"vector_query() skips this label; check that embedding_dimensions "
                    f"({self._embedding_dimensions}) matches the stored embeddings.",
                    UserWarning,
                    stacklevel=3,
                )
                continue
            self._vector_indexed_labels.add(label)

    def _lookup(self, cache: dict[str, int], key: str, value: str) -> int | None:
        """Resolve a LlamaIndex ID or entity name to a Grafeo node ID.

        A cache hit is used only while its node still exists: a node deleted outside the store
        (``structured_query``, ``client``) is looked up again instead of failing the next write.
        """
        gid = cache.get(value)
        if gid is not None:
            if self._db.get_node_labels(gid) is not None:
                return gid
            cache.pop(value, None)
        matches = self._db.find_nodes_by_property(key, value)
        if not matches:
            return None
        cache[value] = matches[0]
        return matches[0]

    def _find_node_id(self, li_id: str) -> int | None:
        return self._lookup(self._id_cache, _ID_KEY, li_id)

    def _find_node_by_name(self, name: str) -> int | None:
        return self._lookup(self._name_cache, _NAME_KEY, name)

    def _resolve_endpoint(self, ref: str) -> int | None:
        """Resolve a relation endpoint, by entity name first and LlamaIndex ID second."""
        gid = self._find_node_by_name(ref)
        return gid if gid is not None else self._find_node_id(ref)

    def _matching_nodes(self, cache: dict[str, int], key: str, value: str) -> set[int]:
        """Return every node whose ``key`` property is ``value``, plus a live cache entry for it."""
        gids = set(self._db.find_nodes_by_property(key, value))
        cached = self._lookup(cache, key, value)  # also covers IDs merged into another node by dedup
        if cached is not None:
            gids.add(cached)
        return gids

    def _find_similar_node(self, label: str, embedding: list[float], threshold: float) -> int | None:
        """Find an existing node whose embedding similarity meets the threshold.

        Returns the Grafeo node ID of the closest match, or None.
        """
        grafeo_label = _node_label(label)

        if grafeo_label not in self._vector_indexed_labels:
            return None

        try:
            results = self._db.vector_search(
                label=grafeo_label,
                property=_EMBEDDING_KEY,
                query=embedding,
                k=1,
            )
        except RuntimeError:
            return None

        if not results:
            return None

        node_id, distance = results[0]
        score = self._distance_to_score(distance)
        if score >= threshold:
            return node_id
        return None

    def _find_existing_edge(self, source_gid: int, target_gid: int, edge_type: str) -> int | None:
        """Find an existing edge of the given type between source and target."""
        rows = self._db.execute(
            "MATCH (a)-[r]->(b) WHERE id(a) = $source AND id(b) = $target AND type(r) = $edge_type "
            "RETURN id(r) AS edge_id LIMIT 1",
            {"source": source_gid, "target": target_gid, "edge_type": edge_type},
        )
        for row in rows:
            return row["edge_id"]
        return None

    def _build_triplet(self, rel_id: int) -> Triplet | None:
        """Build a LlamaIndex triplet, in its stored direction, from a Grafeo edge ID."""
        edge = self._db.get_edge(rel_id)
        if edge is None:
            return None
        src_node = self._grafeo_node_to_labelled(edge.source_id)
        tgt_node = self._grafeo_node_to_labelled(edge.target_id)
        if src_node is None or tgt_node is None:
            return None
        edge_props = {k: v for k, v in edge.properties().items() if k not in _INTERNAL_EDGE_KEYS}
        relation = Relation(
            label=edge.edge_type,
            source_id=src_node.id,
            target_id=tgt_node.id,
            properties=edge_props,
        )
        return (src_node, relation, tgt_node)

    def _grafeo_node_to_labelled(self, node_id: int) -> LabelledNode | None:
        node = self._db.get_node(node_id)
        if node is None:
            return None

        props = node.properties()
        li_label = props.pop(_LABEL_KEY, "entity")
        li_id = props.pop(_ID_KEY, str(node_id))
        embedding = props.pop(_EMBEDDING_KEY, None)
        props.pop(_CREATED_AT_KEY, None)
        props.pop(_UPDATED_AT_KEY, None)

        if li_label == "text_chunk":
            text = props.pop(_TEXT_KEY, "")
            return ChunkNode(id_=li_id, text=text, label=li_label, properties=props, embedding=embedding)

        name = props.pop(_NAME_KEY, li_id)
        return EntityNode(name=name, label=li_label, properties=props, embedding=embedding)

    def _ensure_vector_index(self, labels: set[str]) -> None:
        # Creating an index that already exists rebuilds it, so this needs no "already exists"
        # handling. A failure (stored embeddings of other dimensions) propagates: before, the label
        # was recorded as indexed anyway and vector_query() silently returned nothing for it.
        for label in sorted(labels - self._vector_indexed_labels):
            self._db.create_vector_index(
                label=label,
                property=_EMBEDDING_KEY,
                dimensions=self._embedding_dimensions,
                metric=self._embedding_metric,
            )
            self._vector_indexed_labels.add(label)

    def _graph_changed(self) -> None:
        """Drop what is derived from the whole graph, after a write through the store."""
        self._schema_cache = None
        self._pagerank_cache = None

    def _pagerank(self) -> dict[int, float]:
        """PageRank per Grafeo node ID, scaled so that the highest score is 1.0.

        Computed by Grafeo and cached until the graph changes: writes through the store drop the
        cache, and writes made directly through ``client`` advance the database epoch it is keyed by.
        """
        epoch = self._db.current_epoch()
        cached = self._pagerank_cache
        if cached is not None and cached[0] == epoch:
            return cached[1]
        raw = self._db.algorithms.pagerank()
        top = max(raw.values(), default=0.0) or 1.0
        scores = {gid: score / top for gid, score in raw.items()}
        self._pagerank_cache = (epoch, scores)
        return scores

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

        query = "MATCH (n) RETURN id(n) AS node_id"
        params: dict[str, Any] = {}
        if properties:
            where_clause = " AND ".join(_property_conditions("n", properties, params))
            query = f"MATCH (n) WHERE {where_clause} RETURN id(n) AS node_id"
        for row in self._db.execute(query, params):
            labelled = self._grafeo_node_to_labelled(row["node_id"])
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
        """Get triplets (source, relation, target) matching filters.

        Filters combine with AND. ``entity_names`` and ``ids`` match either endpoint,
        ``relation_names`` matches the relation label as given or as normalized on upsert
        (``"works at"`` matches ``WORKS_AT``), and each ``properties`` entry must hold on the
        source or the target node. ``limit`` caps the number of triplets returned.
        """
        params: dict[str, Any] = {}
        where_parts: list[str] = []
        if entity_names:
            params["entity_names"] = list(entity_names)
            where_parts.append(f"(src.{_NAME_KEY} IN $entity_names OR tgt.{_NAME_KEY} IN $entity_names)")
        if ids:
            params["ids"] = list(ids)
            where_parts.append(f"(src.{_ID_KEY} IN $ids OR tgt.{_ID_KEY} IN $ids)")
        if relation_names:
            params["relation_types"] = _edge_types(relation_names)
            where_parts.append("type(rel) IN $relation_types")
        if properties:
            for key, value in properties.items():
                param = f"p{len(params)}"
                params[param] = value
                src_prop, tgt_prop = _property_ref("src", key), _property_ref("tgt", key)
                where_parts.append(f"({src_prop} = ${param} OR {tgt_prop} = ${param})")

        where_clause = f" WHERE {' AND '.join(where_parts)}" if where_parts else ""
        query = f"MATCH (src)-[rel]->(tgt){where_clause} RETURN id(rel) AS rel_id LIMIT {int(limit)}"

        triplets: list[Triplet] = []
        for row in self._db.execute(query, params):
            triplet = self._build_triplet(row["rel_id"])
            if triplet is not None:
                triplets.append(triplet)
        return triplets

    def get_rel_map(
        self,
        graph_nodes: list[LabelledNode],
        depth: int = 2,
        limit: int = 30,
        ignore_rels: list[str] | None = None,
    ) -> list[Triplet]:
        """Get depth-aware relationship map around the given nodes via batched BFS.

        Follows relations in both directions, like LlamaIndex's reference stores, up to ``depth``
        hops from the given nodes, and returns each relation once, in its stored direction.
        Relations whose label is in ``ignore_rels`` are neither returned nor followed. Returns at
        most ``limit`` triplets, nearest hops first.
        """
        if not graph_nodes:
            return []

        triplets: list[Triplet] = []
        seen_edges: set[int] = set()
        ignored = set(_edge_types(ignore_rels)) if ignore_rels else set()

        frontier: set[int] = set()
        for node in graph_nodes:
            gid = self._find_node_id(node.id)
            if gid is not None:
                frontier.add(gid)
        visited = set(frontier)

        for _ in range(depth):
            if not frontier or len(triplets) >= limit:
                break

            # One query per hop for the whole frontier. The undirected pattern matches incoming
            # and outgoing edges, and seeks by node ID from either end.
            rows = self._db.execute(
                "MATCH (n)-[rel]-(other) WHERE id(n) IN $frontier "
                "RETURN id(rel) AS rel_id, id(other) AS other_id, type(rel) AS rel_type",
                {"frontier": sorted(frontier)},
            )

            next_frontier: set[int] = set()
            for row in rows:
                if row["rel_type"] in ignored:
                    continue

                other_id = row["other_id"]
                if other_id not in visited:
                    visited.add(other_id)
                    next_frontier.add(other_id)

                rel_id = row["rel_id"]
                if rel_id in seen_edges:  # met from both ends, or a self-loop
                    continue
                seen_edges.add(rel_id)

                triplet = self._build_triplet(rel_id)
                if triplet is not None:
                    triplets.append(triplet)

            frontier = next_frontier

        return triplets[:limit]

    def upsert_nodes(self, nodes: Sequence[LabelledNode]) -> None:
        """Insert or update nodes in the graph.

        Raises:
            ValueError: if a node's embedding does not have ``embedding_dimensions`` values.
                The whole batch is checked before anything is written.
        """
        for node in nodes:
            if node.embedding and len(node.embedding) != self._embedding_dimensions:
                raise ValueError(
                    f"Node {node.id!r} has an embedding of {len(node.embedding)} dimensions, "
                    f"but the store is configured for {self._embedding_dimensions}"
                )

        embedded_labels: set[str] = set()
        now = datetime.now(UTC).isoformat()

        with self._lock:
            for node in nodes:
                existing_gid = self._find_node_id(node.id)

                # Dedup: if no exact match, try semantic similarity
                if (
                    existing_gid is None
                    and self.dedup_threshold is not None
                    and isinstance(node, EntityNode)
                    and node.embedding
                ):
                    similar_gid = self._find_similar_node(node.label, node.embedding, self.dedup_threshold)
                    if similar_gid is not None:
                        existing_gid = similar_gid
                        self._id_cache[node.id] = similar_gid

                props: dict[str, Any] = {_ID_KEY: node.id, _LABEL_KEY: node.label, **node.properties}

                if isinstance(node, EntityNode):
                    props[_NAME_KEY] = node.name
                elif isinstance(node, ChunkNode):
                    props[_TEXT_KEY] = node.text

                grafeo_label = _node_label(node.label)

                if node.embedding:
                    props[_EMBEDDING_KEY] = node.embedding
                    embedded_labels.add(grafeo_label)

                props[_UPDATED_AT_KEY] = now

                if existing_gid is not None:
                    for key, value in props.items():
                        self._db.set_node_property(existing_gid, key, value)
                    gid = existing_gid
                else:
                    props[_CREATED_AT_KEY] = now
                    created = self._db.create_node(labels=[grafeo_label], properties=props)
                    gid = created.id

                self._id_cache[node.id] = gid
                if isinstance(node, EntityNode):
                    self._name_cache[node.name] = gid

            if embedded_labels:
                self._ensure_vector_index(embedded_labels)

            self._graph_changed()

    def upsert_relations(self, relations: list[Relation]) -> None:
        """Insert or update relations (edges) in the graph.

        If an edge of the same type already exists between the same source and
        target, its properties are updated in place and ``updated_at`` is
        refreshed while ``created_at`` is preserved.
        """
        now = datetime.now(UTC).isoformat()

        with self._lock:
            for rel in relations:
                source_gid = self._resolve_endpoint(rel.source_id)
                target_gid = self._resolve_endpoint(rel.target_id)

                if source_gid is None or target_gid is None:
                    missing = []
                    if source_gid is None:
                        missing.append(f"source_id={rel.source_id!r}")
                    if target_gid is None:
                        missing.append(f"target_id={rel.target_id!r}")
                    warnings.warn(
                        f"Skipping relation '{rel.label}': missing {', '.join(missing)}",
                        UserWarning,
                        stacklevel=2,
                    )
                    continue

                edge_type = _edge_type(rel.label)
                existing_eid = self._find_existing_edge(source_gid, target_gid, edge_type)

                if existing_eid is not None:
                    for key, value in rel.properties.items():
                        self._db.set_edge_property(existing_eid, key, value)
                    self._db.set_edge_property(existing_eid, _UPDATED_AT_KEY, now)
                else:
                    props = {**rel.properties, _CREATED_AT_KEY: now, _UPDATED_AT_KEY: now}
                    self._db.create_edge(
                        source_id=source_gid,
                        target_id=target_gid,
                        edge_type=edge_type,
                        properties=props,
                    )

            self._graph_changed()

    def _evict_nodes_from_caches(self, gids: set[int]) -> None:
        """Remove all cache entries pointing to the given Grafeo node IDs."""
        for cache in (self._id_cache, self._name_cache):
            for key, cached_gid in list(cache.items()):
                if cached_gid in gids:
                    cache.pop(key, None)

    def delete(
        self,
        entity_names: list[str] | None = None,
        relation_names: list[str] | None = None,
        properties: dict | None = None,
        ids: list[str] | None = None,
    ) -> None:
        """Delete matching nodes and/or relations.

        Nodes matched by ``ids``, ``entity_names`` or ``properties`` are deleted together with
        every relation attached to them. ``relation_names`` deletes the relations of those
        labels (as given or as normalized on upsert) and keeps their endpoints.
        """
        with self._lock:
            gids: set[int] = set()
            for li_id in ids or []:
                gids |= self._matching_nodes(self._id_cache, _ID_KEY, li_id)
            for name in entity_names or []:
                gids |= self._matching_nodes(self._name_cache, _NAME_KEY, name)
            if properties:
                params: dict[str, Any] = {}
                where_clause = " AND ".join(_property_conditions("n", properties, params))
                rows = self._db.execute(f"MATCH (n) WHERE {where_clause} RETURN id(n) AS node_id", params)
                gids.update(row["node_id"] for row in rows)

            if gids:
                # Grafeo's delete_node() refuses a node that still has edges (since 0.5.44),
                # so remove the nodes and their edges in one statement.
                self._db.execute("MATCH (n) WHERE id(n) IN $ids DETACH DELETE n", {"ids": sorted(gids)})
                self._evict_nodes_from_caches(gids)

            if relation_names:
                self._db.execute(
                    "MATCH ()-[r]->() WHERE type(r) IN $edge_types DELETE r",
                    {"edge_types": _edge_types(relation_names)},
                )

            self._graph_changed()

    def structured_query(self, query: str, param_map: dict[str, Any] | None = None, language: str | None = None) -> Any:
        """Execute a raw query and return the result rows as a list of dicts.

        The query runs in ``language`` (``"gql"`` or ``"cypher"``), by default the store's
        ``query_language``; a query that starts with ``g.`` runs as Gremlin. ``param_map`` is
        passed to Grafeo as query parameters, referenced as ``$name`` in the query.
        """
        language = self._query_language if language is None else _check_query_language(language)
        params = dict(param_map) if param_map else None
        if query.strip().lower().startswith("g."):
            rows = list(self._db.execute_gremlin(query, params))
        elif language == "cypher":
            rows = list(self._db.execute_cypher(query, params))
        else:
            rows = list(self._db.execute(query, params))
        self._graph_changed()  # the query may have written
        return rows

    async def astructured_query(
        self, query: str, param_map: dict[str, Any] | None = None, language: str | None = None
    ) -> Any:
        """Async ``structured_query``, with the same ``language`` override."""
        return self.structured_query(query, param_map, language=language)

    def vector_query(self, query: VectorStoreQuery, **kwargs: Any) -> tuple[list[LabelledNode], list[float]]:
        """Perform vector similarity search using Grafeo's HNSW index."""
        if query.query_embedding is None:
            return [], []

        if len(query.query_embedding) != self._embedding_dimensions:
            raise ValueError(
                f"Query embedding has {len(query.query_embedding)} dimensions, "
                f"but the store is configured for {self._embedding_dimensions}"
            )

        k = query.similarity_top_k

        all_results: list[tuple[int, float]] = []
        # Iterate over a copy: an upsert in another thread may add a label meanwhile.
        for label in tuple(self._vector_indexed_labels):
            try:
                results = self._db.vector_search(
                    label=label,
                    property=_EMBEDDING_KEY,
                    query=query.query_embedding,
                    k=k,
                )
                all_results.extend(results)
            except RuntimeError:
                continue  # the index was dropped outside the store

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

    # ─── Convenience API ───────────────────────────────────────────────────────────────────────────

    @property
    def node_count(self) -> int:
        """Total number of nodes in the graph."""
        return self._db.node_count

    @property
    def edge_count(self) -> int:
        """Total number of edges in the graph."""
        return self._db.edge_count

    def summary(self) -> str:
        """Return a human-readable graph overview."""
        schema = self.get_schema(refresh=True)

        lines = [
            f"Nodes: {self._db.node_count}",
            f"Edges: {self._db.edge_count}",
        ]

        labels = schema.get("labels", [])
        if labels:
            lines.append("Labels:")
            for entry in labels:
                lines.append(f"  {entry['name']}: {entry['count']}")

        edge_types = schema.get("edge_types", [])
        if edge_types:
            lines.append("Edge types:")
            for entry in edge_types:
                lines.append(f"  {entry['name']}: {entry['count']}")

        if self.dedup_threshold is not None:
            lines.append(f"Dedup: enabled (threshold={self.dedup_threshold})")
        else:
            lines.append("Dedup: disabled")

        lines.append(f"Persistent: {self._db.is_persistent}")

        return "\n".join(lines)

    def neighbors(self, name_or_id: str, *, depth: int = 1, limit: int = 30) -> list[Triplet]:
        """Get triplets around a node by name or ID without writing GQL."""
        nodes = self.get(ids=[name_or_id])
        if not nodes:
            nodes = self.get(properties={_NAME_KEY: name_or_id})
        if not nodes:
            return []
        return self.get_rel_map(nodes, depth=depth, limit=limit)

    def __repr__(self) -> str:
        return (
            f"GrafeoPropertyGraphStore("
            f"nodes={self._db.node_count}, "
            f"edges={self._db.edge_count}, "
            f"persistent={self._db.is_persistent})"
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
