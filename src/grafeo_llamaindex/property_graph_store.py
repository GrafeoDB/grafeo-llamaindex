"""GrafeoDB-backed PropertyGraphStore for LlamaIndex."""

from __future__ import annotations

import os
import re
import threading
import uuid
import warnings
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import grafeo
from llama_index.core.graph_stores.types import (
    KG_SOURCE_REL,
    TRIPLET_SOURCE_KEY,
    ChunkNode,
    EntityNode,
    LabelledNode,
    PropertyGraphStore,
    Relation,
    Triplet,
)
from llama_index.core.vector_stores.types import FilterCondition, FilterOperator, MetadataFilters, VectorStoreQuery

if TYPE_CHECKING:
    from llama_index.core.base.embeddings.base import BaseEmbedding

_ID_KEY = "li_id"
_NAME_KEY = "name"
_TEXT_KEY = "text"
_LABEL_KEY = "li_label"
_EMBEDDING_KEY = "embedding"
_CREATED_AT_KEY = "created_at"
_UPDATED_AT_KEY = "updated_at"
_ALIASES_KEY = "li_aliases"  # LlamaIndex IDs merged into this node by dedup or a name match
_NAME_FOLD_KEY = "li_name_key"  # an entity's name in folded case, for case-insensitive matching
_SOURCES_KEY = "li_sources"  # IDs of the chunks an entity or relation was extracted from
_KIND_KEY = "li_kind"  # "chunk" for a ChunkNode, "entity" for any other node
_CHUNK_KIND = "chunk"
_ENTITY_KIND = "entity"
_CHUNK_LABEL = "text_chunk"  # the default label of LlamaIndex's ChunkNode

# Keys the store writes for its own bookkeeping, and LlamaIndex's private keys: left out of the
# schema text an LLM writes queries from (get_schema_str), as is every key starting with "_".
_SCHEMA_HIDDEN_KEYS = frozenset(
    {_ID_KEY, _LABEL_KEY, _KIND_KEY, _ALIASES_KEY, _SOURCES_KEY, _NAME_FOLD_KEY, _EMBEDDING_KEY, TRIPLET_SOURCE_KEY}
)

# LlamaIndex filter operators and the Grafeo vector_search operators they become (EQ is a plain value).
_FILTER_OPERATORS = {
    FilterOperator.NE: "$ne",
    FilterOperator.GT: "$gt",
    FilterOperator.GTE: "$gte",
    FilterOperator.LT: "$lt",
    FilterOperator.LTE: "$lte",
    FilterOperator.IN: "$in",
    FilterOperator.NIN: "$nin",
}

_INTERNAL_EDGE_KEYS = frozenset({_CREATED_AT_KEY, _UPDATED_AT_KEY, _SOURCES_KEY})

QUERY_LANGUAGES = ("gql", "cypher")
"""Languages ``structured_query`` accepts; a query starting with ``g.`` always runs as Gremlin."""

DEFAULT_QUERY_LANGUAGE = "cypher"
"""The language ``structured_query`` uses unless the store or the call names one.

The single place this default is defined. Cypher since grafeo-llamaindex 0.3.0 (GQL before), the
language LlamaIndex's ``TextToCypherRetriever`` and ``CypherTemplateRetriever`` generate.
"""

DEFAULT_EMBEDDING_DIMENSIONS = 1536
"""Embedding size used when neither ``embedding_dimensions`` nor ``embed_model`` is given (OpenAI's)."""

_DIMENSION_PROBE_TEXT = "grafeo-llamaindex dimension probe"

# Every query below passes values as Grafeo query parameters ($name), never as text in the query.
# Only property keys (validated identifiers, quoted) and the integer LIMIT are written into the query.


def _validate_key(key: str) -> str:
    """Validate a property key: any non-empty text without a backtick, which would end the quoted key."""
    if not isinstance(key, str) or not key or "`" in key:
        raise ValueError(f"Invalid property key: {key!r}")
    return key


def _vector_filters(filters: MetadataFilters | None, node_ids: list[str] | None) -> dict[str, Any] | None:
    """Translate the filters and node IDs of a vector query into Grafeo ``vector_search`` filters.

    Grafeo combines its filters with AND. Another condition, nested filters, and operators without
    a Grafeo counterpart raise ``NotImplementedError`` rather than being ignored.
    """
    equal: dict[str, Any] = {}
    operators: dict[str, dict[str, Any]] = {}
    if filters is not None and filters.filters:
        condition = filters.condition
        if (
            condition is not None
            and condition != FilterCondition.AND
            and (len(filters.filters) > 1 or condition.value == "not")
        ):
            raise NotImplementedError(
                f"vector_query filters combined with {condition.value!r}: only 'and' is supported"
            )
        for item in filters.filters:
            if isinstance(item, MetadataFilters):
                raise NotImplementedError("vector_query does not support nested MetadataFilters")
            key = _validate_key(item.key)
            if item.operator == FilterOperator.EQ:
                if key in equal:
                    raise NotImplementedError(f"vector_query got two equality filters on {key!r}")
                equal[key] = item.value
            elif item.operator in _FILTER_OPERATORS:
                operator = _FILTER_OPERATORS[item.operator]
                if operator in operators.setdefault(key, {}):
                    raise NotImplementedError(f"vector_query got two {item.operator.value!r} filters on {key!r}")
                operators[key][operator] = item.value
            else:
                supported = ", ".join(repr(op.value) for op in (FilterOperator.EQ, *_FILTER_OPERATORS))
                raise NotImplementedError(
                    f"vector_query does not support the filter operator {item.operator.value!r}: use {supported}"
                )
    if node_ids:
        operators.setdefault(_ID_KEY, {})["$in"] = list(node_ids)
    if both := equal.keys() & operators.keys():
        raise NotImplementedError(f"vector_query cannot combine equality with other filters on {sorted(both)}")
    result: dict[str, Any] = {**equal, **operators}
    return result or None


def _property_ref(var: str, key: str) -> str:
    """Reference property ``key`` of ``var``, quoted so that keywords such as ``group`` work as keys."""
    return f"{var}.`{_validate_key(key)}`"


# One markdown code fence around a whole query, as LLMs often write one: ```cypher ... ```,
# with an optional language tag. Text outside the fence is not removed.
_CODE_FENCE = re.compile(
    r"\A\s*```[ \t]*(?:(?:cypher|opencypher|gql|sql|gremlin|groovy)\b)?[ \t]*\n?(?P<query>.*?)\s*```\s*\Z",
    re.IGNORECASE | re.DOTALL,
)


def _strip_code_fence(query: str) -> str:
    """Return ``query`` without one markdown code fence around it, as an LLM may write it."""
    match = _CODE_FENCE.match(query)
    return match.group("query").strip() if match else query


def _check_query_language(language: str) -> str:
    """Return ``language`` in lower case, or raise ``ValueError`` if it is not a supported language."""
    normalized = language.lower() if isinstance(language, str) else language
    if normalized not in QUERY_LANGUAGES:
        allowed = ", ".join(repr(name) for name in QUERY_LANGUAGES)
        raise ValueError(f"Unsupported query language {language!r}: use one of {allowed}")
    return normalized


def _resolve_dimensions(dimensions: int | None, embed_model: BaseEmbedding | None) -> int:
    """Return the store's embedding size: read from ``embed_model`` (one probe embedding) or given.

    Raises ``ValueError`` if the model returns an empty embedding, or one of a size other than
    ``dimensions`` when both are given.
    """
    if embed_model is None:
        return DEFAULT_EMBEDDING_DIMENSIONS if dimensions is None else dimensions
    probed = len(embed_model.get_text_embedding(_DIMENSION_PROBE_TEXT))
    if probed == 0:
        raise ValueError("embed_model returned an empty embedding: cannot read its dimensions")
    if dimensions is not None and dimensions != probed:
        raise ValueError(
            f"embedding_dimensions={dimensions} does not match embed_model, whose embeddings have "
            f"{probed} dimensions: pass one of them, or the same size"
        )
    return probed


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


def _merged_sources(stored: dict[str, Any], source: str) -> list[str]:
    """The source chunks of a stored node or edge, with ``source`` added.

    Data written before ``li_sources`` existed has only ``triplet_source_id``, kept as its first source.
    """
    sources = stored.get(_SOURCES_KEY)
    if sources is None:
        legacy = stored.get(TRIPLET_SOURCE_KEY)
        sources = [legacy] if legacy is not None else []
    return list(sources) if source in sources else [*sources, source]


def _remaining_sources(sources: list[str] | None, source: str | None, deleted: set[str]) -> list[str]:
    """The source chunks left once ``deleted`` are gone, from ``li_sources`` or else ``triplet_source_id``."""
    if sources is None:
        sources = [source] if source is not None else []
    return [s for s in sources if s not in deleted]


def _schema_keys(entry: dict[str, Any]) -> list[str]:
    """The property names of a schema label or edge type entry that an LLM may query."""
    names = {p["name"] if isinstance(p, dict) else str(p) for p in entry.get("properties", [])}
    return sorted(k for k in names if k not in _SCHEMA_HIDDEN_KEYS and not k.startswith("_"))


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

    ``query_language`` (``"cypher"`` or ``"gql"``, default ``DEFAULT_QUERY_LANGUAGE``, Cypher) is
    the language ``structured_query`` runs queries in. Cypher is what LlamaIndex's text-to-Cypher
    retrievers generate.

    ``db_path`` is a single database file (use the ``.grafeo`` extension): grafeo 0.6 keeps its
    write-ahead log in ``<db_path>.wal/`` while the store is open. A database written by grafeo 0.5
    is migrated on the first open, which keeps the old database as ``<db_path>.pre-0.6``.

    With ``case_insensitive_names`` (the default), an entity whose name matches an existing one
    ignoring case, such as the ``"Gavin belson"`` LlamaIndex's default extractor writes for
    ``"Gavin Belson"``, merges into it, as relation endpoints and name lookups also match.

    ``embedding_dimensions`` is the size of the node embeddings. With ``embed_model``, it is read
    from the model, which embeds one short text when the store is created; without either, it is
    ``DEFAULT_EMBEDDING_DIMENSIONS`` (1536).
    """

    supports_structured_queries: bool = True
    supports_vector_queries: bool = True

    def __init__(
        self,
        *,
        db_path: str | os.PathLike[str] | None = None,
        embedding_dimensions: int | None = None,
        embedding_metric: str = "cosine",
        dedup_threshold: float | None = None,
        query_language: str = DEFAULT_QUERY_LANGUAGE,
        embed_model: BaseEmbedding | None = None,
        case_insensitive_names: bool = True,
    ) -> None:
        self._query_language = _check_query_language(query_language)
        db_path = os.fspath(db_path) if db_path is not None else None
        self._db_path = db_path
        self._embedding_dimensions = _resolve_dimensions(embedding_dimensions, embed_model)
        self._embedding_metric = embedding_metric
        self.dedup_threshold = dedup_threshold
        self._case_insensitive_names = case_insensitive_names

        self._db = grafeo.GrafeoDB(db_path) if db_path else grafeo.GrafeoDB()

        if not self._db.has_property_index(_ID_KEY):
            self._db.create_property_index(_ID_KEY)
        if not self._db.has_property_index(_NAME_KEY):
            self._db.create_property_index(_NAME_KEY)
        if not self._db.has_property_index(_NAME_FOLD_KEY):
            self._db.create_property_index(_NAME_FOLD_KEY)

        self._vector_indexed_labels: set[str] = set()
        # Grafeo labels other than text_chunk that hold chunks: vector searches on them keep entities only.
        self._chunk_labels: set[str] = set()
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
            self._discover_chunk_labels()
            self._load_aliases()

    @property
    def client(self) -> grafeo.GrafeoDB:
        """Return the underlying GrafeoDB instance."""
        return self._db

    @property
    def query_language(self) -> str:
        """The language ``structured_query`` uses when a call does not name one."""
        return self._query_language

    @property
    def embedding_dimensions(self) -> int:
        """The size every node embedding and vector query must have."""
        return self._embedding_dimensions

    # ─── Internal helpers ─────────────────────────────────────────────────────────────────────────

    def _rediscover_vector_indexes(self) -> None:
        """Index the embeddings of every label of a reopened persistent database.

        Every label is indexed again with this store's dimensions and metric, whatever index the
        file kept (a database migrated from grafeo 0.5 may have none), so ``vector_query`` uses the
        metric the store was opened with. A label whose stored embeddings have other dimensions
        cannot be indexed: it is reported with a ``UserWarning`` and ``vector_query`` skips it.
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

    def _discover_chunk_labels(self) -> None:
        """Find the labels other than ``text_chunk`` that hold chunks, in a reopened database."""
        for entry in self._db.schema().get("labels", []):
            label = entry["name"]
            if label == _CHUNK_LABEL or "`" in label:
                continue
            rows = self._db.execute(
                f"MATCH (n:`{label}`) WHERE n.{_KIND_KEY} = $kind RETURN id(n) AS gid LIMIT 1", {"kind": _CHUNK_KIND}
            )
            if list(rows):
                self._chunk_labels.add(label)

    def _entity_filters(self, label: str, filters: dict[str, Any] | None) -> dict[str, Any] | None:
        """The vector_search filters for ``label``: ``filters``, plus entities only on a label holding chunks."""
        if label not in self._chunk_labels:
            return filters
        return {**(filters or {}), _KIND_KEY: _ENTITY_KIND}

    def _load_aliases(self) -> None:
        """Cache the LlamaIndex IDs that dedup merged into other nodes of a reopened database.

        A merged ID is kept only in the ``li_aliases`` list of the node it merged into, which the
        property index cannot find, so it is read once here and resolved through the caches.
        """
        if _ALIASES_KEY not in self._db.schema().get("property_keys", []):
            return
        rows = self._db.execute(
            f"MATCH (n) WHERE n.{_ALIASES_KEY} IS NOT NULL RETURN id(n) AS gid, n.{_ALIASES_KEY} AS aliases"
        )
        for row in rows:
            for alias in row["aliases"] or []:
                self._id_cache.setdefault(alias, row["gid"])
                self._name_cache.setdefault(alias, row["gid"])

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

    def _find_node_by_folded_name(self, name: str) -> int | None:
        """An entity whose name matches ``name`` ignoring case, with ``case_insensitive_names`` on."""
        if not self._case_insensitive_names:
            return None
        for gid in self._db.find_nodes_by_property(_NAME_FOLD_KEY, name.casefold()):
            self._name_cache[name] = gid
            return gid
        return None

    def _resolve_endpoint(self, ref: str) -> int | None:
        """Resolve a relation endpoint: by entity name, by LlamaIndex ID, then by name ignoring case."""
        for find in (self._find_node_by_name, self._find_node_id, self._find_node_by_folded_name):
            gid = find(ref)
            if gid is not None:
                return gid
        return None

    def _matching_names(self, name: str) -> set[int]:
        """Every entity named ``name``, also ignoring case with ``case_insensitive_names`` on."""
        gids = self._matching_nodes(self._name_cache, _NAME_KEY, name)
        if self._case_insensitive_names:
            gids.update(self._db.find_nodes_by_property(_NAME_FOLD_KEY, name.casefold()))
        return gids

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
                filters=self._entity_filters(grafeo_label, None),
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
        props.pop(_ALIASES_KEY, None)
        props.pop(_NAME_FOLD_KEY, None)
        props.pop(_SOURCES_KEY, None)
        kind = props.pop(_KIND_KEY, None)

        # Nodes written before li_kind existed are chunks by their default label.
        if kind == _CHUNK_KIND or (kind is None and li_label == _CHUNK_LABEL):
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

        Filters combine with AND. ``entity_names`` and ``ids`` match either endpoint, also through
        the aliases dedup keeps for merged nodes (as ``get(ids=...)`` does); ``relation_names``
        matches the relation label as given or as normalized on upsert (``"works at"`` matches
        ``WORKS_AT``), and each ``properties`` entry must hold on the source or the target node.
        ``limit`` caps the number of triplets returned.
        """
        params: dict[str, Any] = {}
        where_parts: list[str] = []
        for param, match, values in (
            ("name_nodes", self._matching_names, entity_names),
            ("id_nodes", lambda li_id: self._matching_nodes(self._id_cache, _ID_KEY, li_id), ids),
        ):
            if not values:
                continue
            gids: set[int] = set()
            for value in values:
                gids |= match(value)
            if not gids:
                return []
            params[param] = sorted(gids)
            where_parts.append(f"(id(src) IN ${param} OR id(tgt) IN ${param})")
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
                if existing_gid is None and isinstance(node, EntityNode):
                    existing_gid = self._find_node_by_folded_name(node.name)

                # Dedup: if no exact match, try semantic similarity
                if (
                    existing_gid is None
                    and self.dedup_threshold is not None
                    and isinstance(node, EntityNode)
                    and node.embedding
                ):
                    existing_gid = self._find_similar_node(node.label, node.embedding, self.dedup_threshold)

                props: dict[str, Any] = {_ID_KEY: node.id, _LABEL_KEY: node.label, **node.properties}
                is_chunk = isinstance(node, ChunkNode)
                props[_KIND_KEY] = _CHUNK_KIND if is_chunk else _ENTITY_KIND

                if isinstance(node, EntityNode):
                    props[_NAME_KEY] = node.name
                    props[_NAME_FOLD_KEY] = node.name.casefold()
                elif isinstance(node, ChunkNode):
                    props[_TEXT_KEY] = node.text

                grafeo_label = _node_label(node.label)
                if is_chunk and grafeo_label != _CHUNK_LABEL:
                    self._chunk_labels.add(grafeo_label)
                source = None if is_chunk else node.properties.get(TRIPLET_SOURCE_KEY)

                if node.embedding:
                    props[_EMBEDDING_KEY] = node.embedding
                    embedded_labels.add(grafeo_label)

                props[_UPDATED_AT_KEY] = now

                if existing_gid is not None:
                    self._prepare_update(existing_gid, node, props, grafeo_label, source, embedded_labels)
                    for key, value in props.items():
                        self._db.set_node_property(existing_gid, key, value)
                    gid = existing_gid
                else:
                    props[_CREATED_AT_KEY] = now
                    if source is not None:
                        props[_SOURCES_KEY] = [source]
                    created = self._db.create_node(labels=[grafeo_label], properties=props)
                    gid = created.id

                self._id_cache[node.id] = gid
                if isinstance(node, EntityNode):
                    self._name_cache[node.name] = gid

            if embedded_labels:
                self._ensure_vector_index(embedded_labels)

            self._graph_changed()

    def _prepare_update(
        self,
        gid: int,
        node: LabelledNode,
        props: dict[str, Any],
        grafeo_label: str,
        source: str | None,
        embedded_labels: set[str],
    ) -> None:
        """Prepare the update of existing node ``gid`` with ``node``: keep its identity, follow its label.

        When ``node`` reached ``gid`` through dedup, a name match ignoring case, or an alias (the
        stored ID differs), the node keeps its ID, name and label, and ``node.id`` is added to its
        aliases, so the merge survives a reopen. Otherwise, when the label changed, the Grafeo label
        follows, and the new label gets a vector index if the node has an embedding. The chunk
        ``source`` is added to the node's sources.
        """
        stored = self._db.get_node(gid).properties()
        if source is not None:
            props[_SOURCES_KEY] = _merged_sources(stored, source)
        stored_id = stored.get(_ID_KEY)
        if stored_id is not None and stored_id != node.id:
            for key in (_ID_KEY, _NAME_KEY, _NAME_FOLD_KEY, _LABEL_KEY):
                props.pop(key, None)
            aliases = list(stored.get(_ALIASES_KEY) or [])
            if node.id not in aliases:
                props[_ALIASES_KEY] = [*aliases, node.id]
            return  # a merged node keeps its label

        old_label = _node_label(str(stored.get(_LABEL_KEY, grafeo_label)))
        if old_label != grafeo_label:
            if old_label in (self._db.get_node_labels(gid) or []):
                self._db.remove_node_label(gid, old_label)
            self._db.add_node_label(gid, grafeo_label)
            if _EMBEDDING_KEY in stored:
                embedded_labels.add(grafeo_label)

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
                    # LlamaIndex's ImplicitPathExtractor links each chunk to its document by a SOURCE
                    # relation, and documents are not graph nodes (a chunk keeps its ref_doc_id): an
                    # expected miss on every build, skipped without a warning.
                    if rel.label == KG_SOURCE_REL and source_gid is not None:
                        continue
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
                source = rel.properties.get(TRIPLET_SOURCE_KEY)

                if existing_eid is not None:
                    if source is not None:
                        stored = self._db.get_edge(existing_eid).properties()
                        self._db.set_edge_property(existing_eid, _SOURCES_KEY, _merged_sources(stored, source))
                    for key, value in rel.properties.items():
                        self._db.set_edge_property(existing_eid, key, value)
                    self._db.set_edge_property(existing_eid, _UPDATED_AT_KEY, now)
                else:
                    props = {**rel.properties, _CREATED_AT_KEY: now, _UPDATED_AT_KEY: now}
                    if source is not None:
                        props[_SOURCES_KEY] = [source]
                    self._db.create_edge(
                        source_id=source_gid,
                        target_id=target_gid,
                        edge_type=edge_type,
                        properties=props,
                    )

            self._graph_changed()

    def _detach_delete(self, gids: set[int]) -> None:
        """Delete nodes with every relation attached to them, and forget their cached IDs."""
        # Grafeo's delete_node() refuses a node that still has edges (since 0.5.44), so remove the
        # nodes and their edges in one statement.
        self._db.execute("MATCH (n) WHERE id(n) IN $ids DETACH DELETE n", {"ids": sorted(gids)})
        self._evict_nodes_from_caches(gids)

    def _chunk_ids(self, gids: set[int]) -> list[str]:
        """The LlamaIndex IDs of the chunks among nodes ``gids``."""
        if not gids:
            return []
        rows = self._db.execute(
            f"MATCH (n) WHERE id(n) IN $ids "
            f"RETURN n.{_ID_KEY} AS li_id, n.{_KIND_KEY} AS kind, n.{_LABEL_KEY} AS label",
            {"ids": sorted(gids)},
        )
        return [
            row["li_id"]
            for row in rows
            if row["li_id"] is not None
            and (row["kind"] == _CHUNK_KIND or (row["kind"] is None and row["label"] == _CHUNK_LABEL))
        ]

    def _delete_extracted(self, chunk_ids: list[str]) -> None:
        """Delete what was extracted only from the deleted chunks ``chunk_ids`` (see ``delete``).

        A relation or entity lists its source chunks in ``li_sources``; data written before that
        existed has only ``triplet_source_id``, which counts as its one source. Relations whose
        sources are all deleted go; the others keep their live sources, and a ``triplet_source_id``
        naming a deleted chunk moves to a live one (LlamaIndex reads a triplet's text through it).
        Entities whose sources are all deleted go once no relation holds them. Each call scans the
        relations and nodes that have sources once.
        """
        deleted = set(chunk_ids)
        params = {"chunks": sorted(deleted)}

        def sourced(var: str) -> str:
            return (
                f"(any(s IN {var}.{_SOURCES_KEY} WHERE s IN $chunks) OR "
                f"({var}.{_SOURCES_KEY} IS NULL AND {var}.{TRIPLET_SOURCE_KEY} IN $chunks))"
            )

        candidates: set[int] = set()  # nodes that may have lost their last source or relation
        dead_edges: list[int] = []
        rows = self._db.execute(
            f"MATCH (a)-[r]->(b) WHERE {sourced('r')} RETURN id(r) AS rid, id(a) AS a, id(b) AS b, "
            f"r.{_SOURCES_KEY} AS sources, r.{TRIPLET_SOURCE_KEY} AS source",
            params,
        )
        for row in rows:
            remaining = _remaining_sources(row["sources"], row["source"], deleted)
            if not remaining:
                dead_edges.append(row["rid"])
                candidates |= {row["a"], row["b"]}
                continue
            self._db.set_edge_property(row["rid"], _SOURCES_KEY, remaining)
            if row["source"] in deleted:
                self._db.set_edge_property(row["rid"], TRIPLET_SOURCE_KEY, remaining[-1])
        if dead_edges:
            self._db.execute("MATCH ()-[r]->() WHERE id(r) IN $ids DELETE r", {"ids": dead_edges})

        rows = self._db.execute(
            f"MATCH (n) WHERE {sourced('n')} RETURN id(n) AS gid, "
            f"n.{_SOURCES_KEY} AS sources, n.{TRIPLET_SOURCE_KEY} AS source",
            params,
        )
        for row in rows:
            remaining = _remaining_sources(row["sources"], row["source"], deleted)
            self._db.set_node_property(row["gid"], _SOURCES_KEY, remaining)  # [] marks "all sources deleted"
            if row["source"] in deleted:
                if remaining:
                    self._db.set_node_property(row["gid"], TRIPLET_SOURCE_KEY, remaining[-1])
                else:
                    self._db.remove_node_property(row["gid"], TRIPLET_SOURCE_KEY)
            candidates.add(row["gid"])
        if not candidates:
            return

        # An entity goes when all its sources are deleted (an empty list, never a missing one: an
        # entity written without a source is kept) and no relation holds it any more.
        rows = self._db.execute(
            f"MATCH (n) WHERE id(n) IN $ids RETURN id(n) AS gid, n.{_SOURCES_KEY} AS sources",
            {"ids": sorted(candidates)},
        )
        exhausted = {row["gid"] for row in rows if row["sources"] is not None and len(row["sources"]) == 0}
        if not exhausted:
            return
        rows = self._db.execute(
            "MATCH (n)-[r]-() WHERE id(n) IN $ids RETURN DISTINCT id(n) AS gid", {"ids": sorted(exhausted)}
        )
        orphans = exhausted - {row["gid"] for row in rows}
        if orphans:
            self._detach_delete(orphans)

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

        Deleting a chunk also deletes what was extracted only from it: the relations whose every
        source chunk is deleted, then the entities left with no source chunk and no relation.
        Entities and relations written without a ``triplet_source_id`` are never deleted this way.
        This is what makes ``PropertyGraphIndex.delete_ref_doc()`` remove a document's knowledge.
        """
        with self._lock:
            gids: set[int] = set()
            for li_id in ids or []:
                gids |= self._matching_nodes(self._id_cache, _ID_KEY, li_id)
            for name in entity_names or []:
                gids |= self._matching_names(name)
            if properties:
                params: dict[str, Any] = {}
                where_clause = " AND ".join(_property_conditions("n", properties, params))
                rows = self._db.execute(f"MATCH (n) WHERE {where_clause} RETURN id(n) AS node_id", params)
                gids.update(row["node_id"] for row in rows)

            chunk_ids = self._chunk_ids(gids)
            if gids:
                self._detach_delete(gids)
            if chunk_ids:
                self._delete_extracted(chunk_ids)

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
        passed to Grafeo as query parameters, referenced as ``$name`` in the query. One markdown
        code fence around the whole query (```` ```cypher ... ``` ````), which LLMs often write
        for ``TextToCypherRetriever``, is removed first.
        """
        language = self._query_language if language is None else _check_query_language(language)
        query = _strip_code_fence(query)
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
        """Find the entity nodes nearest to ``query.query_embedding`` with Grafeo's HNSW index.

        Like LlamaIndex's Neo4j store, only entity nodes are searched, not text chunks, so chunks
        do not take the seed slots of ``VectorContextRetriever``. ``query.filters`` (combined with
        AND; operators ``==``, ``!=``, ``>``, ``>=``, ``<``, ``<=``, ``in``, ``nin``) and
        ``query.node_ids`` narrow the search inside the index.

        Raises:
            ValueError: if the query embedding does not have ``embedding_dimensions`` values.
            NotImplementedError: for filters combined with OR or NOT, nested filters, or another operator.
        """
        if query.query_embedding is None:
            return [], []
        grafeo_filters = _vector_filters(query.filters, query.node_ids)

        if len(query.query_embedding) != self._embedding_dimensions:
            raise ValueError(
                f"Query embedding has {len(query.query_embedding)} dimensions, "
                f"but the store is configured for {self._embedding_dimensions}"
            )

        k = query.similarity_top_k

        all_results: list[tuple[int, float]] = []
        # Iterate over a copy: an upsert in another thread may add a label meanwhile.
        for label in tuple(self._vector_indexed_labels):
            if label == _CHUNK_LABEL:
                continue
            try:
                results = self._db.vector_search(
                    label=label,
                    property=_EMBEDDING_KEY,
                    query=query.query_embedding,
                    k=k,
                    filters=self._entity_filters(label, grafeo_filters),
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
        schema = self._schema_cache
        if schema is None or refresh:
            schema = self._db.schema()
            self._schema_cache = schema
        return schema

    def get_schema_str(self, refresh: bool = False) -> str:
        """Describe the graph for an LLM that writes queries over it, such as ``TextToCypherRetriever``.

        Lists the properties of each node label and relation type, and the patterns relations
        form between labels, in the layout of LlamaIndex's Neo4j store. The store's bookkeeping
        keys (``li_*``, ``embedding``) and LlamaIndex's private keys are left out.
        """
        schema = self.get_schema(refresh)
        lines = ["Node properties:"]
        for entry in sorted(schema.get("labels", []), key=lambda e: e["name"]):
            keys = _schema_keys(entry)
            lines.append(f"- {entry['name']}: {', '.join(keys)}" if keys else f"- {entry['name']}")

        edge_types = sorted(schema.get("edge_types", []), key=lambda e: e["name"])
        relationship_lines = [f"- {e['name']}: {', '.join(keys)}" for e in edge_types if (keys := _schema_keys(e))]
        if relationship_lines:
            lines += ["Relationship properties:", *relationship_lines]

        patterns = sorted(
            {
                f"(:{end['source']})-[:{e['name']}]->(:{end['target']})"
                for e in edge_types
                for end in e.get("endpoints", [])
            }
        )
        if patterns:
            lines += ["The relationships:", *patterns]
        return "\n".join(lines)

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

    def persist(self, persist_path: str | os.PathLike[str], fs: Any = None) -> None:
        """Save an in-memory store to the database file ``persist_path``, replacing a file there.

        A path ending in ``.json`` is saved with ``.grafeo`` instead: ``StorageContext.persist()``
        passes ``<persist_dir>/property_graph_store.json``, which ``StorageContext.from_defaults``
        would later read as JSON. Reopen the store with ``db_path`` set to the ``.grafeo`` file.

        Grafeo's ``save()`` refuses an existing path since grafeo 0.6, so the database is saved
        next to the target and then moved over it: persisting again (as ``StorageContext.persist()``
        does on every call) replaces the earlier copy, as it did with grafeo 0.5. A store opened with
        ``db_path`` already writes to its file, so this does nothing.

        Raises:
            IsADirectoryError: if the target is a directory, such as a grafeo 0.5 database.
        """
        if self._db.is_persistent:
            return
        target = Path(persist_path)
        if target.suffix.lower() == ".json":
            target = target.with_suffix(".grafeo")
        if target.is_dir():
            raise IsADirectoryError(f"Cannot persist to {persist_path!r}: it is a directory, not a database file")
        staging = target.with_name(f"{target.name}.{uuid.uuid4().hex}.tmp")
        try:
            self._db.save(str(staging))
            os.replace(staging, target)
        finally:
            staging.unlink(missing_ok=True)

    def close(self) -> None:
        """Close the database connection."""
        self._db.close()

    def __enter__(self) -> GrafeoPropertyGraphStore:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
