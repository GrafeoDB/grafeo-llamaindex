"""The store used from several threads, as LlamaIndex does with ``num_workers`` and ``asyncio.to_thread``."""

from __future__ import annotations

import sys
import threading
from collections.abc import Callable, Generator
from contextlib import contextmanager

import pytest
from llama_index.core.graph_stores.types import EntityNode, Relation
from llama_index.core.vector_stores.types import VectorStoreQuery

from grafeo_llamaindex import GrafeoPropertyGraphStore

THREADS = 8
TRIALS = 5


@contextmanager
def _frequent_thread_switches() -> Generator[None]:
    """Switch threads as often as possible, to make interleavings likely."""
    previous = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        yield
    finally:
        sys.setswitchinterval(previous)


def _run_together(work: Callable[[], object], threads: int = THREADS) -> list[BaseException]:
    barrier = threading.Barrier(threads)
    errors: list[BaseException] = []

    def run() -> None:
        barrier.wait()
        try:
            work()
        except BaseException as exc:  # collected and asserted on by the caller
            errors.append(exc)

    workers = [threading.Thread(target=run) for _ in range(threads)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    return errors


class TestConcurrentWrites:
    def test_same_nodes_from_several_threads(self) -> None:
        """Upsert looks a node up, then creates it: without a lock two threads both created it."""
        with _frequent_thread_switches():
            for _ in range(TRIALS):
                store = GrafeoPropertyGraphStore()
                errors = _run_together(
                    lambda store=store: [store.upsert_nodes([EntityNode(name=f"n{i}")]) for i in range(30)]
                )
                assert errors == []
                assert store.node_count == 30

    def test_same_relations_from_several_threads(self) -> None:
        with _frequent_thread_switches():
            for _ in range(TRIALS):
                store = GrafeoPropertyGraphStore()
                store.upsert_nodes([EntityNode(name=f"n{i}") for i in range(11)])
                errors = _run_together(
                    lambda store=store: [
                        store.upsert_relations([Relation(label="NEXT", source_id=f"n{i}", target_id=f"n{i + 1}")])
                        for i in range(10)
                    ]
                )
                assert errors == []
                assert store.edge_count == 10

    @pytest.mark.filterwarnings("ignore:Skipping relation:UserWarning")  # endpoints deleted by another thread
    def test_deletes_and_upserts_interleaved(self) -> None:
        store = GrafeoPropertyGraphStore()
        store.upsert_nodes([EntityNode(name=f"n{i}") for i in range(20)])
        store.upsert_relations([Relation(label="R", source_id=f"n{i}", target_id=f"n{i + 1}") for i in range(19)])
        counter = iter(range(1000))

        def work() -> None:
            k = next(counter)
            for i in range(20):
                if (i + k) % 2:
                    store.delete(ids=[f"n{i}"])
                else:
                    store.upsert_nodes([EntityNode(name=f"n{i}")])
                    store.upsert_relations([Relation(label="R", source_id=f"n{i}", target_id=f"n{(i + 1) % 20}")])

        with _frequent_thread_switches():
            errors = _run_together(work, threads=4)
        assert errors == []
        # Every surviving node is reachable by its ID, once.
        ids = [n.id for n in store.get()]
        assert len(ids) == len(set(ids))
        assert all(len(store.get(ids=[li_id])) == 1 for li_id in ids)


class TestReadsDuringWrites:
    def test_vector_query_while_labels_are_added(self) -> None:
        """vector_query iterated the set of indexed labels while upserts added to it:
        RuntimeError('Set changed size during iteration')."""
        store = GrafeoPropertyGraphStore(embedding_dimensions=4)
        store.upsert_nodes([EntityNode(name="seed", label="l0", embedding=[1.0, 0.0, 0.0, 0.0])])
        query = VectorStoreQuery(query_embedding=[1.0, 0.0, 0.0, 0.0], similarity_top_k=3)
        done = threading.Event()
        errors: list[BaseException] = []

        def read() -> None:
            while not done.is_set():
                try:
                    store.vector_query(query)
                except BaseException as exc:  # asserted on below
                    errors.append(exc)
                    done.set()

        def write() -> None:
            try:
                for i in range(1, 200):
                    store.upsert_nodes([EntityNode(name=f"n{i}", label=f"l{i}", embedding=[0.5, 0.5, 0.0, 0.0])])
            finally:
                done.set()

        with _frequent_thread_switches():
            workers = [threading.Thread(target=read) for _ in range(3)] + [threading.Thread(target=write)]
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join()
        assert errors == []
        assert store.node_count == 200
