"""Graph RAG on the Silicon Valley graph with Grafeo, LlamaIndex and Mistral models.

Loads the Silicon Valley (Pied Piper) dataset from grafeo.ai into a GrafeoPropertyGraphStore,
embeds each entity with Mistral, asks questions through a LlamaIndex query engine, and then adds
a document of its own. All data lives in a single .grafeo file.

Usage:
    uv add grafeo-llamaindex "grafeo-datasets[grafeo]" llama-index-llms-mistralai llama-index-embeddings-mistralai
    export MISTRAL_API_KEY=...
    uv run python basic_graph_rag.py
"""

from __future__ import annotations

from grafeo_datasets import silicon_valley
from llama_index.core import Document, PropertyGraphIndex, Settings
from llama_index.core.graph_stores.types import EntityNode, Relation
from llama_index.core.indices.property_graph import SimpleLLMPathExtractor
from llama_index.embeddings.mistralai import MistralAIEmbedding
from llama_index.llms.mistralai import MistralAI

from grafeo_llamaindex import GrafeoPropertyGraphStore

# 1. Models: an alias such as mistral-small-latest, since dated model IDs are rejected
Settings.llm = MistralAI(model="mistral-small-latest")
Settings.embed_model = MistralAIEmbedding(model_name="mistral-embed")

# 2. The store reads the embedding dimensions (1024) from the model
graph_store = GrafeoPropertyGraphStore(db_path="./pied_piper.grafeo", embed_model=Settings.embed_model)

# 3. Load Pied Piper, its people, products and rivals, embedding each entity's description
nodes = silicon_valley.nodes()
names = {n["key"]: n["name"] for n in nodes}
embeddings = Settings.embed_model.get_text_embedding_batch([f"{n['name']}: {n['description']}" for n in nodes])
graph_store.upsert_nodes(
    [
        EntityNode(
            name=n["name"], label=n["label"], properties={"description": n["description"], **n["props"]}, embedding=e
        )
        for n, e in zip(nodes, embeddings, strict=True)
    ]
)
graph_store.upsert_relations(
    [
        Relation(label=e["type"], source_id=names[e["source"]], target_id=names[e["target"]], properties=e["props"])
        for e in silicon_valley.edges()
    ]
)

# 4. Ask questions; use_async=False avoids LlamaIndex's "Event loop is closed" with API clients
index = PropertyGraphIndex.from_existing(property_graph_store=graph_store, use_async=False)
query_engine = index.as_query_engine(include_properties=True)
for question in [
    "Does Jian-Yang's fridge talk?",
    "How is Son of Anton related to Anton?",
    "Who invested in Pied Piper?",
]:
    print(f"Q: {question}\nA: {query_engine.query(question)}\n")

# 5. Add a document of your own. The LLM extracts its entities and relations; a later insert needs
#    an index whose extractor has a new LLM object (see "Adding documents later" in the README).
inserting_index = PropertyGraphIndex.from_existing(
    property_graph_store=graph_store,
    kg_extractors=[SimpleLLMPathExtractor(llm=MistralAI(model="mistral-small-latest"))],
    embed_model=Settings.embed_model,
    embed_kg_nodes=True,
    use_async=False,
)
inserting_index.insert(Document(text="Jack Barker took over from Gavin Belson as CEO of Hooli."))
print("After the insert:", graph_store.node_count, "nodes,", graph_store.edge_count, "relations")

# 6. The graph persists in ./pied_piper.grafeo. Reopen it later with the same embedding model:
#    graph_store = GrafeoPropertyGraphStore(db_path="./pied_piper.grafeo", embed_model=Settings.embed_model)
#    index = PropertyGraphIndex.from_existing(
#        property_graph_store=graph_store, embed_model=Settings.embed_model, use_async=False
#    )
graph_store.close()
