from vector_db_sizer.engines.base import EngineProfile
from vector_db_sizer.engines.elasticsearch import ElasticsearchProfile
from vector_db_sizer.engines.generic import GenericProfile
from vector_db_sizer.engines.milvus import MilvusProfile
from vector_db_sizer.engines.opensearch import OpenSearchProfile
from vector_db_sizer.engines.pgvector import PgvectorProfile
from vector_db_sizer.engines.pinecone import PineconeProfile
from vector_db_sizer.engines.qdrant import QdrantProfile
from vector_db_sizer.engines.weaviate import WeaviateProfile

ENGINE_NAMES = [
    "generic",
    "pgvector",
    "qdrant",
    "milvus",
    "elasticsearch",
    "opensearch",
    "weaviate",
    "pinecone",
]

_REGISTRY: dict[str, EngineProfile] = {
    "generic": GenericProfile(),
    "pgvector": PgvectorProfile(),
    "qdrant": QdrantProfile(),
    "milvus": MilvusProfile(),
    "elasticsearch": ElasticsearchProfile(),
    "opensearch": OpenSearchProfile(),
    "weaviate": WeaviateProfile(),
    "pinecone": PineconeProfile(),
}


def get_engine_profile(engine_name: str) -> EngineProfile:
    try:
        return _REGISTRY[engine_name]
    except KeyError as exc:
        raise ValueError(f"Unknown engine: {engine_name}") from exc


def list_engine_names() -> list[str]:
    return ENGINE_NAMES.copy()


def explain_engine(engine_name: str) -> str:
    return get_engine_profile(engine_name).explain()
