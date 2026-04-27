import pytest

from vector_db_sizer.engines.registry import get_engine_profile, list_engine_names


def test_list_engine_names_exact_order() -> None:
    assert list_engine_names() == [
        "generic",
        "pgvector",
        "qdrant",
        "milvus",
        "elasticsearch",
        "opensearch",
        "weaviate",
        "pinecone",
    ]


def test_all_engines_resolve() -> None:
    for engine in list_engine_names():
        assert get_engine_profile(engine).name == engine


def test_unknown_engine_fails_readably() -> None:
    with pytest.raises(ValueError, match="Unknown engine"):
        get_engine_profile("unknown")
