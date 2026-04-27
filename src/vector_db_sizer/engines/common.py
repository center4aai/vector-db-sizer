import math

from vector_db_sizer.config import ScenarioConfig


def default_quantized_bytes(scenario: ScenarioConfig, n_records: int) -> int:
    q = scenario.index.quantization
    if q == "none":
        return 0
    dim = scenario.embedding.dimensions or 0
    if q == "scalar_int8":
        return n_records * dim
    if q == "int4":
        return n_records * math.ceil(dim / 2)
    if q in {"binary", "bbq"}:
        return n_records * math.ceil(dim / 8)
    if q == "product":
        return n_records * (scenario.index.pq_code_bytes or 0)
    return 0
