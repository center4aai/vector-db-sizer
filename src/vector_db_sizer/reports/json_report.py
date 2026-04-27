from vector_db_sizer.models import EstimateResult, MultiEstimateResult


def to_json(result: EstimateResult | list[EstimateResult]) -> str:
    if isinstance(result, list):
        return MultiEstimateResult(results=result).model_dump_json(indent=2)
    return result.model_dump_json(indent=2)
