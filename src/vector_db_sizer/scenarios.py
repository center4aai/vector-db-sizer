from dataclasses import dataclass
from pathlib import Path

from vector_db_sizer.config import MultiScenarioConfig, ScenarioConfig, load_yaml


@dataclass(frozen=True)
class LoadedScenarios:
    scenarios: list[ScenarioConfig]
    is_multi_scenario: bool


def load_scenarios(path: Path) -> LoadedScenarios:
    data = load_yaml(path)
    if "scenarios" in data:
        multi = MultiScenarioConfig.model_validate(data)
        return LoadedScenarios(scenarios=multi.scenarios, is_multi_scenario=True)
    return LoadedScenarios(
        scenarios=[ScenarioConfig.model_validate(data)],
        is_multi_scenario=False,
    )
