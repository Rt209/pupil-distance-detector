from __future__ import annotations

from pupil_distance_detector.pipelines.base import BasePipeline
from pupil_distance_detector.pipelines.pipeline_v1 import PipelineV1
from pupil_distance_detector.pipelines.pipeline_v2 import PipelineV2
from pupil_distance_detector.pipelines.pipeline_v3 import PipelineV3


PIPELINE_REGISTRY: dict[str, type[BasePipeline]] = {
    "v1": PipelineV1,
    "v2": PipelineV2,
    "v3": PipelineV3,
}


def available_pipelines() -> list[str]:
    return sorted(PIPELINE_REGISTRY)


def build_pipeline(name: str) -> BasePipeline:
    try:
        return PIPELINE_REGISTRY[name.lower()]()
    except KeyError as exc:
        available = ", ".join(available_pipelines())
        raise ValueError(f"Unsupported pipeline '{name}'. Available pipelines: {available}") from exc
