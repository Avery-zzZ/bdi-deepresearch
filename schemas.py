from dataclasses import dataclass


@dataclass
class DeepResearchContext:
    kb_ragflow_dataset_name: str | None = None
    kb_ragflow_dataset_id: str | None = None
