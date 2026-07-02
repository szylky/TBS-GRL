from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PreprocessConfig:
    code_dir: str = "02_Data_Preprocessing"
    output_subdir: str = "02_Data_Preprocessing"


@dataclass(frozen=True)
class TemporalConfig:
    code_dir: str = "03_Temporal_Modeling"
    output_subdir: str = "03_Temporal_Modeling"


@dataclass(frozen=True)
class GroupConfig:
    code_dir: str = "04_Group_Classification"
    output_subdir: str = "04_Group_Classification"


@dataclass(frozen=True)
class ModelOutputConfig:
    output_subdir: str = "05_Model_Output"
    model_subdir: str = "models/group"


@dataclass(frozen=True)
class DefaultConfig:
    dataset_dir: str = "01_Dataset"
    output_dir: str = "05_Outputs"
    latest_run_file: str = "latest_run.txt"
    preprocess: PreprocessConfig = PreprocessConfig()
    temporal: TemporalConfig = TemporalConfig()
    group: GroupConfig = GroupConfig()
    model_output: ModelOutputConfig = ModelOutputConfig()


def get_default_config() -> DefaultConfig:
    return DefaultConfig()


def get_run_root(base_dir: str | Path, run_id: str) -> Path:
    config = get_default_config()
    return Path(base_dir) / config.output_dir / run_id


def get_latest_run_path(base_dir: str | Path) -> Path:
    config = get_default_config()
    return Path(base_dir) / config.output_dir / config.latest_run_file
