"""Config module for TiraLib."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class TiraLibCppConfig:
    """Config for TiraLibCpp."""

    use_sqlite: bool = False


@dataclass
class Dependencies:
    """Config for dependencies."""

    includes: list[str] = field(default_factory=list)
    libs: list[str] = field(default_factory=list)


@dataclass
class TiraLibConfig:
    """Config for TiraLib."""

    workspace: str = "workspace"
    env_vars: Dict[str, str] = field(default_factory=dict)
    tiralib_cpp: TiraLibCppConfig = field(default_factory=TiraLibCppConfig)
    dependencies: Dependencies = field(default_factory=Dependencies)


def read_yaml_file(path):
    """Read a yaml file and return its content as a string."""
    with open(path) as yaml_file:
        return yaml_file.read()


def parse_yaml_file(yaml_string: str) -> Dict[Any, Any]:
    """Parse a yaml string and return a dictionary."""
    return yaml.safe_load(yaml_string)


def dict_to_config(parsed_yaml: Dict[Any, Any]) -> TiraLibConfig:
    """Convert a dictionary to a TiraLibConfig object."""
    env_vars = parsed_yaml["env_vars"] if "env_vars" in parsed_yaml else {}
    tiralibcpp = (
        TiraLibCppConfig(**parsed_yaml["tiralib_cpp"])
        if "tiralib_cpp" in parsed_yaml
        else TiraLibCppConfig()
    )
    deps = (
        Dependencies(**parsed_yaml["dependencies"])
        if "dependencies" in parsed_yaml
        else Dependencies()
    )
    return TiraLibConfig(
        workspace=parsed_yaml["workspace"]
        if "workspace" in parsed_yaml
        else "workspace",
        env_vars=env_vars,
        tiralib_cpp=tiralibcpp,
        dependencies=deps,
    )


class BaseConfig:
    """Base config class."""

    base_config = None

    @classmethod
    def init(cls, config_yaml="config.yaml", logging_level=logging.DEBUG):
        """Initialize the config."""
        parsed_yaml_dict = parse_yaml_file(read_yaml_file(config_yaml))
        BaseConfig.base_config = dict_to_config(parsed_yaml_dict)
        base_logger = logging.getLogger(__name__.split(".")[0])
        base_logger.setLevel(logging_level)
        Path(BaseConfig.base_config.workspace).mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_tiralib_config(
        cls, tiralib_config: TiraLibConfig, logging_level=logging.DEBUG
    ):
        """Initialize the config from a TiraLibConfig object."""
        BaseConfig.base_config = tiralib_config
        base_logger = logging.getLogger(__name__.split(".")[0])
        base_logger.setLevel(logging_level)
        Path(BaseConfig.base_config.workspace).mkdir(parents=True, exist_ok=True)
