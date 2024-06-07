import logging
from dataclasses import dataclass, field
from typing import Any, Dict

import yaml


@dataclass
class TiramisuConfig:
    is_new_tiramisu: bool = False
    max_runs: int = 30


@dataclass
class TiraLibConfig:
    tiramisu: TiramisuConfig
    workspace: str = "workspace"
    env_vars: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.tiramisu, dict):
            self.tiramisu = TiramisuConfig(**self.tiramisu)


def read_yaml_file(path):
    with open(path) as yaml_file:
        return yaml_file.read()


def parse_yaml_file(yaml_string: str) -> Dict[Any, Any]:
    return yaml.safe_load(yaml_string)


def dict_to_config(parsed_yaml: Dict[Any, Any]) -> TiraLibConfig:
    tiramisu = (
        TiramisuConfig(**parsed_yaml["tiramisu"])
        if "tiramisu" in parsed_yaml
        else TiramisuConfig()
    )
    tiralib = parsed_yaml["tiralib"] if "tiralib" in parsed_yaml else {}
    env_vars = parsed_yaml["env_vars"] if "env_vars" in parsed_yaml else {}
    return TiraLibConfig(
        **tiralib,
        env_vars=env_vars,
        tiramisu=tiramisu,
    )


class BaseConfig:
    base_config = None

    @classmethod
    def init(cls, config_yaml="config.yaml", logging_level=logging.DEBUG):
        parsed_yaml_dict = parse_yaml_file(read_yaml_file(config_yaml))
        BaseConfig.base_config = dict_to_config(parsed_yaml_dict)
        logging.basicConfig(
            level=logging_level,
            format="|%(asctime)s|%(levelname)s| %(message)s",
        )

    @classmethod
    def from_tiralib_config(
        cls, tiralib_config: TiraLibConfig, logging_level=logging.DEBUG
    ):
        BaseConfig.base_config = tiralib_config
        logging.basicConfig(
            level=logging_level,
            format="|%(asctime)s|%(levelname)s| %(message)s",
        )
