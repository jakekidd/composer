import json
import os
import random
from typing import Any
from jsonschema import validate, ValidationError
from .frozen import FrozenDict

config_schema = {
    "type": "object",
    "properties": {
        "start": {"type": "integer"},
        "end": {"type": "integer"},
        "stable": {"type": "string"},
        "tokens": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "initial": {"type": "number"},
                    "volatility": {"type": "number"},
                    "popularity": {"type": "number"},
                    "total": {"type": "integer"},
                    "slope": {"type": "number"},
                    "cycle": {
                        "type": "object",
                        "properties": {
                            "period": {"type": "integer"},
                            "weight": {"type": "number"}
                        },
                        "required": ["period", "weight"]
                    },
                    "seed": {"type": "integer"}
                },
                "required": ["name", "initial", "volatility", "popularity", "total", "slope"]
            }
        }
    },
    "required": ["start", "end", "stable", "tokens"]
}

def load_config(config_path: str = 'config.json') -> FrozenDict[str, Any]:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as file:
        config = json.load(file)

    # Fill in default values if some are missing
    default_config = {
        "start": 1577836800,
        "end": 1735689600,
        "stable": "USDC",
        "tokens": []
    }

    for key, value in default_config.items():
        if key not in config:
            config[key] = value

    try:
        validate(instance=config, schema=config_schema)
    except ValidationError as e:
        raise ValueError(f"Invalid configuration: {e.message}")

    for token in config["tokens"]:
        if "seed" not in token or token["seed"] is None:
            token["seed"] = random.randint(0, 1e6)

    frozen = FrozenDict(config)
    return frozen
