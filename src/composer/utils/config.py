import json
import os
import random
from typing import Any, Dict
from jsonschema import validate, ValidationError

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

def load_config(config_path: str = 'config.json') -> Dict[str, Any]:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as file:
        config = json.load(file)

    try:
        validate(instance=config, schema=config_schema)
    except ValidationError as e:
        raise ValueError(f"Invalid configuration: {e.message}")

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

    for token in config["tokens"]:
        if "seed" not in token or token["seed"] is None:
            token["seed"] = random.randint(0, 1000000)

    return config
