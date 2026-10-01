"""Central recipes and explicit validation/test boundaries for V2."""

import hashlib
import json
from pathlib import Path


def read_recipe(path="configs/v2/recipe.json"):
    recipe = json.loads(Path(path).read_text(encoding="utf8"))
    assert recipe["version"] == 2
    assert recipe["training"]["seeds"] == [0, 1, 2]
    assert 100000 <= recipe["evaluation"]["validation_first_seed"] < 200000
    assert recipe["evaluation"]["reserved_test_first_seed"] >= 300000
    assert recipe["training"]["execute_steps"] <= recipe["training"]["horizon"]
    return recipe


def json_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(8 * 1024**2):
            digest.update(chunk)
    return digest.hexdigest()


def source_hash(path):
    """Freeze code while allowing Git's Windows CRLF/LF checkout conversion."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
