"""Read the versioned per-case region-evaluation contracts without model imports."""
from __future__ import annotations

import json
import re
from pathlib import Path


DEFAULT_CONTRACTS_DIR = Path(__file__).resolve().parents[2] / "protocol/contracts"
COLORS = ("red", "orange", "yellow", "green", "blue", "purple", "brown",
          "gray/silver", "white", "black", "other", "none")


def load_contract(case: str, contracts_dir=None) -> dict:
    """Load and validate one contract; returned data is unchanged from its JSON."""
    if not re.fullmatch(r"[A-F]0[1-8]", case):
        raise ValueError(f"Invalid case id: {case!r}")
    directory = Path(contracts_dir) if contracts_dir is not None else DEFAULT_CONTRACTS_DIR
    path = directory / f"{case}.json"
    contract = json.loads(path.read_text())
    if contract.get("case") != case:
        raise ValueError(f"{path}: case does not match filename")
    if not isinstance(contract.get("noun"), str) or not contract["noun"].strip():
        raise ValueError(f"{path}: nonempty noun is required")
    if not contract.get("skip_reason"):
        if not isinstance(contract.get("gt_root_pattern"), str) or not contract["gt_root_pattern"]:
            raise ValueError(f"{path}: gt_root_pattern is required")
        re.compile(contract["gt_root_pattern"])
    attributes = contract.get("attributes")
    if not isinstance(attributes, list):
        raise ValueError(f"{path}: attributes must be a list")
    attribute_ids = set()
    for attribute in attributes:
        attribute_id = attribute.get("id")
        if not isinstance(attribute_id, str) or not attribute_id or attribute_id in attribute_ids:
            raise ValueError(f"{path}: empty or duplicate attribute id {attribute_id!r}")
        attribute_ids.add(attribute_id)
        entities = attribute.get("entities")
        if not isinstance(entities, list) or not entities or not all(isinstance(x, str) and x for x in entities):
            raise ValueError(f"{path}: {attribute_id} needs nonempty entity identifiers")
        if not isinstance(attribute.get("surface"), str) or not attribute["surface"]:
            raise ValueError(f"{path}: {attribute_id} needs a surface")
        if attribute.get("target") not in COLORS:
            raise ValueError(f"{path}: {attribute_id} target is not a color family")
        if attribute.get("source") is not None and attribute["source"] not in COLORS:
            raise ValueError(f"{path}: {attribute_id} source is not a color family")
        if attribute.get("pattern") is not None and not isinstance(attribute["pattern"], str):
            raise ValueError(f"{path}: {attribute_id} pattern must be text")
        if not isinstance(attribute.get("score_color", True), bool):
            raise ValueError(f"{path}: {attribute_id} score_color must be boolean")
        if not attribute.get("score_color", True) and not attribute.get("pattern"):
            raise ValueError(f"{path}: {attribute_id} without color scoring needs a pattern")
    return contract


def load_contracts(contracts_dir=None) -> dict[str, dict]:
    """Load all contract files in filename order, keyed by case identifier."""
    directory = Path(contracts_dir) if contracts_dir is not None else DEFAULT_CONTRACTS_DIR
    return {path.stem: load_contract(path.stem, directory)
            for path in sorted(directory.glob("[A-F]0[1-8].json"))}
