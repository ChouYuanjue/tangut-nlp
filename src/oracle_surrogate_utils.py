"""Utilities for adapting external oracle-bone ID inventories to the Tangut pipeline.

The current Tangut code assumes that each source-side symbol is a single Python
character so that maximum-forward matching, synthetic replacement, and lexical
coverage can all operate over plain strings. Public oracle-bone resources such
as EVOBC expose external IDs like ``00001`` instead. This module keeps the ID
semantics intact while compiling each ID to a reversible single-codepoint
surrogate in a private-use Unicode range.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable


DEFAULT_SURROGATE_START = 0xF0000
DEFAULT_SURROGATE_END = 0xFFFFD
UNKNOWN_TOKEN = "[UNK]"

_ID_PATTERN = re.compile(r"^(?:ID)?\s*0*(\d+)$", re.IGNORECASE)
_ID_SPLIT_PATTERN = re.compile(r"[\s,;|]+")


def canonicalize_oracle_id(raw: object, width: int = 5) -> str:
    """Normalize external oracle IDs to a deterministic zero-padded format."""
    text = str(raw).strip()
    if not text:
        raise ValueError("Empty oracle ID.")

    match = _ID_PATTERN.match(text)
    if not match:
        raise ValueError(f"Unsupported oracle ID format: {raw!r}")

    return match.group(1).zfill(width)


def load_id_to_char(path: str | Path) -> dict[str, str]:
    """Load a EVOBC-style ID -> modern-character mapping JSON."""
    mapping_path = Path(path)
    with mapping_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"Expected a non-empty dict in {mapping_path}")

    normalized = {}
    for raw_id, raw_char in raw.items():
        symbol_id = canonicalize_oracle_id(raw_id)
        modern_char = str(raw_char).strip()
        if not modern_char:
            continue
        normalized[symbol_id] = modern_char

    if not normalized:
        raise ValueError(f"No usable ID mappings found in {mapping_path}")

    return normalized


def build_char_to_ids(id_to_char: dict[str, str]) -> dict[str, list[str]]:
    """Invert an ID -> char map while preserving deterministic ordering."""
    char_to_ids = defaultdict(list)
    for symbol_id in sorted(id_to_char):
        char_to_ids[id_to_char[symbol_id]].append(symbol_id)
    return dict(char_to_ids)


def build_surrogate_tables(
    id_to_char: dict[str, str],
    start_codepoint: int = DEFAULT_SURROGATE_START,
) -> tuple[dict[str, str], dict[str, str]]:
    """Assign one private-use Unicode codepoint to each external ID."""
    needed = len(id_to_char)
    available = DEFAULT_SURROGATE_END - start_codepoint + 1
    if needed > available:
        raise ValueError(
            "Not enough single-codepoint surrogates in the configured private-use range: "
            f"need {needed}, have {available}."
        )

    id_to_surrogate = {}
    surrogate_to_id = {}
    for offset, symbol_id in enumerate(sorted(id_to_char)):
        surrogate = chr(start_codepoint + offset)
        id_to_surrogate[symbol_id] = surrogate
        surrogate_to_id[surrogate] = symbol_id

    return id_to_surrogate, surrogate_to_id


def build_tangut_style_dictionary(
    id_to_char: dict[str, str],
    id_to_surrogate: dict[str, str],
) -> list[dict]:
    """Materialize a Tangut-style dictionary JSON from oracle IDs."""
    entries = []
    for symbol_id in sorted(id_to_char):
        entries.append(
            {
                "character": id_to_surrogate[symbol_id],
                "external_id": symbol_id,
                "source": "EVOBC",
                "GX": "",
                "GHC": "",
                "LFW": "",
                "explanationEN": "",
                "explanationCN": id_to_char[symbol_id],
            }
        )
    return entries


def build_reward_dict(
    id_to_char: dict[str, str],
    id_to_surrogate: dict[str, str],
) -> dict[str, dict[str, list[str]]]:
    """Create the minimal reward-dictionary structure used by lexical coverage."""
    reward_lookup = {}
    for symbol_id in sorted(id_to_char):
        reward_lookup[id_to_surrogate[symbol_id]] = [id_to_char[symbol_id]]
    return {"1": reward_lookup}


def parse_id_sequence(raw_value: object) -> list[str]:
    """Parse a whitespace- or comma-delimited ID sequence from JSON content."""
    if isinstance(raw_value, list):
        parts = [str(item).strip() for item in raw_value if str(item).strip()]
    elif isinstance(raw_value, str):
        stripped = raw_value.strip()
        if not stripped:
            return []
        parts = [part for part in _ID_SPLIT_PATTERN.split(stripped) if part]
    else:
        raise TypeError(f"Unsupported ID sequence type: {type(raw_value).__name__}")

    return [canonicalize_oracle_id(part) for part in parts]


def encode_id_sequence(
    symbol_ids: Iterable[str],
    id_to_surrogate: dict[str, str],
    *,
    strict: bool = False,
    unknown_token: str = UNKNOWN_TOKEN,
) -> str:
    """Encode an external ID sequence as a surrogate string."""
    encoded = []
    for raw_id in symbol_ids:
        symbol_id = canonicalize_oracle_id(raw_id)
        surrogate = id_to_surrogate.get(symbol_id)
        if surrogate is None:
            if strict:
                raise KeyError(f"Unknown oracle ID: {symbol_id}")
            encoded.append(unknown_token)
        else:
            encoded.append(surrogate)
    return "".join(encoded)


def decode_surrogate_sequence(
    text: str,
    surrogate_to_id: dict[str, str],
    *,
    unknown_token: str = UNKNOWN_TOKEN,
) -> list[str]:
    """Decode a surrogate string back to external IDs."""
    decoded = []
    index = 0
    while index < len(text):
        if text.startswith(unknown_token, index):
            decoded.append(unknown_token)
            index += len(unknown_token)
            continue

        symbol = text[index]
        decoded.append(surrogate_to_id.get(symbol, unknown_token))
        index += 1

    return decoded

