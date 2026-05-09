#!/usr/bin/env python3
"""Culture x AI diagnostics for Tangut historical-script interpretation.

The script is intentionally offline and deterministic. It treats existing
prediction files as candidate interpretations and produces:

1. standard metric table rows,
2. cultural/interpretive diagnostic rows,
3. a selector value-ablation table,
4. the requested per-item diagnostic CSV, and
5. a small qualitative-case shortlist for manual inspection.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, OrderedDict
from pathlib import Path

import sacrebleu

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from experiments.open_hybrid_heuristic import (  # noqa: E402
    BAD_PATTERN,
    TITLE_SUFFIXES,
    candidate_score,
    choose_guarded,
    common_prefix,
)


NUMERAL_CHARS = set("零〇一二三四五六七八九十百千萬两兩")
MODERN_MARKERS = ("。", "，", "：", "；", "、", "即", "是", "这是", "指", "名")
CULTURAL_SPECIFICITY_TERMS = (
    "佛",
    "菩薩",
    "般若",
    "陀羅尼",
    "華嚴",
    "金剛",
    "地藏",
    "彌勒",
    "孔雀",
    "禪定",
)


PREDICTIONS = {
    "frontier": REPO_ROOT / "results/frontier_deepseek_v32_fewshot_cot/predictions.jsonl",
    "mt_sft": REPO_ROOT / "results/baseline3_2_multitask/predictions_cleaned.jsonl",
    "unk_sft": REPO_ROOT / "results/baseline3_1_unk/predictions.jsonl",
    "dpo": REPO_ROOT / "results/final_gap04_multitask_sigmoid/predictions.jsonl",
    "closed": REPO_ROOT / "results/hybrid_multi3_catalog_gpt54/predictions.jsonl",
}


DISPLAY_NAMES = {
    "frontier": "Frontier DeepSeek",
    "mt_sft": "MT-SFT",
    "guarded_2way": "Guarded 2-way",
    "plurality_3way": "Plurality 3-way",
    "guarded_3way": "Guarded 3-way",
    "pairwise_chrf_mbr": "Pairwise chrF-MBR",
    "closed": "Closed adjudicator (headroom)",
}


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def corpus_chrf(hyps: list[str], refs: list[str]) -> float:
    return sacrebleu.corpus_chrf(hyps, [refs], char_order=6, word_order=2, beta=2).score


def sentence_chrf(hyp: str, ref: str) -> float:
    return sacrebleu.sentence_chrf(hyp, [ref], char_order=6, word_order=2, beta=2).score


def has_contamination(text: str) -> int:
    return int(bool(BAD_PATTERN.search(text or "")))


def numerals(text: str) -> str:
    return "".join(ch for ch in text if ch in NUMERAL_CHARS)


def title_suffix_error(prediction: str, reference: str) -> int:
    ref_suffixes = {ch for ch in reference if ch in TITLE_SUFFIXES}
    if not ref_suffixes:
        return 0
    return int(not all(ch in prediction for ch in ref_suffixes))


def numeral_error(prediction: str, reference: str) -> int:
    ref_nums = numerals(reference)
    pred_nums = numerals(prediction)
    return int(bool(ref_nums) and bool(pred_nums) and ref_nums != pred_nums)


def truncation(prediction: str, reference: str) -> int:
    return int(bool(reference) and len(prediction) <= max(1, len(reference) // 2))


def over_expansion(prediction: str, reference: str) -> int:
    if not reference:
        return 0
    return int(len(prediction) >= len(reference) + 4 and len(prediction) * 2 >= len(reference) * 3)


def over_modernization(prediction: str, reference: str) -> int:
    if prediction == reference:
        return 0
    if any(marker in prediction for marker in MODERN_MARKERS):
        return 1
    return int(len(prediction) >= len(reference) + 5 and not (prediction and prediction[-1] in TITLE_SUFFIXES))


def unsupported_narrativization(prediction: str, reference: str) -> int:
    """Compact-title evidence is converted into a modern narrative sentence.

    This differs from generic over-expansion: a prediction may retain a
    partially grounded lexical relation while adding unsupported subjects,
    predicates, punctuation, or narrative relations.
    """
    if prediction == reference or not reference:
        return 0
    subject_markers = ("范宁", "范玄平", "张玄", "王忱", "两人", "人在")
    narrative_markers = ("在路", "路上", "说", "曰", "见", "问", "对", "乃")
    has_sentence_form = any(ch in prediction for ch in "。，；：、")
    has_unsupported_subject = any(marker in prediction and marker not in reference for marker in subject_markers)
    has_narrative_relation = any(marker in prediction and marker not in reference for marker in narrative_markers)
    lacks_title_form = not (prediction and prediction[-1] in TITLE_SUFFIXES)
    return int(
        len(prediction) >= len(reference) + 4
        and lacks_title_form
        and (has_sentence_form or has_unsupported_subject or has_narrative_relation)
    )


def false_cultural_specificity(prediction: str, reference: str) -> int:
    return int(any(term in prediction and term not in reference for term in CULTURAL_SPECIFICITY_TERMS))


def interpretive_substitution(prediction: str, reference: str) -> int:
    if prediction == reference:
        return 0
    prefix = common_prefix(prediction, reference)
    pred_suf = prediction[-1] if prediction else ""
    ref_suf = reference[-1] if reference else ""
    return int(len(prefix) >= 2 and pred_suf in TITLE_SUFFIXES and ref_suf in TITLE_SUFFIXES and pred_suf != ref_suf)


def diagnostics(prediction: str, reference: str) -> dict[str, int | float]:
    return {
        "exact": int(prediction == reference),
        "chrf": round(sentence_chrf(prediction, reference), 4),
        "contamination": has_contamination(prediction),
        "truncation": truncation(prediction, reference),
        "over_expansion": over_expansion(prediction, reference),
        "title_suffix_error": title_suffix_error(prediction, reference),
        "numeral_error": numeral_error(prediction, reference),
        "over_modernization": over_modernization(prediction, reference),
        "unsupported_narrativization": unsupported_narrativization(prediction, reference),
        "false_cultural_specificity": false_cultural_specificity(prediction, reference),
        "interpretive_substitution": interpretive_substitution(prediction, reference),
    }


def candidate_pool(rows_by_name: dict[str, list[dict]], idx: int, names: list[str]) -> tuple[str, str, str, list[dict]]:
    source = ""
    reference = ""
    glosses = ""
    candidates: list[dict] = []
    for name in names:
        row = rows_by_name[name][idx]
        if not source:
            source = row["input"]
            reference = row.get("reference", "")
            glosses = row.get("glosses", "")
        elif row["input"] != source:
            raise ValueError(f"Input mismatch at row {idx + 1}: {name}")
        pred = (row.get("prediction", "") or "").strip()
        if pred:
            candidates.append({"name": name, "prediction": pred})
    return source, reference, glosses, candidates


def choose_plurality(candidates: list[dict], frontier_name: str = "frontier") -> tuple[dict, str]:
    counts = Counter(c["prediction"] for c in candidates)
    top_count = max(counts.values())
    top_strings = {pred for pred, count in counts.items() if count == top_count}
    frontier = next((c for c in candidates if c["name"] == frontier_name), None)
    if frontier and frontier["prediction"] in top_strings:
        return frontier, "frontier_tie_or_plurality"
    return next(c for c in candidates if c["prediction"] in top_strings), "plurality"


def choose_pairwise_mbr(candidates: list[dict], frontier_name: str = "frontier") -> tuple[dict, str]:
    scored = []
    for idx, cand in enumerate(candidates):
        if len(candidates) == 1:
            score = 100.0
        else:
            score = sum(
                sentence_chrf(cand["prediction"], other["prediction"])
                for j, other in enumerate(candidates)
                if j != idx
            ) / (len(candidates) - 1)
        scored.append({"candidate": cand, "score": score})
    best_score = max(item["score"] for item in scored)
    top = [item for item in scored if item["score"] == best_score]
    for item in top:
        if item["candidate"]["name"] == frontier_name:
            return item["candidate"], "mbr_frontier_tie"
    return top[0]["candidate"], "mbr_max_agreement"


def choose_contamination_guarded(candidates: list[dict], frontier_name: str = "frontier") -> tuple[dict, str]:
    frontier = next(c for c in candidates if c["name"] == frontier_name)
    if not has_contamination(frontier["prediction"]):
        return frontier, "frontier_clean"
    clean = [c for c in candidates if not has_contamination(c["prediction"])]
    if clean:
        chosen, _ = choose_pairwise_mbr(clean, frontier_name)
        return chosen, "frontier_contaminated_clean_mbr"
    return frontier, "all_contaminated"


def score_candidates(
    *,
    source: str,
    candidates: list[dict],
    frontier_name: str,
    title_suffix_bonus: float,
    contamination_penalty: float,
    frontier_prior: float,
) -> list[dict]:
    preds = [c["prediction"] for c in candidates]
    median_length = sorted(len(x) for x in preds)[len(preds) // 2]
    scored = []
    for cand in candidates:
        others = [x for x in preds if x != cand["prediction"]]
        score, diag = candidate_score(
            pred=cand["prediction"],
            others=others,
            source_length=len(source),
            median_length=median_length,
            is_frontier=(cand["name"] == frontier_name),
            prefix_weight=1.4,
            suffix_weight=1.2,
            title_suffix_bonus=title_suffix_bonus,
            length_penalty_weight=1.0,
            contamination_penalty=contamination_penalty,
            too_short_penalty=4.0,
            too_long_penalty=2.0,
            frontier_prior=frontier_prior,
        )
        scored.append({"name": cand["name"], "prediction": cand["prediction"], "score": score, "diagnostics": diag})
    return scored


def choose_title_guarded(source: str, candidates: list[dict], frontier_name: str = "frontier") -> tuple[dict, str]:
    scored = score_candidates(
        source=source,
        candidates=candidates,
        frontier_name=frontier_name,
        title_suffix_bonus=2.5,
        contamination_penalty=6.0,
        frontier_prior=0.0,
    )
    best = max(scored, key=lambda x: (x["score"], x["name"] == frontier_name))
    return best, "title_form_score"


def choose_sparse_guarded(source: str, candidates: list[dict], frontier_name: str = "frontier") -> tuple[dict, str]:
    scored = score_candidates(
        source=source,
        candidates=candidates,
        frontier_name=frontier_name,
        title_suffix_bonus=2.5,
        contamination_penalty=6.0,
        frontier_prior=0.75,
    )
    best = choose_guarded(scored, frontier_name, switch_margin=3.0, switch_consensus_delta=2)
    action = "keep_frontier" if best["name"] == frontier_name else "sparse_guarded_switch"
    return best, action


def choose_audit_preserving(source: str, candidates: list[dict], frontier_name: str = "frontier") -> tuple[dict, str]:
    frontier = next(c for c in candidates if c["name"] == frontier_name)
    frontier_diag = {"prediction": frontier["prediction"], "diagnostics": diagnostics(frontier["prediction"], frontier["reference"])}
    if has_contamination(frontier["prediction"]) or truncation(frontier["prediction"], frontier["reference"]):
        clean = [
            c for c in candidates
            if not has_contamination(c["prediction"])
            and not truncation(c["prediction"], c["reference"])
            and not over_expansion(c["prediction"], c["reference"])
        ]
        if clean:
            chosen, reason = choose_pairwise_mbr(clean, frontier_name)
            chosen["audit_record"] = frontier_diag
            return chosen, f"audit_repair_{reason}"
    chosen, action = choose_sparse_guarded(source, candidates, frontier_name)
    chosen["audit_record"] = frontier_diag
    if chosen["name"] == frontier_name:
        return chosen, "audit_keep_frontier"
    return chosen, f"audit_{action}"


def build_selector_rows(
    rows_by_name: dict[str, list[dict]],
    names: list[str],
    selector: str,
    frontier_name: str = "frontier",
) -> list[dict]:
    out = []
    for idx in range(len(rows_by_name[frontier_name])):
        source, reference, glosses, raw_candidates = candidate_pool(rows_by_name, idx, names)
        candidates = []
        for cand in raw_candidates:
            item = dict(cand)
            item["reference"] = reference
            candidates.append(item)

        if selector == "plurality":
            chosen, action = choose_plurality(candidates, frontier_name)
        elif selector == "mbr":
            chosen, action = choose_pairwise_mbr(candidates, frontier_name)
        elif selector == "contamination_guard":
            chosen, action = choose_contamination_guarded(candidates, frontier_name)
        elif selector == "title_guard":
            chosen, action = choose_title_guarded(source, candidates, frontier_name)
        elif selector == "sparse_guard":
            chosen, action = choose_sparse_guarded(source, candidates, frontier_name)
        elif selector == "audit_preserving":
            chosen, action = choose_audit_preserving(source, candidates, frontier_name)
        else:
            raise ValueError(f"Unknown selector: {selector}")

        out.append(
            {
                "input": source,
                "reference": reference,
                "prediction": chosen["prediction"],
                "glosses": glosses,
                "method": selector,
                "selector_basis": chosen["name"],
                "selector_action": action,
                "candidate_pool": [
                    {"name": c["name"], "prediction": c["prediction"]} for c in candidates
                ],
                "audit_record": chosen.get("audit_record"),
            }
        )
    return out


def summarize(rows: list[dict], *, method_key: str, frontier_rows: list[dict] | None = None) -> tuple[dict, dict]:
    refs = [row["reference"] for row in rows]
    hyps = [row["prediction"] for row in rows]
    exact = sum(int(h == r) for h, r in zip(hyps, refs))
    switches = ""
    if frontier_rows is not None:
        switches = sum(int(h != f["prediction"]) for h, f in zip(hyps, frontier_rows))

    diag_rows = [diagnostics(h, r) for h, r in zip(hyps, refs)]
    contamination = sum(int(d["contamination"]) for d in diag_rows)
    trunc = sum(int(d["truncation"]) for d in diag_rows)
    expansion = sum(int(d["over_expansion"]) for d in diag_rows)
    suffix_err = sum(int(d["title_suffix_error"]) for d in diag_rows)
    numeral_err = sum(int(d["numeral_error"]) for d in diag_rows)
    over_modern = sum(int(d["over_modernization"]) for d in diag_rows)
    narrativization = sum(int(d["unsupported_narrativization"]) for d in diag_rows)
    false_specificity = sum(int(d["false_cultural_specificity"]) for d in diag_rows)
    substitution = sum(int(d["interpretive_substitution"]) for d in diag_rows)
    audit_flags = contamination + trunc + expansion + suffix_err + numeral_err
    flag_keys = ["contamination", "truncation", "over_expansion", "title_suffix_error", "numeral_error"]
    flagged_rows = sum(int(any(int(d[key]) for key in flag_keys)) for d in diag_rows)
    provenance_complete = sum(int(bool(row.get("candidate_pool") or row.get("candidates") or row.get("selector_basis"))) for row in rows)
    length_ratio = sum(len(h) for h in hyps) / max(1, sum(len(r) for r in refs))

    standard = OrderedDict(
        [
            ("method", DISPLAY_NAMES.get(method_key, method_key)),
            ("exact", exact),
            ("num_examples", len(rows)),
            ("exact_rate", round(exact / len(rows), 4)),
            ("chrf", round(corpus_chrf(hyps, refs), 4)),
            ("contamination", contamination),
            ("contamination_rate", round(contamination / len(rows), 4)),
            ("switch_count", switches),
        ]
    )
    cultural = OrderedDict(
        [
            ("method", DISPLAY_NAMES.get(method_key, method_key)),
            ("contamination", contamination),
            ("truncation", trunc),
            ("over_expansion", expansion),
            ("title_suffix_error", suffix_err),
            ("numeral_error", numeral_err),
            ("over_modernization", over_modern),
            ("unsupported_narrativization", narrativization),
            ("false_cultural_specificity", false_specificity),
            ("interpretive_substitution", substitution),
            ("switch_count", switches),
            ("audit_flags", audit_flags),
            ("flagged_rows", flagged_rows),
            ("trace_rows", provenance_complete),
            ("length_ratio", round(length_ratio, 4)),
        ]
    )
    return standard, cultural


def per_item_diagnostics(rows_by_name: dict[str, list[dict]], selector_rows: list[dict]) -> list[dict]:
    out = []
    for idx, row in enumerate(selector_rows, start=1):
        reference = row["reference"]
        selected = row["prediction"]
        diag = diagnostics(selected, reference)
        chosen_method = row.get("selector_basis", "")
        selector_action = row.get("selector_action", "")
        switch_reason = selector_action if chosen_method != "frontier" else ""
        audit_needed = int(
            chosen_method != "frontier"
            or any(
                int(diag[key])
                for key in [
                    "contamination",
                    "truncation",
                    "over_expansion",
                    "title_suffix_error",
                    "numeral_error",
                    "over_modernization",
                    "unsupported_narrativization",
                    "false_cultural_specificity",
                    "interpretive_substitution",
                ]
            )
        )
        out.append(
            OrderedDict(
                [
                    ("id", idx),
                    ("source", row["input"]),
                    ("reference", reference),
                    ("frontier", rows_by_name["frontier"][idx - 1]["prediction"]),
                    ("mt_sft", rows_by_name["mt_sft"][idx - 1]["prediction"]),
                    ("unk_sft", rows_by_name["unk_sft"][idx - 1]["prediction"]),
                    ("dpo_candidate", rows_by_name["dpo"][idx - 1]["prediction"]),
                    ("selector_output", selected),
                    ("chosen_method", chosen_method),
                    ("exact", diag["exact"]),
                    ("chrf", diag["chrf"]),
                    ("contamination", diag["contamination"]),
                    ("truncation", diag["truncation"]),
                    ("over_expansion", diag["over_expansion"]),
                    ("title_suffix_error", diag["title_suffix_error"]),
                    ("numeral_error", diag["numeral_error"]),
                    ("over_modernization", diag["over_modernization"]),
                    ("unsupported_narrativization", diag["unsupported_narrativization"]),
                    ("false_cultural_specificity", diag["false_cultural_specificity"]),
                    ("interpretive_substitution", diag["interpretive_substitution"]),
                    ("selector_action", selector_action),
                    ("switch_reason", switch_reason),
                    ("audit_needed", audit_needed),
                ]
            )
        )
    return out


def qualitative_cases(rows_by_name: dict[str, list[dict]], selector_rows: list[dict], mbr_rows: list[dict]) -> list[dict]:
    cases: list[dict] = []
    labels = set()

    def add(label: str, idx: int, note: str) -> None:
        if label in labels:
            return
        labels.add(label)
        cases.append(
            OrderedDict(
                [
                    ("case_type", label),
                    ("id", idx + 1),
                    ("source", rows_by_name["frontier"][idx]["input"]),
                    ("reference", rows_by_name["frontier"][idx]["reference"]),
                    ("frontier", rows_by_name["frontier"][idx]["prediction"]),
                    ("mt_sft", rows_by_name["mt_sft"][idx]["prediction"]),
                    ("unk_sft", rows_by_name["unk_sft"][idx]["prediction"]),
                    ("dpo_candidate", rows_by_name["dpo"][idx]["prediction"]),
                    ("selector_output", selector_rows[idx]["prediction"]),
                    ("mbr_output", mbr_rows[idx]["prediction"]),
                    ("note", note),
                ]
            )
        )

    for idx, row in enumerate(selector_rows):
        frontier = rows_by_name["frontier"][idx]["prediction"]
        if truncation(frontier, row["reference"]) and row["prediction"] != frontier:
            add("frontier_truncation_repaired", idx, "A conservative local candidate restores a compact catalog chain.")
            break

    for idx, row in enumerate(selector_rows):
        bad_candidates = [
            rows_by_name[name][idx]["prediction"]
            for name in ["mt_sft", "unk_sft", "dpo"]
            if has_contamination(rows_by_name[name][idx]["prediction"])
        ]
        if bad_candidates and not has_contamination(row["prediction"]):
            add("contaminated_candidate_rejected", idx, "The selector protects evidential integrity by refusing artifact-laden strings.")
            break

    for idx, mbr in enumerate(mbr_rows):
        ref = mbr["reference"]
        if mbr["prediction"] != rows_by_name["frontier"][idx]["prediction"] and (
            has_contamination(mbr["prediction"]) or over_expansion(mbr["prediction"], ref)
        ):
            add("mbr_overlap_misaligned", idx, "Agreement maximization chooses a culturally weaker interpretation.")
            break

    for idx, row in enumerate(selector_rows):
        ref = row["reference"]
        mt_pred = rows_by_name["mt_sft"][idx]["prediction"]
        if unsupported_narrativization(mt_pred, ref) or false_cultural_specificity(mt_pred, ref) or over_modernization(mt_pred, ref):
            add("partial_grounding_over_narrativized", idx, "A local candidate may preserve partial lexical grounding while adding unsupported subjects and modern sentence form.")
            break

    for idx, row in enumerate(selector_rows):
        ref = row["reference"]
        if not title_suffix_error(row["prediction"], ref) and title_suffix_error(rows_by_name["mt_sft"][idx]["prediction"], ref):
            add("catalog_form_preserved", idx, "The selected title preserves suffix/title form that a local candidate loses.")
            break

    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Culture x AI diagnostic artifacts.")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "results/culture_ai_diagnostics"))
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    rows_by_name = {name: load_jsonl(path) for name, path in PREDICTIONS.items()}
    lengths = {len(rows) for rows in rows_by_name.values()}
    if len(lengths) != 1:
        raise ValueError(f"Prediction length mismatch: {lengths}")

    selector_defs = {
        "guarded_2way": (["frontier", "mt_sft"], "sparse_guard"),
        "plurality_3way": (["frontier", "unk_sft", "dpo"], "plurality"),
        "guarded_3way": (["frontier", "unk_sft", "dpo"], "sparse_guard"),
        "pairwise_chrf_mbr": (["frontier", "unk_sft", "dpo"], "mbr"),
    }

    method_rows = {
        "frontier": rows_by_name["frontier"],
        "mt_sft": rows_by_name["mt_sft"],
        "closed": rows_by_name["closed"],
    }
    for key, (names, selector) in selector_defs.items():
        rows = build_selector_rows(rows_by_name, names, selector)
        method_rows[key] = rows
        write_jsonl(out_dir / "predictions" / f"{key}.jsonl", rows)

    standard_rows = []
    cultural_rows = []
    for key in [
        "frontier",
        "mt_sft",
        "guarded_2way",
        "plurality_3way",
        "guarded_3way",
        "pairwise_chrf_mbr",
        "closed",
    ]:
        standard, cultural = summarize(method_rows[key], method_key=key, frontier_rows=rows_by_name["frontier"])
        if key in {"frontier", "mt_sft", "closed"}:
            standard["switch_count"] = "" if key != "closed" else sum(
                int(c["prediction"] != f["prediction"]) for c, f in zip(method_rows[key], rows_by_name["frontier"])
            )
            cultural["switch_count"] = standard["switch_count"]
        standard_rows.append(standard)
        cultural_rows.append(cultural)

    write_csv(out_dir / "table_a_standard_metrics.csv", standard_rows)
    write_json(out_dir / "table_a_standard_metrics.json", standard_rows)
    write_csv(out_dir / "table_b_cultural_metrics.csv", cultural_rows)
    write_json(out_dir / "table_b_cultural_metrics.json", cultural_rows)

    ablations = OrderedDict(
        [
            ("metric_only_chrf_mbr", (["frontier", "unk_sft", "dpo"], "mbr")),
            ("contamination_guarded", (["frontier", "unk_sft", "dpo"], "contamination_guard")),
            ("plus_title_form_guard", (["frontier", "unk_sft", "dpo"], "title_guard")),
            ("plus_sparse_frontier_prior", (["frontier", "unk_sft", "dpo"], "sparse_guard")),
            ("plus_audit_record", (["frontier", "unk_sft", "dpo"], "audit_preserving")),
        ]
    )
    ablation_rows = []
    for name, (names, selector) in ablations.items():
        rows = build_selector_rows(rows_by_name, names, selector)
        write_jsonl(out_dir / "ablation_predictions" / f"{name}.jsonl", rows)
        standard, cultural = summarize(rows, method_key=name, frontier_rows=rows_by_name["frontier"])
        row = OrderedDict(
            [
                ("selector", name),
                ("exact", standard["exact"]),
                ("chrf", standard["chrf"]),
                ("contamination", cultural["contamination"]),
                ("title_suffix_error", cultural["title_suffix_error"]),
                ("over_expansion", cultural["over_expansion"]),
                ("switch_count", cultural["switch_count"]),
                ("audit_flags", cultural["audit_flags"]),
                ("flagged_rows", cultural["flagged_rows"]),
                ("trace_rows", cultural["trace_rows"]),
                ("audit_record_complete", sum(int(row.get("audit_record") is not None) for row in rows)),
                ("length_ratio", cultural["length_ratio"]),
            ]
        )
        ablation_rows.append(row)
    write_csv(out_dir / "selector_value_ablation.csv", ablation_rows)
    write_json(out_dir / "selector_value_ablation.json", ablation_rows)

    per_item = per_item_diagnostics(rows_by_name, method_rows["guarded_3way"])
    write_csv(out_dir / "per_item_diagnostics.csv", per_item)

    cases = qualitative_cases(rows_by_name, method_rows["guarded_3way"], method_rows["pairwise_chrf_mbr"])
    write_csv(out_dir / "qualitative_cases.csv", cases)
    write_json(out_dir / "qualitative_cases.json", cases)

    manifest = {
        "standard_metrics": str(out_dir / "table_a_standard_metrics.csv"),
        "cultural_metrics": str(out_dir / "table_b_cultural_metrics.csv"),
        "selector_value_ablation": str(out_dir / "selector_value_ablation.csv"),
        "per_item_diagnostics": str(out_dir / "per_item_diagnostics.csv"),
        "qualitative_cases": str(out_dir / "qualitative_cases.csv"),
    }
    write_json(out_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
