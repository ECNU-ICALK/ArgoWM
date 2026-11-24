"""Per-field summaries, grouped diagnostics and observable consistency."""
from __future__ import annotations

from collections import Counter, defaultdict
import re
import statistics
from typing import Any

from .metrics import classification, direction_metrics, numeric, ratio, stage_metrics


def explicit_metric_direction(text: str, key: str) -> set[str]:
    """Find direct metric predicates, excluding modifiers of other plant parts."""
    anchor = {
        "culm": r"(?:culm height|plant height|height|culm|plant)",
        "leaves_area": r"(?:leaf area|visible (?:leaf )?area|foliage area)",
        "yellow_brown_fraction": r"yellow.brown (?:fraction|proportion|coloration|coloring)",
    }[key]
    # Do not cross clauses or nouns: "lower leaves" is a location and "greater
    # share of green foliage" is a different quantity, not the named metric.
    auxiliary = r"(?:(?:is expected to (?:be )?|will be |will |has |have |is |are |appears |seems |becomes |become |shows |show ))?"
    modifier = r"(?:(?:visibly|slightly|clearly|substantially|much|now)\s+){0,2}"
    predicates = {
        "increase": r"(?:increase[ds]?|higher|greater|taller|expand(?:ed|s)?|more apparent|more prominent)",
        "decrease": r"(?:decrease[ds]?|lower|smaller|shorter|reduce[ds]?|less apparent|less prominent)",
    }
    return {direction for direction, predicate in predicates.items()
            if re.search(r"\b" + anchor + r"\s+" + auxiliary + modifier + predicate + r"\b", text, re.I)}


def summarize(rows: list[dict[str, Any]], stages: list[str], contract: dict[str, Any]) -> dict[str, Any]:
    stage = {role: stage_metrics([r[side] for r in rows], [r["parsed"]["stages"][role] for r in rows], stages)
             for role, side in (("current", "source_stage"), ("future", "target_stage"))}
    transitions, reference_transitions = [], []
    for r in rows:
        current, future = (r["parsed"]["stages"][k] for k in ("current", "future"))
        transitions.append(None if current is None or future is None else ("same" if current == future else ("forward" if stages.index(future) > stages.index(current) else "backward")))
        reference_transitions.append("same" if r["source_stage"] == r["target_stage"] else "forward")
    stage["transition"] = classification(reference_transitions, transitions, ["same", "forward"])
    stage["transition"].update(backward_count=transitions.count("backward"),
                                backward_rate_all=ratio(transitions.count("backward"), len(rows)),
                                parsed_transition_count=sum(t is not None for t in transitions))
    num = {}
    for key in contract["continuous_metrics"]:
        fields = [r["numeric"][key] for r in rows if key in r["numeric"]]
        if not fields:
            continue
        record: dict[str, Any] = {"n_applicable": len(fields)}
        for role in ("source", "target", "declared_change", "derived_change"):
            truth = [f["reference"][role] for f in fields]
            prediction = [f["prediction"][role] for f in fields]
            record[role] = numeric(truth, prediction, tolerances=contract["numeric_scoring"]["tolerances_pp"])
            if role in {"declared_change", "derived_change"}:
                record[role]["direction"] = {
                    str(band): direction_metrics(truth, prediction, deadband=band)
                    for band in [contract["numeric_scoring"]["direction_deadband_pp"], *contract["numeric_scoring"]["direction_deadband_sensitivity_pp"]]}
        residuals = [f["arithmetic_residual"] for f in fields if f["arithmetic_residual"] is not None]
        consistent = sum(f["arithmetic_consistent"] is True for f in fields)
        record["arithmetic"] = {"n_applicable": len(fields), "n_parsed_triplets": len(residuals),
                                 "consistent": consistent, "consistent_fraction_all": ratio(consistent, len(fields)),
                                 "consistent_fraction_parsed": ratio(consistent, len(residuals)),
                                 "mean_absolute_residual_pp": statistics.fmean(abs(v) for v in residuals) if residuals else None}
        record["unscorable_assertions"] = sum(len(f["unscorable_assertions"]) for f in fields)
        num[key] = record
    panicle = {}
    for role in ("source", "target"):
        fields = [r["numeric"]["panicle"] for r in rows if r["numeric"]["panicle"]["reference"][role] is not None]
        panicle[role] = classification([f["reference"][role] for f in fields], [f["prediction"][role] for f in fields], [0, 1])
        panicle[role]["positive_class"] = panicle[role]["per_class"][1]
    valid = [r["numeric"]["panicle"] for r in rows if r["numeric"]["panicle"]["reference"]["derived_change"] is not None]
    def label(f: dict[str, Any], side: str) -> str | None:
        a, b = f[side]["source"], f[side]["target"]
        return f"{int(a)}_to_{int(b)}" if a is not None and b is not None else None
    truth, prediction = [label(f, "reference") for f in valid], [label(f, "prediction") for f in valid]
    panicle["transition"] = classification(truth, prediction, ["0_to_0", "0_to_1", "1_to_1"])
    panicle["transition"]["reverse_count"] = prediction.count("1_to_0")
    panicle["emergence"] = classification([int(t == "0_to_1") for t in truth], [int(p == "0_to_1") if p is not None else None for p in prediction], [0, 1])
    panicle["emergence"]["positive_class"] = panicle["emergence"]["per_class"][1]
    panicle["arithmetic"] = {"n_applicable": len(rows), "n_parsed_triplets": sum(r["numeric"]["panicle"]["arithmetic_residual"] is not None for r in rows),
                              "consistent": sum(r["numeric"]["panicle"]["arithmetic_consistent"] is True for r in rows)}
    expected_points = sum(sum(len(v) for v in r["parsed"]["qualitative"].values()) for r in rows)
    covered_points = sum(sum(bool(text) for text in fields.values()) for r in rows for fields in r["parsed"]["qualitative"].values())
    return {"rows": len(rows), "plants": len({r["plant_id"] for r in rows}),
            "structure": {"strict_valid": sum(r["strict_structure_valid"] for r in rows),
                          "strict_valid_rate": ratio(sum(r["strict_structure_valid"] for r in rows), len(rows)),
                          "expected_qualitative_points": expected_points, "covered_qualitative_points": covered_points,
                          "point_coverage": ratio(covered_points, expected_points),
                          "issue_counts": dict(Counter(i for r in rows for i in r["issues"]))},
            "stages": stage, "numeric": num, "panicle": panicle}


def consistency(rows: list[dict[str, Any]], stages: list[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    sources: dict[str, list[dict[str, Any]]] = defaultdict(list)
    flags = []
    for r in rows:
        groups[r["pair_group_id"]].append(r)
        sources[r["source_path"]].append(r)
        # These deliberately narrow rules flag candidates, not proven factual errors.
        for key, field in r["numeric"].items():
            if key not in {"culm", "leaves_area", "yellow_brown_fraction"}:
                continue
            delta = field["prediction"]["derived_change"]
            if delta is None or abs(delta) <= 1:
                continue
            text = r["parsed"]["qualitative"]["Target phenotype"].get(key) or ""
            directions = explicit_metric_direction(text, key)
            if ("increase" in directions and delta < -1) or ("decrease" in directions and delta > 1):
                flags.append({"request_id": r["request_id"], "field": key, "text": text, "derived_change": delta,
                              "kind": "explicit_direction_candidate", "status": "automatic_flag_unconfirmed"})
    cross = {}
    for role in ("current", "future", "panicle_source", "panicle_target"):
        counts = Counter()
        for group in groups.values():
            if len(group) != 2 or {r["view"] for r in group} != {"front", "top"}:
                raise ValueError("invalid paired-view group")
            if role.startswith("panicle_"):
                endpoint = role.split("_", 1)[1]
                ps = [r["numeric"]["panicle"]["prediction"][endpoint] for r in group]
                ys = [r["numeric"]["panicle"]["reference"][endpoint] for r in group]
            else:
                ps = [r["parsed"]["stages"][role] for r in group]
                ys = [r["source_stage" if role == "current" else "target_stage"] for r in group]
            if any(y is None for y in ys):
                counts["reference_unavailable"] += 1
                continue
            if ys[0] != ys[1]:
                raise ValueError("paired reference labels differ")
            if any(p is None for p in ps):
                counts["missing_prediction"] += 1
                continue
            counts["both_parsed"] += 1
            counts["agree"] += int(ps[0] == ps[1])
            counts["both_correct"] += int(all(p == y for p, y in zip(ps, ys)))
            counts["agree_both_wrong"] += int(ps[0] == ps[1] and ps[0] != ys[0])
            counts["only_one_correct"] += int(sum(p == y for p, y in zip(ps, ys)) == 1)
        denominator = len(groups) - counts["reference_unavailable"]
        cross[role] = {**counts, "n_reference_groups": denominator,
                       "agreement_rate_all": ratio(counts["agree"], denominator),
                       "agreement_rate_parsed": ratio(counts["agree"], counts["both_parsed"])}
    repeated = [values for values in sources.values() if len(values) > 1]
    source_diagnostics = []
    for group in repeated:
        current = [r["parsed"]["stages"]["current"] for r in group]
        values = {"source_path": group[0]["source_path"], "plant_id": group[0]["plant_id"], "view": group[0]["view"],
                  "n_predictions": len(group), "stage_missing": sum(v is None for v in current),
                  "stage_disagreement": len({v for v in current if v is not None}) > 1, "numeric": {}}
        for key in group[0]["numeric"]:
            nums = [r["numeric"][key]["prediction"]["source"] for r in group]
            valid = [v for v in nums if v is not None]
            values["numeric"][key] = {"n_parsed": len(valid), "range": max(valid)-min(valid) if len(valid)>1 else None,
                                       "std": statistics.pstdev(valid) if len(valid)>1 else None}
        source_diagnostics.append(values)
    dedup_numeric = {}
    for key in sorted({key for r in rows for key in r["numeric"] if key != "panicle"}):
        errors, coverages = [], []
        for group in sources.values():
            if key not in group[0]["numeric"] or group[0]["numeric"][key]["reference"]["source"] is None:
                continue
            vals = [abs(r["numeric"][key]["prediction"]["source"]-r["numeric"][key]["reference"]["source"])
                    for r in group if r["numeric"][key]["prediction"]["source"] is not None]
            coverages.append(len(vals)/len(group))
            if vals:
                errors.append(statistics.fmean(vals))
        dedup_numeric[key] = {"source_weighted_mae_pp": statistics.fmean(errors) if errors else None,
                              "source_weighted_coverage": statistics.fmean(coverages) if coverages else None,
                              "n_reference_sources": len(coverages), "n_sources_with_parsed_predictions": len(errors)}
    unique_stage = classification([r["source_stage"] for r in rows], [r["parsed"]["stages"]["current"] for r in rows], stages,
                                  weights=[1 / len(sources[r["source_path"]]) for r in rows])
    return {"cross_view": cross, "unique_source_images": len(sources), "repeated_source_groups": len(repeated),
            "repeated_stage_disagreement_groups": sum(r["stage_disagreement"] for r in source_diagnostics),
            "repeated_stage_disagreement_rate": ratio(sum(r["stage_disagreement"] for r in source_diagnostics), len(repeated)),
            "repeated_groups_with_missing_stage": sum(r["stage_missing"] > 0 for r in source_diagnostics),
            "source_weighted_current_stage": unique_stage, "source_weighted_numeric": dedup_numeric,
            "repeated_source_details": source_diagnostics, "automatic_text_numeric_flags": len(flags),
            "flag_interpretation": "conservative automatic candidates; not human-verified contradictions"}, flags


def flattened(summary: dict[str, Any]) -> dict[str, Any]:
    record = {"n_rows": summary["rows"], "n_plants": summary["plants"],
              "strict_structure_rate": summary["structure"]["strict_valid_rate"]}
    for role in ("current", "future"):
        for field in ("accuracy", "macro_f1", "coverage", "ordinal_mae", "signed_stage_bias"):
            record[f"stage_{role}_{field}"] = summary["stages"][role][field]
    for key, fields in summary["numeric"].items():
        for role in ("source", "target", "declared_change", "derived_change"):
            for metric in ("mae", "coverage", "n_reference", "signed_bias"):
                record[f"{key}_{role}_{metric}"] = fields[role][metric]
    for role in ("source", "target", "emergence"):
        record[f"panicle_{role}_accuracy"] = summary["panicle"][role]["accuracy"]
        record[f"panicle_{role}_positive_f1"] = summary["panicle"][role]["positive_class"]["f1"]
    return record
