"""Local BGE scoring of aligned qualitative fields, with a resumable text cache."""
from __future__ import annotations

from collections import defaultdict
import json
import os
from pathlib import Path
import statistics
from typing import Any

import numpy as np

from ..evaluation_io import sha256_file, write_json_atomic
from .reference import text_hash


def score_semantics(rows: list[dict[str, Any]], config: dict[str, Any], output: Path,
                    *, device: str, batch_size: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import torch
    from transformers import AutoModel, AutoTokenizer

    model_path = Path(config["local_path"])
    if not model_path.is_dir():
        raise FileNotFoundError(f"semantic model must be prepared explicitly: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    max_tokens = config["max_tokens"]
    pairs, unique = [], {}
    for row in rows:
        for section, fields in row["teacher_parsed"]["qualitative"].items():
            for key, reference in fields.items():
                predicted = row["parsed"]["qualitative"][section][key]
                assert reference
                record = {"request_id": row["request_id"], "view": row["view"], "section": section,
                          "field": key, "prediction_text": predicted, "reference_text": reference,
                          "status": "matched" if predicted else "missing_prediction", "cosine": None}
                pairs.append(record)
                for text in (reference, predicted):
                    if text:
                        unique[text_hash(text)] = text
    keys = sorted(unique)
    texts = [unique[key] for key in keys]
    # Length auditing is separate from the actual truncated embedding call.
    lengths = [len(tokenizer(text, truncation=False, verbose=False)["input_ids"]) for text in texts]
    files = {str(p.relative_to(model_path)): sha256_file(p) for p in sorted(model_path.iterdir()) if p.is_file() and p.suffix in {".json", ".txt", ".safetensors", ".bin"}}
    identity = {"model_id": config["model_id"], "revision": config["revision"], "model_files": files,
                "text_hashes": keys, "max_tokens": max_tokens, "pooling": config["pooling"],
                "dtype": "float32", "device_type": device.split(":")[0], "batch_size": batch_size}
    cache = output / "semantic_embeddings.npz"
    cache_identity = output / "semantic_cache_identity.json"
    if cache.is_file() and cache_identity.is_file() and json.loads(cache_identity.read_text()) == identity:
        with np.load(cache) as loaded:
            vectors = loaded["embeddings"]
        print(f"Semantic cache reused: {len(texts)} unique points", flush=True)
    else:
        model = AutoModel.from_pretrained(model_path, local_files_only=True).to(device).eval()
        chunks = []
        for start in range(0, len(texts), batch_size):
            tokens = tokenizer(texts[start:start+batch_size], padding=True, truncation=True,
                               max_length=max_tokens, return_tensors="pt").to(device)
            with torch.inference_mode():
                vector = model(**tokens).last_hidden_state[:, 0]
                vector = torch.nn.functional.normalize(vector.float(), dim=-1)
            chunks.append(vector.cpu().numpy())
            if start % (batch_size*20) == 0 or start+batch_size >= len(texts):
                print(f"Semantic encoded {min(start+batch_size,len(texts))}/{len(texts)}", flush=True)
        vectors = np.concatenate(chunks)
        temporary = cache.with_name(f".{cache.name}.{os.getpid()}.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, embeddings=vectors)
        os.replace(temporary, cache)
        write_json_atomic(cache_identity, identity)
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    if vectors.shape[0] != len(keys) or not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5):
        raise ValueError("invalid semantic embedding cache")
    lookup = dict(zip(keys, vectors))
    for pair in pairs:
        if pair["prediction_text"]:
            cosine = float(np.dot(lookup[text_hash(pair["prediction_text"])], lookup[text_hash(pair["reference_text"])]))
            pair["cosine"] = max(-1.0, min(1.0, cosine))
    def aggregate(items: list[dict[str, Any]]) -> dict[str, Any]:
        values = [p["cosine"] for p in items if p["cosine"] is not None]
        return {"expected_points": len(items), "matched_points": len(values), "point_coverage": len(values)/len(items),
                "mean_cosine_on_matched": statistics.fmean(values) if values else None,
                "coverage_adjusted_cosine": sum(max(0, v) for v in values)/len(items)}
    groups = defaultdict(list)
    for pair in pairs:
        groups[(pair["view"], pair["section"], pair["field"])].append(pair)
    summary = {**aggregate(pairs), "by_field": [{"view": key[0], "section": key[1], "field": key[2], **aggregate(values)} for key, values in groups.items()],
               "model_id": config["model_id"], "revision": config["revision"], "local_path": str(model_path),
               "model_files": files, "unique_texts": len(texts), "truncated_unique_texts": sum(n>max_tokens for n in lengths),
               "max_observed_tokens": max(lengths), "max_tokens": max_tokens, "dtype": "float32", "device": device,
               "batch_size": batch_size, "pooling": "CLS + L2 normalization; no query instruction",
               "teacher_self_similarity_max_error": float(np.max(np.abs(np.sum(vectors*vectors, axis=1)-1))),
               "interpretation": "qualitative reference agreement, not factual correctness", "human_evaluation": "excluded_by_user"}
    return summary, pairs
