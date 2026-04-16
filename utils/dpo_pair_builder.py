#!/usr/bin/env python3
"""Build offline DPO preference pairs from a scored VITA-Audio JSON/JSONL dataset.

This utility converts samples that contain multiple scored model generations into a
single (chosen, rejected) pair per prompt, optionally filtering by score gap.

Expected input schema (per sample, minimal):
  - "messages": list[{"role": str, "content": str}]
  - "audios": list[str]                       # prompt audio basenames/paths
  - "output": dict[str, {"output-text": str,
                         "output-audio": str,
                         "gemini_score": {"semantic_score": int,
                                          "paralinguistic_score": int}}]

Output schema (per line, JSONL):
  - "messages", "audios"
  - "sft_target_text", "sft_target_audio"     # chosen (mapped to SFT target)
  - "rejected_text", "rejected_audio"
  - metadata: chosen/rejected scores, keys, score_gap, original id fields
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
import heapq
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple


ScoreTuple = Tuple[float, str, Dict[str, Any]]


@dataclass(frozen=True)
class ScoreConfig:
    mode: str
    semantic_weight: float = 1.0
    paralinguistic_weight: float = 1.0

    def compute(self, entry: Dict[str, Any]) -> Optional[float]:
        score = entry.get("gemini_score")
        if not isinstance(score, dict):
            return None
        semantic = score.get("semantic_score")
        para = score.get("paralinguistic_score")
        if not isinstance(semantic, (int, float)) or not isinstance(para, (int, float)):
            return None

        if self.mode == "semantic":
            return float(semantic)
        if self.mode == "paralinguistic":
            return float(para)
        if self.mode == "weighted":
            return float(semantic) * float(self.semantic_weight) + float(para) * float(
                self.paralinguistic_weight
            )
        # default: sum
        return float(semantic) + float(para)


def _iter_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _iter_json(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fp:
        data = json.load(fp)
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                yield item
        return
    raise ValueError(f"Unsupported JSON root type: {type(data).__name__} (expected list)")


def iter_samples(path: str) -> Iterator[Dict[str, Any]]:
    if path.endswith(".jsonl"):
        yield from _iter_jsonl(path)
        return
    yield from _iter_json(path)


def resolve_audio_paths(audio_paths: List[str], root: Optional[str]) -> List[str]:
    if not root:
        return [p for p in audio_paths if p]
    resolved: List[str] = []
    for path in audio_paths:
        if not path:
            continue
        cleaned = str(path).strip()
        if not cleaned:
            continue
        if os.path.isabs(cleaned):
            resolved.append(os.path.normpath(cleaned))
        else:
            resolved.append(os.path.normpath(os.path.join(root, cleaned)))
    return resolved


def resolve_audio_path(path: Optional[str], root: Optional[str]) -> Optional[str]:
    if not path:
        return None
    cleaned = str(path).strip()
    if not cleaned:
        return None
    if os.path.isabs(cleaned) or not root:
        return os.path.normpath(cleaned)
    return os.path.normpath(os.path.join(root, cleaned))


def select_best_worst(outputs: Dict[str, Any], scorer: ScoreConfig) -> Optional[Tuple[ScoreTuple, ScoreTuple]]:
    scored: List[ScoreTuple] = []
    for key, entry in outputs.items():
        if not isinstance(entry, dict):
            continue
        s = scorer.compute(entry)
        if s is None:
            continue
        scored.append((s, key, entry))

    if len(scored) < 2:
        return None

    best = max(scored, key=lambda x: x[0])
    worst = min(scored, key=lambda x: x[0])
    return best, worst


def build_pair_record(
    sample: Dict[str, Any],
    *,
    prompt_audio_root: Optional[str],
    gen_audio_root: Optional[str],
    sft_audio_root: Optional[str],
    chosen_source: str,
    rejected_source: str,
    scorer: ScoreConfig,
) -> Optional[Dict[str, Any]]:
    messages = sample.get("messages")
    if not isinstance(messages, list) or not messages:
        return None

    prompt_audios = sample.get("audios") or []
    if not isinstance(prompt_audios, list):
        prompt_audios = []
    resolved_prompt_audios = resolve_audio_paths([str(x) for x in prompt_audios if x], prompt_audio_root)

    outputs = sample.get("output") or {}
    if not isinstance(outputs, dict) or not outputs:
        return None

    selected = select_best_worst(outputs, scorer)
    if selected is None:
        return None

    (best_score, best_key, best_entry), (worst_score, worst_key, worst_entry) = selected
    output_score_gap = float(best_score) - float(worst_score)

    if rejected_source != "worst_output":
        raise ValueError("Only rejected_source=worst_output is supported currently.")

    if chosen_source == "best_output":
        chosen_text = best_entry.get("output-text") or ""
        chosen_audio = resolve_audio_path(best_entry.get("output-audio"), gen_audio_root)
    elif chosen_source == "sft_target":
        chosen_text = sample.get("sft_target_text") or ""
        chosen_audio = resolve_audio_path(sample.get("sft_target_audio"), sft_audio_root or prompt_audio_root)
    else:
        raise ValueError("chosen_source must be one of: best_output|sft_target")

    rejected_text = worst_entry.get("output-text") or ""
    rejected_audio = resolve_audio_path(worst_entry.get("output-audio"), gen_audio_root)

    record: Dict[str, Any] = {
        "id": sample.get("id"),
        "messages": messages,
        "audios": resolved_prompt_audios,
        "sft_target_text": chosen_text,
        "sft_target_audio": chosen_audio,
        "rejected_text": rejected_text,
        "rejected_audio": rejected_audio,
        "chosen_score": best_score if chosen_source == "best_output" else None,
        "rejected_score": worst_score,
        "score_gap": output_score_gap,
        "chosen_key": best_key if chosen_source == "best_output" else "sft_target",
        "rejected_key": worst_key,
        "best_output_score": best_score,
        "worst_output_score": worst_score,
        "best_output_key": best_key,
        "worst_output_key": worst_key,
        "output_score_gap": output_score_gap,
        "task_type": sample.get("task_type"),
        "question_text": sample.get("question_text"),
        "question_text_raw": sample.get("question_text_raw"),
        "history_text": sample.get("history_text"),
        "source_dataset": (sample.get("original_sample") or {}).get("source_dataset"),
    }

    # Keep a trace of the original supervised target (often human/curated) for auditing.
    if chosen_source != "sft_target":
        record["orig_sft_target_text"] = sample.get("sft_target_text")
        record["orig_sft_target_audio"] = resolve_audio_path(
            sample.get("sft_target_audio"), sft_audio_root or prompt_audio_root
        )

    return record


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_path", required=True, help="Input JSON/JSONL path.")
    parser.add_argument("--output_path", required=True, help="Output JSONL path.")
    parser.add_argument(
        "--prompt_audio_root",
        default=None,
        help="Prefix for sample['audios'] (question/prompt audio).",
    )
    parser.add_argument(
        "--gen_audio_root",
        default=None,
        help="Prefix for model generation audios (output[*]['output-audio']).",
    )
    parser.add_argument(
        "--sft_audio_root",
        default=None,
        help="Prefix for sample['sft_target_audio'] when chosen_source=sft_target (defaults to prompt_audio_root).",
    )
    parser.add_argument(
        "--chosen_source",
        choices=("best_output", "sft_target"),
        default="best_output",
        help="Use best-scored model output as chosen, or keep existing sft_target_*.",
    )
    parser.add_argument(
        "--rejected_source",
        choices=("worst_output",),
        default="worst_output",
        help="How to select rejected completion.",
    )
    parser.add_argument(
        "--score_mode",
        choices=("sum", "semantic", "paralinguistic", "weighted"),
        default="sum",
        help="How to combine gemini_score fields into a scalar.",
    )
    parser.add_argument("--semantic_weight", type=float, default=1.0)
    parser.add_argument("--paralinguistic_weight", type=float, default=1.0)
    parser.add_argument("--min_score_gap", type=float, default=1.0, help="Filter out pairs with gap < min_score_gap.")
    parser.add_argument(
        "--validate_audio",
        action="store_true",
        help="Skip samples whose referenced audio files do not exist on disk.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Keep only top-K samples by score gap (requires reading full dataset).",
    )
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    if not args.output_path.endswith(".jsonl"):
        raise ValueError("--output_path must end with .jsonl for streaming writes.")

    random.seed(args.seed)

    scorer = ScoreConfig(
        mode=args.score_mode,
        semantic_weight=args.semantic_weight,
        paralinguistic_weight=args.paralinguistic_weight,
    )

    kept = 0
    skipped = 0
    scanned = 0
    max_gap: Optional[float] = None
    max_gap_id: Any = None
    missing_prompt_audio = 0
    missing_chosen_audio = 0
    missing_rejected_audio = 0

    top_k = args.top_k
    heap: List[Tuple[float, int, Dict[str, Any]]] = []
    seq = 0

    def _maybe_keep(rec: Dict[str, Any]) -> None:
        nonlocal kept, max_gap, max_gap_id, seq
        gap_val = float(rec.get("score_gap") or 0.0)
        if max_gap is None or gap_val > max_gap:
            max_gap = gap_val
            max_gap_id = rec.get("id")
        if top_k is None:
            out_fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1
            return
        if top_k <= 0:
            return
        seq += 1
        if len(heap) < top_k:
            heapq.heappush(heap, (gap_val, seq, rec))
            return
        if gap_val > heap[0][0]:
            heapq.heapreplace(heap, (gap_val, seq, rec))

    with open(args.output_path, "w", encoding="utf-8") as out_fp:
        for sample in iter_samples(args.input_path):
            scanned += 1
            if args.max_samples is not None and scanned > args.max_samples:
                break

            record = build_pair_record(
                sample,
                prompt_audio_root=args.prompt_audio_root,
                gen_audio_root=args.gen_audio_root,
                sft_audio_root=args.sft_audio_root,
                chosen_source=args.chosen_source,
                rejected_source=args.rejected_source,
                scorer=scorer,
            )
            if record is None:
                skipped += 1
                continue

            gap = float(record.get("score_gap") or 0.0)
            if gap < float(args.min_score_gap):
                skipped += 1
                continue

            if args.validate_audio:
                missing = False
                for audio_path in record.get("audios") or []:
                    if audio_path and not os.path.exists(audio_path):
                        missing_prompt_audio += 1
                        missing = True
                        break
                chosen_audio = record.get("sft_target_audio")
                if chosen_audio and not os.path.exists(str(chosen_audio)):
                    missing_chosen_audio += 1
                    missing = True
                rejected_audio = record.get("rejected_audio")
                if rejected_audio and not os.path.exists(str(rejected_audio)):
                    missing_rejected_audio += 1
                    missing = True
                if missing:
                    skipped += 1
                    continue

            _maybe_keep(record)

        if top_k is not None:
            out_fp.seek(0)
            out_fp.truncate(0)
            heap.sort(key=lambda x: x[0], reverse=True)
            for _, __, rec in heap:
                out_fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept = len(heap)

    print(
        json.dumps(
            {
                "input_path": args.input_path,
                "output_path": args.output_path,
                "scanned": scanned,
                "kept": kept,
                "skipped": skipped,
                "chosen_source": args.chosen_source,
                "rejected_source": args.rejected_source,
                "score_mode": args.score_mode,
                "min_score_gap": args.min_score_gap,
                "validate_audio": args.validate_audio,
                "missing_prompt_audio": missing_prompt_audio,
                "missing_chosen_audio": missing_chosen_audio,
                "missing_rejected_audio": missing_rejected_audio,
                "top_k": top_k,
                "max_gap": max_gap,
                "max_gap_id": max_gap_id,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
