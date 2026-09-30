"""Cross-path comparison.

Given a job report containing several paths, produce the table + the "winners"
that the frontend renders.  Every number here is a reference-free proxy (there
is no ground truth for a user upload), so the comparison is presented as
*evidence*, not as a verdict -- the UI keeps the audio players next to it
because your ears are the real metric.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# (key, label, where to find it, higher_is_better, unit)
METRIC_SPECS = [
    ("snr_improvement_db", "SNR gain", "denoise", True, "dB"),
    ("noise_reduction_db", "Noise floor drop", "denoise", True, "dB"),
    ("speech_preserved", "Speech preserved", "denoise", True, ""),
    ("quality_score", "Denoise score", "denoise", True, "/100"),
    ("separation_score", "Separation score", "separate", True, "/100"),
    ("mean_cross_correlation", "Cross-talk", "separate", False, ""),
    ("energy_conservation", "Energy conservation", "separate", True, ""),
    ("n_speakers", "Speakers found", "path", True, ""),
    ("total", "Total time", "timing", False, "s"),
    ("realtime_factor", "x real time", "timing", False, "x"),
]


def _value(record: Dict[str, Any], key: str, where: str) -> Optional[float]:
    try:
        if where == "denoise":
            return float(record.get("denoised", {}).get("metrics", {}).get(key))
        if where == "separate":
            return float(record.get("separation", {}).get("metrics", {}).get(key))
        if where == "timing":
            return float(record.get("timings", {}).get(key))
        return float(record.get(key))
    except (TypeError, ValueError):
        return None


def build_comparison(report: Dict[str, Any]) -> Dict[str, Any]:
    paths: List[Dict[str, Any]] = [p for p in report.get("paths", []) if p.get("status") == "ok"]
    if not paths:
        return {"rows": [], "winners": {}, "ranking": [], "note": "no path completed successfully"}

    rows = []
    for key, label, where, higher, unit in METRIC_SPECS:
        values = {p["id"]: _value(p, key, where) for p in paths}
        present = {k: v for k, v in values.items() if v is not None}
        if not present:
            continue
        best_id = (max if higher else min)(present, key=lambda k: present[k])
        rows.append(
            {
                "key": key,
                "label": label,
                "unit": unit,
                "higher_is_better": higher,
                "values": values,
                "best": best_id,
            }
        )

    winners = {
        "cleanest": _pick(paths, lambda p: _value(p, "quality_score", "denoise")),
        "best_separation": _pick(paths, lambda p: _value(p, "separation_score", "separate")),
        "fastest": _pick(paths, lambda p: -(_value(p, "total", "timing") or 1e9)),
        "most_speakers": _pick(paths, lambda p: _value(p, "n_speakers", "path")),
    }

    ranking = []
    for p in paths:
        quality = _value(p, "quality_score", "denoise") or 0.0
        separation = _value(p, "separation_score", "separate") or 0.0
        seconds = _value(p, "total", "timing") or 0.0
        # 45% denoise, 45% separation, 10% speed (normalised against the slowest)
        slowest = max((_value(q, "total", "timing") or 0.0) for q in paths) or 1.0
        speed = 100.0 * (1.0 - seconds / slowest)
        ranking.append(
            {
                "id": p["id"],
                "name": p.get("name", p["id"]),
                "overall": round(0.45 * quality + 0.45 * separation + 0.10 * speed, 1),
                "denoise_score": round(quality, 1),
                "separation_score": round(separation, 1),
                "seconds": round(seconds, 2),
                "n_speakers": p.get("n_speakers", 0),
            }
        )
    ranking.sort(key=lambda r: -r["overall"])

    speaker_counts = {p["id"]: p.get("n_speakers", 0) for p in paths}
    agreement = len(set(speaker_counts.values())) == 1

    return {
        "rows": rows,
        "winners": winners,
        "ranking": ranking,
        "speaker_counts": speaker_counts,
        "speaker_count_agreement": agreement,
        "note": (
            "All scores are reference-free estimates (no clean ground truth exists for "
            "an arbitrary upload). Use them to shortlist, then listen."
        ),
    }


def _pick(paths: List[Dict[str, Any]], score) -> Optional[str]:
    scored = [(score(p), p["id"]) for p in paths]
    scored = [(s, i) for s, i in scored if s is not None]
    if not scored:
        return None
    return max(scored)[1]
