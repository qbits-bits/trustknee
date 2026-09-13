"""Run a recorded TrustKnee trial through the real calibrated prototype."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from src import config
from src.models.artifacts import load_transformer_artifact
from src.persistence.db import DatabaseManager
from src.streaming.replay import replay_trial_pipeline

DISCLAIMER = (
    "Research prototype only. This result is not a diagnosis or confirmation that an exercise "
    "is clinically safe. Follow the plan provided by a qualified clinician."
)


def _cautious_feedback(execution: str, uncertain: bool) -> str:
    if uncertain:
        return (
            "The model is not confident enough to classify this movement. Check sensor placement "
            "and repeat only if that agrees with the clinician-provided exercise plan."
        )
    if execution == "Correct":
        return (
            "The model classified this recorded movement as correct. Treat this as a software "
            "result, not clinical confirmation; continue to follow clinician guidance."
        )
    return (
        "The model detected a movement pattern associated with an incorrect execution label. "
        "Do not change the exercise independently; review the movement with a clinician."
    )


def _render_html(result: dict[str, object]) -> str:
    sensors = result.get("top_contributing_sensors") or []
    sensor_items = "".join(f"<li>{html.escape(str(sensor))}</li>" for sensor in sensors)
    if not sensor_items:
        sensor_items = "<li>No explanation requested or no non-zero attribution.</li>"
    status_class = "uncertain" if result["is_uncertain"] else "decision"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>TrustKnee prototype result</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:860px;margin:40px auto;padding:0 20px;color:#263238}}
h1{{color:#1f4e78}} .card{{border:1px solid #ccd8df;border-radius:12px;padding:20px;margin:16px 0}}
.decision{{border-left:7px solid #159a9c}} .uncertain{{border-left:7px solid #e67e22}}
.metric{{font-size:1.4rem;font-weight:700}} .notice{{background:#fff4e5;padding:16px;border-radius:8px}}
table{{border-collapse:collapse;width:100%}} td{{padding:8px;border-bottom:1px solid #e3e8eb}}
</style></head><body>
<h1>TrustKnee recorded-trial demonstration</h1>
<div class="card {status_class}"><div class="metric">{html.escape(str(result["display_result"]))}</div>
<p>Confidence: {float(result["confidence"]):.1%} · Windows: {int(result["n_windows"])}</p></div>
<div class="card"><h2>Trial</h2><table>
<tr><td>Subject</td><td>{int(result["subject_id"])}</td></tr>
<tr><td>Recorded label</td><td>{html.escape(str(result["recorded_label"]))}</td></tr>
<tr><td>Model</td><td>{html.escape(str(result["model_name"]))}</td></tr>
<tr><td>Persisted trial</td><td>{int(result["persisted_trial_id"])}</td></tr></table></div>
<div class="card"><h2>Top contributing sensors</h2><ul>{sensor_items}</ul></div>
<div class="card"><h2>Cautious feedback</h2><p>{html.escape(str(result["feedback"]))}</p></div>
<p class="notice">{html.escape(str(result["disclaimer"]))}</p>
</body></html>"""


def run_demo(
    *,
    artifact_path: Path,
    imu_path: Path,
    emg_path: Path,
    output_dir: Path,
    subject_id: int,
    label_id: int,
    trial_num: int,
    device: str = "cpu",
    explain: bool = True,
    explanation_steps: int = 8,
) -> dict[str, object]:
    """Replay, infer, explain, persist, and render one recorded trial."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    artifact = load_transformer_artifact(artifact_path, device=device)
    engine = artifact.build_inference_engine(
        device=device,
        explain=explain,
        explanation_steps=explanation_steps,
    )
    database_path = destination / "trustknee_demo.sqlite"
    database = DatabaseManager(database_path)
    try:
        replay = replay_trial_pipeline(
            imu_path=imu_path,
            emg_path=emg_path,
            subject_id=subject_id,
            label_id=label_id,
            trial_num=trial_num,
            inference_engine=engine,
            database_manager=database,
            pacing_realtime=False,
            generate_feedback=False,
        )
        prediction = replay.trial_prediction
        label_info = config.LABELS.get(label_id)
        result: dict[str, object] = {
            "subject_id": subject_id,
            "label_id": label_id,
            "trial_num": trial_num,
            "recorded_label": label_info.description if label_info else str(label_id),
            "recorded_execution": label_info.execution if label_info else "Unknown",
            "predicted_label_id": prediction.predicted_label_id,
            "display_result": "Uncertain"
            if prediction.is_flagged_uncertain
            else prediction.execution,
            "confidence": prediction.confidence_score,
            "is_uncertain": prediction.is_flagged_uncertain,
            "n_windows": replay.n_windows,
            "model_name": prediction.model_name,
            "top_contributing_sensors": prediction.top_contributing_sensors,
            "feedback": _cautious_feedback(
                prediction.execution,
                prediction.is_flagged_uncertain,
            ),
            "persisted_trial_id": replay.trial_id,
            "database_path": str(database_path),
            "disclaimer": DISCLAIMER,
        }
    finally:
        database.close()

    json_path = destination / "demo_result.json"
    html_path = destination / "demo_result.html"
    temporary_json = json_path.with_suffix(".json.tmp")
    temporary_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary_json.replace(json_path)
    temporary_html = html_path.with_suffix(".html.tmp")
    temporary_html.write_text(_render_html(result), encoding="utf-8")
    temporary_html.replace(html_path)
    result["json_path"] = str(json_path)
    result["html_path"] = str(html_path)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--imu", type=Path, required=True)
    parser.add_argument("--emg", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subject-id", type=int, required=True)
    parser.add_argument("--label-id", type=int, choices=range(9), required=True)
    parser.add_argument("--trial-num", type=int, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--no-explain", action="store_true")
    parser.add_argument("--explanation-steps", type=int, default=8)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_demo(
        artifact_path=args.artifact,
        imu_path=args.imu,
        emg_path=args.emg,
        output_dir=args.output_dir,
        subject_id=args.subject_id,
        label_id=args.label_id,
        trial_num=args.trial_num,
        device=args.device,
        explain=not args.no_explain,
        explanation_steps=args.explanation_steps,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
