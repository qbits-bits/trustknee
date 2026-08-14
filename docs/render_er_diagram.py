"""Renders the Phase 2 Section 5.4 ER diagram for TrustKnee (docs/er_diagram.png).

Entities and relationships mirror db/schema.sql. Run with:
    .venv/bin/python docs/render_er_diagram.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

REF_HEADER = "#7A6A4F"     # lookup/reference tables: Subjects, Sensors, Labels
REF_BODY = "#F4EFE6"
CORE_HEADER = "#2C3E50"    # pipeline/working tables
CORE_BODY = "#EAF1F8"
BODY_EDGE = "#7F8C9B"
HEADER_TEXT = "white"
FONT = "DejaVu Sans"
HEADER_H = 0.42
LINE_H = 0.255

# Layout: 3 columns x 3 rows. Trials/WindowFeatures/Predictions form the
# central pipeline spine (top to bottom); reference tables sit either side
# of the row they feed; Feedback sits one column over so its line from
# Trials runs through open space instead of through Sensors/WindowFeatures.
entities = {
    "Subjects": {
        "pos": (0.4, 8.5), "w": 3.6, "kind": "ref",
        "pk": "subject_id (PK)",
        "attrs": ["gender", "height_cm", "weight_kg", "age_years",
                  "injured_leg", "pathology"],
    },
    "Trials": {
        "pos": (5.0, 8.0), "w": 4.2, "kind": "core",
        "pk": "trial_id (PK)",
        "attrs": ["subject_id (FK)", "label_id (FK)", "trial_num",
                  "imu_path", "emg_path", "duration_s",
                  "n_imu_samples", "n_emg_samples"],
    },
    "Labels": {
        "pos": (10.2, 9.2), "w": 4.0, "kind": "ref",
        "pk": "label_id (PK)",
        "attrs": ["execution (Correct/Wrong)", "exercise", "description"],
    },
    "Sensors": {
        "pos": (0.4, 4.8), "w": 3.6, "kind": "ref",
        "pk": "sensor_id (PK)",
        "attrs": ["muscle_placement", "emg_channels = 1", "imu_channels = 6"],
    },
    "WindowFeatures": {
        "pos": (5.0, 4.1), "w": 4.2, "kind": "core",
        "pk": "window_id (PK)",
        "attrs": ["trial_id (FK)", "window_index", "start_time_s",
                  "end_time_s", "feature_* (many float cols,",
                  "per sensor/channel/stat)"],
    },
    "Feedback": {
        "pos": (10.2, 4.8), "w": 4.0, "kind": "core",
        "pk": "feedback_id (PK)",
        "attrs": ["trial_id (FK)", "source_prediction_id (FK)",
                  "feedback_text"],
    },
    "Predictions": {
        "pos": (5.0, 0.9), "w": 4.2, "kind": "core",
        "pk": "prediction_id (PK)",
        "attrs": ["window_id (FK)", "model_name",
                  "predicted_label_id (FK)", "confidence_score",
                  "is_flagged_uncertain"],
    },
    "ShapAttributions": {
        "pos": (10.2, 1.35), "w": 4.0, "kind": "core",
        "pk": "attribution_id (PK)",
        "attrs": ["prediction_id (FK)", "feature_name", "attribution_value"],
    },
}

# height derived from attribute count so text never clips the box
for e in entities.values():
    e["h"] = HEADER_H + 0.75 + len(e["attrs"]) * LINE_H

relations = [
    ("Subjects", "Trials", "1", "*"),
    ("Labels", "Trials", "1", "*"),
    ("Trials", "WindowFeatures", "1", "*"),
    ("Trials", "Feedback", "1", "*"),
    ("WindowFeatures", "Predictions", "1", "*"),
    ("Predictions", "ShapAttributions", "1", "*"),
]
ref_relations = [("Sensors", "WindowFeatures")]


def center(e):
    x, y = e["pos"]
    return x + e["w"] / 2, y + e["h"] / 2


def box_edge_point(e, target_center):
    x, y = e["pos"]
    w, h = e["w"], e["h"]
    cx, cy = center(e)
    tx, ty = target_center
    dx, dy = tx - cx, ty - cy
    if dx == 0 and dy == 0:
        return cx, cy
    if abs(dx) * h > abs(dy) * w:
        ex = x + w if dx > 0 else x
        ey = cy + dy * (ex - cx) / dx
    else:
        ey = y + h if dy > 0 else y
        ex = cx + dx * (ey - cy) / dy
    return ex, ey


def draw_entity(ax, name, e):
    x, y = e["pos"]
    w, h = e["w"], e["h"]
    header_color = REF_HEADER if e["kind"] == "ref" else CORE_HEADER
    body_color = REF_BODY if e["kind"] == "ref" else CORE_BODY

    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.06",
                                 linewidth=1.1, edgecolor=BODY_EDGE, facecolor=body_color, zorder=2))
    ax.add_patch(FancyBboxPatch((x, y + h - HEADER_H), w, HEADER_H,
                                 boxstyle="round,pad=0,rounding_size=0.06",
                                 linewidth=1.1, edgecolor=BODY_EDGE, facecolor=header_color, zorder=3))
    ax.text(x + w / 2, y + h - HEADER_H / 2, name, ha="center", va="center",
            fontsize=12.5, fontweight="bold", color=HEADER_TEXT, family=FONT, zorder=4)

    ax.text(x + 0.15, y + h - HEADER_H - 0.22, e["pk"], ha="left", va="top",
            fontsize=9.5, fontweight="bold", color="#1A2A3A", family=FONT, zorder=4)
    ax.plot([x + 0.1, x + w - 0.1], [y + h - HEADER_H - 0.38] * 2,
            color=BODY_EDGE, linewidth=0.6, zorder=4)

    ay = y + h - HEADER_H - 0.58
    for attr in e["attrs"]:
        ax.text(x + 0.15, ay, attr, ha="left", va="top", fontsize=8.6,
                color="#33414F", family=FONT, zorder=4)
        ay -= LINE_H


def draw_relation(ax, a, b, card_a=None, card_b=None, dashed=False):
    ea, eb = entities[a], entities[b]
    ca, cb = center(ea), center(eb)
    pa = box_edge_point(ea, cb)
    pb = box_edge_point(eb, ca)
    style = dict(color="#8A7A5A" if dashed else "#5D4037",
                 linewidth=1.1 if dashed else 1.4,
                 linestyle=(0, (5, 4)) if dashed else "solid", zorder=1)
    ax.add_line(Line2D([pa[0], pb[0]], [pa[1], pb[1]], **style))
    if card_a and card_b:
        for frac, card in ((0.16, card_a), (0.84, card_b)):
            mx, my = pa[0] + (pb[0] - pa[0]) * frac, pa[1] + (pb[1] - pa[1]) * frac
            ax.text(mx, my, card, fontsize=9.5, fontweight="bold", color="#5D4037", family=FONT,
                    ha="center", va="center", bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))


fig, ax = plt.subplots(figsize=(15.5, 12.5))
ax.set_xlim(0, 14.6)
ax.set_ylim(0, 12.6)
ax.axis("off")

for name, e in entities.items():
    draw_entity(ax, name, e)
for a, b, ca, cb in relations:
    draw_relation(ax, a, b, ca, cb)
for a, b in ref_relations:
    draw_relation(ax, a, b, dashed=True)

ax.text(0.4, 12.35, "TrustKnee: Phase 2 Database Schema (Section 5.4)",
        fontsize=16, fontweight="bold", family=FONT, color="#1A2A3A")
ax.text(0.4, 11.95,
        "Gold header = lookup/reference table (Subjects, Sensors, Labels)   |   "
        "Navy header = pipeline table   |   Dashed line = fixed hardware reference, not an FK",
        fontsize=10, family=FONT, color="#555")

fig.tight_layout()
out_path = Path(__file__).parent / "er_diagram.png"
fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
print(f"saved {out_path}")
