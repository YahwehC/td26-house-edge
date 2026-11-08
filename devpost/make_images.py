"""Devpost gallery images (3:2, 1500x1000 PNG). Numbers are validation scores from runs/*/validation_score.json
and runs/r2_*/r2g_report.json."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parent
plt.rcParams.update({"font.size": 15, "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE, GREY, GREEN = "#2f6db5", "#e07b28", "#9aa3ad", "#2e9a5a"


def save(fig, name):
    fig.savefig(OUT / name, dpi=100)
    plt.close(fig)


# 1. Score progression (main round, out of 60)
vers = ["v8", "v9", "v10", "v12", "v13"]
pts = [58.271, 59.052, 59.071, 59.117, 59.123]
fig, ax = plt.subplots(figsize=(15, 10))
ax.plot(vers, pts, "o-", color=BLUE, lw=3, ms=12)
for v, p in zip(vers, pts):
    ax.annotate(f"{p:.2f}", (v, p), textcoords="offset points", xytext=(0, 14), ha="center", fontsize=16)
ax.axhline(60, color=GREY, ls="--")
ax.text(0, 59.93, "perfect knowledge = 60", color=GREY, fontsize=14)
ax.set_ylim(58, 60.1)
ax.set_ylabel("Validation points (out of 60)")
ax.set_title("Main round: validation score by model version", fontsize=22, pad=18)
ax.text(1, 58.55, "+0.78: size regime shifts win odds\n+ shared hidden 'hot/cold' state", fontsize=14, color=ORANGE)
fig.tight_layout()
save(fig, "1_score_progression.png")

# 2. Error vs pooled reference per condition (lower is better)
conds = ["stable", "short_history", "drift", "shared_change"]
main = [0.0062, 0.0346, 0.0112, 0.0073]
r2v0 = [0.0267, 0.0394, 0.0310, 0.0275]
r2v1 = [0.0142, 0.0196, 0.0090, 0.0100]
fig, ax = plt.subplots(figsize=(15, 10))
w = .26
x = range(len(conds))
ax.bar([i - w for i in x], main, w, label="Main round (v13), 59.12/60", color=BLUE)
ax.bar(list(x), r2v0, w, label="Adaptation, refit base (r2_v0), 19.40/20", color=GREY)
ax.bar([i + w for i in x], r2v1, w, label="Adaptation + hidden state (r2_v1), 19.74/20", color=ORANGE)
ax.set_xticks(list(x), [c.replace("_", " ") for c in conds])
ax.set_ylabel("Squared error / pooled-reference error")
ax.set_title("Error relative to the pooled baseline (lower is better; baseline = 1.0)", fontsize=20, pad=18)
ax.legend(frameon=False, fontsize=15)
fig.tight_layout()
save(fig, "2_error_by_condition.png")

# 3. Pipeline diagram
fig, ax = plt.subplots(figsize=(15, 10))
ax.set_xlim(0, 15)
ax.set_ylim(0, 10)
ax.axis("off")
ax.set_title("HouseEdge-3Lsahur pipeline", fontsize=24, pad=10)
boxes = [
    (0.4, 6.6, "Payout logs\n8 machines per\nepisode", GREY),
    (4.0, 6.6, "Rule discovery\nlag-pair win cells,\nsize regime,\ncoupling lag", BLUE),
    (7.6, 6.6, "Bayesian cell model\nBeta-pooled rates,\nepisode-type mixture", BLUE),
    (11.2, 6.6, "Hidden shared state\n2-state HMM (EM),\npeer-win coupling", ORANGE),
    (11.2, 2.0, "Short histories\nintegrate missing\nwin / size categories", ORANGE),
    (7.6, 2.0, "Validation gate\ntrain-only fit, adopt\nonly if score improves", GREEN),
    (4.0, 2.0, "Forecast CSV\np(0x, 1x, 2x, 5x, 20x)\n+ reproducible ZIP", GREY),
]
for x0, y0, t, c in boxes:
    ax.add_patch(FancyBboxPatch((x0, y0), 3.3, 2.3, boxstyle="round,pad=0.1", fc=c, ec="none", alpha=.9))
    ax.text(x0 + 1.65, y0 + 1.15, t, ha="center", va="center", color="white", fontsize=14, weight="bold")
arrow = dict(arrowstyle="-|>", color="#333", lw=2.5, mutation_scale=25)
for a, b in [((3.8, 7.75), (4.0, 7.75)), ((7.4, 7.75), (7.6, 7.75)), ((11.0, 7.75), (11.2, 7.75)),
             ((12.85, 6.5), (12.85, 4.4)), ((11.1, 3.15), (11.0, 3.15)), ((7.5, 3.15), (7.4, 3.15))]:
    ax.annotate("", b, a, arrowprops=arrow)
ax.text(7.5, 0.7, "Same pipeline refits automatically on new data (adaptation round: 90 minutes)  |  "
        "stdlib-only inference", ha="center", fontsize=14, color="#444")
save(fig, "3_pipeline.png")
print("ok")
