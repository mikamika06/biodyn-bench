import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = pathlib.Path(__file__).resolve().parent.parent / "out" / "figs"
fig, ax = plt.subplots(figsize=(8.4, 4.0))
ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")

def box(x, y, w, h, text, fc, fs=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12", fc=fc, ec="#333333", lw=1.0))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

def arrow(x1, y1, x2, y2, text="", color="#333333"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, color=color, lw=1.2))
    if text:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.16, text, ha="center", fontsize=8, color=color)

box(0.3, 3.4, 2.6, 1.1, "закладений граф\n(іменований мотив)", "#dce9f5")
box(0.3, 0.6, 2.6, 1.1, "синтетичні клітини\n(SDE + шум)", "#dce9f5")
arrow(1.6, 3.4, 1.6, 1.7, "генерація")
box(3.9, 2.0, 2.2, 1.1, "masked-модель\n(навчена з нуля)", "#f5e9dc")
arrow(2.9, 1.15, 4.4, 2.0, "навчання")
box(7.2, 3.4, 2.5, 1.1, "істина даних:\nребро в графі?", "#dff0dd")
box(7.2, 0.6, 2.5, 1.1, "істина моделі:\nвтручання міняє\nпередбачення?", "#f5dddd")
arrow(2.9, 3.95, 7.2, 3.95, "пряме читання")
arrow(6.1, 2.3, 7.2, 1.4, "підміна входу")
arrow(6.1, 2.9, 7.2, 3.6, "увага / градієнт / проба", "#7a7a7a")
ax.text(8.45, 2.55, "розрив = моралізація:\nколайдер 1.000 проти 0.5", ha="center", fontsize=9, style="italic")
ax.add_patch(FancyArrowPatch((8.45, 3.3), (8.45, 1.85), arrowstyle="<|-|>", mutation_scale=12, color="#a03030", lw=1.4))
fig.tight_layout()
fig.savefig(OUT / "fig_two_truths_schema.png", dpi=200)
print("done")
