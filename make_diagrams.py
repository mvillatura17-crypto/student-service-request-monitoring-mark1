"""Create assets/use_case_diagram.png and assets/decision_workflow.png."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

ASSETS = Path(__file__).resolve().parent.parent / "assets"


def use_case_diagram():
    fig, ax = plt.subplots(figsize=(12, 8.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 9.2)
    ax.axis("off")

    ax.add_patch(Rectangle((3.0, 0.5), 6.0, 8.3, fill=False, linewidth=2))
    ax.text(6, 8.5, "Student Service Request Monitoring System",
            ha="center", va="center", fontsize=12, weight="bold")

    cases = [
        (7.5, "Upload and Validate Dataset"),
        (6.3, "Explore Data Analysis"),
        (5.1, "Predict SLA Breach"),
        (3.9, "View Model Metrics and Limitations"),
        (2.7, "View Dashboard and Review Queue"),
        (1.5, "Record Final Review Decision"),
    ]
    ys = {}
    for y, label in cases:
        ax.add_patch(Ellipse((6, y), 4.6, 0.9, facecolor="white", edgecolor="black", linewidth=1.4))
        ax.text(6, y, label, ha="center", va="center", fontsize=9.5)
        ys[label] = y

    def actor(x, y, label):
        ax.add_patch(Ellipse((x, y + 0.65), 0.38, 0.38, fill=False, linewidth=1.5))
        ax.plot([x, x], [y + 0.46, y - 0.25], color="black")
        ax.plot([x - 0.38, x + 0.38], [y + 0.2, y + 0.2], color="black")
        ax.plot([x, x - 0.3], [y - 0.25, y - 0.7], color="black")
        ax.plot([x, x + 0.3], [y - 0.25, y - 0.7], color="black")
        ax.text(x, y - 0.95, label, ha="center", va="top", fontsize=10, weight="bold")

    actor(1.2, 5.0, "Administrator")
    actor(10.8, 5.0, "Service Staff")

    admin = ["Upload and Validate Dataset", "Explore Data Analysis",
             "View Model Metrics and Limitations", "View Dashboard and Review Queue"]
    staff = ["Explore Data Analysis", "Predict SLA Breach", "View Model Metrics and Limitations",
             "View Dashboard and Review Queue", "Record Final Review Decision"]
    for name in admin:
        ax.plot([1.6, 3.7], [5.2, ys[name]], color="gray", linewidth=1)
    for name in staff:
        ax.plot([10.4, 8.3], [5.2, ys[name]], color="gray", linewidth=1)

    ax.text(1.2, 3.3, "Views results,\nuploads data", ha="center", fontsize=8.5, style="italic")
    ax.text(10.8, 3.3, "Views results and\nrecords final decisions", ha="center",
            fontsize=8.5, style="italic")
    ax.set_title("Updated UML Use Case Diagram", pad=12)
    fig.savefig(ASSETS / "use_case_diagram.png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def workflow_diagram():
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis("off")

    def box(x, y, w, h, text, color="#e8f1fb"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.15",
                                    facecolor=color, edgecolor="black", linewidth=1.3))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.5)

    def arrow(p1, p2, label=None, off=(0, 0.15)):
        ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=14, linewidth=1.3))
        if label:
            ax.text((p1[0] + p2[0]) / 2 + off[0], (p1[1] + p2[1]) / 2 + off[1], label,
                    ha="center", fontsize=9, weight="bold")

    w, h = 2.4, 1.1
    row1 = 5.0
    xs = [0.4, 3.6, 6.8, 10.0]
    texts = [
        "1. Upload CSV\n(Administrator)",
        "2. Validate schema,\nclean data, check\nnulls and duplicates",
        "3. Build submission-time\nfeatures\n(no leakage)",
        "4. Saved Decision Tree\npipeline predicts\nbreach probability",
    ]
    for x, t in zip(xs, texts):
        box(x, row1, w, h, t)
    for i in range(3):
        arrow((xs[i] + w, row1 + h / 2), (xs[i + 1], row1 + h / 2))

    # Down to row 2
    row2 = 2.6
    box(10.0, row2, w, h, "5. Assign risk level\nHigh / Medium / Low", "#fff4d6")
    arrow((11.2, row1), (11.2, row2 + h))
    box(6.8, row2, w, h, "6. Show review queue\nand KPIs on dashboard", "#fff4d6")
    arrow((10.0, row2 + h / 2), (6.8 + w, row2 + h / 2))

    # Decision diamond
    cx, cy = 3.9, row2 + h / 2
    ax.add_patch(Polygon([(cx, cy + 1.0), (cx + 1.6, cy), (cx, cy - 1.0), (cx - 1.6, cy)],
                         closed=True, facecolor="#fde2e2", edgecolor="black", linewidth=1.3))
    ax.text(cx, cy, "7. Staff reviews\nrequest. Agree\nwith model?", ha="center",
            va="center", fontsize=9)
    arrow((6.8, row2 + h / 2), (cx + 1.6, cy))

    box(0.4, 0.5, 3.0, 1.0, "8a. Yes: apply recommended\nfollow-up and record\nfinal decision", "#e4f5e4")
    box(4.4, 0.5, 3.0, 1.0, "8b. No: override with notes\nand record final\ndecision", "#e4f5e4")
    arrow((cx - 0.6, cy - 0.6), (1.9, 1.5), "Yes", off=(-0.4, 0.1))
    arrow((cx + 0.2, cy - 1.0), (5.9, 1.5), "No", off=(0.4, 0.0))
    box(9.2, 0.5, 4.2, 1.0, "9. Monitor request until completed;\nresult feeds the next data upload", "#e4f5e4")
    arrow((3.4, 1.0), (4.4, 1.0))  # visual link between 8a and 8b is not needed -> remove below
    ax.patches[-1].set_visible(False)
    arrow((7.4, 1.0), (9.2, 1.0))

    ax.text(7, 6.7, "Analytical Process Flow and Human Decision Workflow",
            ha="center", fontsize=13, weight="bold")
    ax.text(7, 0.05, "The model only recommends. A staff member makes and records the final decision.",
            ha="center", fontsize=9.5, style="italic")
    fig.savefig(ASSETS / "decision_workflow.png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    use_case_diagram()
    workflow_diagram()
    print("Diagrams saved in", ASSETS)
