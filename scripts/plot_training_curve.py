"""
scripts/plot_training_curve.py

Parses a PaddleOCR tools/train.py log and produces a two-panel training
curve figure for the paper:
  - Left: training loss (raw per-batch, light + per-epoch mean, bold)
  - Right: best-accuracy-so-far and normalized edit distance at each
    periodic evaluation checkpoint during training
"""

import argparse
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STEP_LINE = re.compile(
    r"epoch:\s*\[(\d+)/(\d+)\],\s*global_step:\s*(\d+),.*?"
    r"loss:\s*([\d.]+)"
)
EVAL_ACC_LINE = re.compile(r"ppocr INFO:\s*acc:([\d.]+)\s*$")
EVAL_NED_LINE = re.compile(r"ppocr INFO:\s*norm_edit_dis:([\d.]+)\s*$")
EVAL_MARKER = re.compile(r"metric eval \*+")
BEST_METRIC_LINE = re.compile(
    r"best metric,\s*acc:\s*([\d.]+).*?norm_edit_dis:\s*([\d.]+)"
)


def parse_log(path):
    steps, epochs, losses = [], [], []
    eval_steps, eval_accs, eval_neds = [], [], []

    last_global_step = 0
    awaiting_eval = False
    pending_acc = None

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = STEP_LINE.search(line)
            if m:
                epoch, total_epochs, global_step, loss = m.groups()
                steps.append(int(global_step))
                epochs.append(int(epoch))
                losses.append(float(loss))
                last_global_step = int(global_step)
                continue

            m_best = BEST_METRIC_LINE.search(line)
            if m_best:
                eval_steps.append(last_global_step)
                eval_accs.append(float(m_best.group(1)))
                eval_neds.append(float(m_best.group(2)))
                continue

            if EVAL_MARKER.search(line):
                awaiting_eval = True
                pending_acc = None
                continue

            if awaiting_eval:
                m_acc = EVAL_ACC_LINE.search(line.strip())
                if m_acc:
                    pending_acc = float(m_acc.group(1))
                    continue
                m_ned = EVAL_NED_LINE.search(line.strip())
                if m_ned and pending_acc is not None:
                    eval_steps.append(last_global_step)
                    eval_accs.append(pending_acc)
                    eval_neds.append(float(m_ned.group(1)))
                    awaiting_eval = False
                    pending_acc = None
                    continue

    if not steps:
        raise ValueError(
            f"No training step lines matched in {path}. "
            "Check the log format hasn't changed, or that the file isn't empty."
        )

    epoch_losses = defaultdict(list)
    for e, l in zip(epochs, losses):
        epoch_losses[e].append(l)
    epoch_nums = sorted(epoch_losses)
    epoch_mean_loss = [sum(epoch_losses[e]) / len(epoch_losses[e]) for e in epoch_nums]
    epoch_last_step = {}
    for s, e in zip(steps, epochs):
        epoch_last_step[e] = s
    epoch_x = [epoch_last_step[e] for e in epoch_nums]

    return {
        "steps": steps,
        "losses": losses,
        "epoch_x": epoch_x,
        "epoch_mean_loss": epoch_mean_loss,
        "eval_steps": eval_steps,
        "eval_accs": eval_accs,
        "eval_neds": eval_neds,
    }


def plot(data, out_path, title_suffix=""):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))

    ax = axes[0]
    ax.plot(data["steps"], data["losses"], color="#888888", alpha=0.35,
            linewidth=0.7, label="per-batch loss")
    ax.plot(data["epoch_x"], data["epoch_mean_loss"], color="#1f4e79",
            linewidth=2.0, marker="o", markersize=3, label="per-epoch mean loss")
    ax.set_xlabel("Global step")
    ax.set_ylabel("CTC loss")
    ax.set_title(f"Training loss{title_suffix}")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.2)

    ax2 = axes[1]
    if data["eval_steps"]:
        ax2.plot(data["eval_steps"], data["eval_accs"], color="#1f4e79",
                 marker="o", markersize=4, label="best accuracy so far (eval)")
        ax2b = ax2.twinx()
        ax2b.plot(data["eval_steps"], data["eval_neds"], color="#c0392b",
                  marker="s", markersize=4, linestyle="--",
                  label="norm. edit dist. (eval)")
        ax2b.set_ylabel("Normalized edit distance", color="#c0392b")
        ax2b.tick_params(axis="y", labelcolor="#c0392b")
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2b.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=8, frameon=False, loc="lower right")
    else:
        ax2.text(0.5, 0.5, "No periodic eval checkpoints found in log",
                  ha="center", va="center", transform=ax2.transAxes, fontsize=9)
    ax2.set_xlabel("Global step")
    ax2.set_ylabel("Best accuracy so far", color="#1f4e79")
    ax2.tick_params(axis="y", labelcolor="#1f4e79")
    ax2.set_title(f"Held-out evaluation (best-so-far){title_suffix}")
    ax2.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved figure to {out_path}")
    print(f"  {len(data['steps'])} training-step points, "
          f"{len(data['epoch_x'])} epochs, "
          f"{len(data['eval_steps'])} eval checkpoints")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", default="results/training_curve.pdf")
    ap.add_argument("--title_suffix", default="")
    args = ap.parse_args()

    data = parse_log(args.log)
    plot(data, args.out, args.title_suffix)


if __name__ == "__main__":
    main()
