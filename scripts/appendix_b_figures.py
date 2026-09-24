"""Appendix B stimulus-quality figures: the intent-polarity Sankey.

Answers the question the appendix poses: does a stimulus generated from a
negative concept actually get a negative-toned response, and so on. The flow is
computed for every stimulus in all three corpora -- not the n=132 calibration
subset the old `plot_sankey` in annotators_performance_analysis.py ran on.

Polarity comes from src/nlp/emotion_polarity.py on both sides: the target
concept the stimulus was generated from, and the judge's first Ekman label for
the response. No extra LLM pass is involved.

Drawn in matplotlib rather than plotly so the figure lands at its exact on-page
size and needs no browser-based image exporter.

    python scripts/appendix_b_figures.py
"""

import json
import os
import re
import sys
from collections import Counter

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Rectangle

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src", "nlp"))
sys.path.insert(0, os.path.join(REPO, "src", "probes"))

from emotion_polarity import (  # noqa: E402
    concept_polarity, label_polarity, POLARITY_ORDER, POLARITY_COLORS)
from paper_style import apply_paper_style, TEXTWIDTH_IN, SAVE_DPI  # noqa: E402

OUT_DIR = os.path.join(REPO, "figures", "manuscript_revised")

MODEL = "Llama-2-7b-chat-hf"
TAXONOMY = "ekman_basic_emotions"
DOMAINS = [
    ("aicentric", "AI-centric"),
    ("humanprompts", "human 3rd-person"),
    ("humanconv", "human conversational"),
]
CUSTOM_ID = re.compile(r"response-(\d+)-(.+)")


def load_flow(model, domain_key):
    """(prompt polarity, response polarity) counts for one model x domain."""
    out_path = os.path.join(REPO, "data", "02_generated",
                            f"outputs_{model}_20260625_{domain_key}.jsonl")
    ann_path = os.path.join(REPO, "data", "04_annotated",
                            f"batch_results_{model}_20260625_{domain_key}.jsonl")

    # The response file is ordered, and prompt_key is prompt_<line index>; the
    # annotation custom_id carries the same index. Verified over the full file.
    concepts = []
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            try:
                concepts.append(json.loads(line).get("emotion_considered"))
            except ValueError:
                concepts.append(None)

    flow, unmapped = Counter(), Counter()
    with open(ann_path, encoding="utf-8") as f:
        for line in f:
            if TAXONOMY not in line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            m = CUSTOM_ID.match(rec.get("custom_id", "") or "")
            if not m or m.group(2) != TAXONOMY:
                continue
            idx = int(m.group(1))
            if idx >= len(concepts):
                continue
            labels = _emotions(rec)
            if not labels:
                continue
            src = concept_polarity(concepts[idx])
            dst = label_polarity(labels[0], TAXONOMY)
            if src is None:
                unmapped[("concept", concepts[idx])] += 1
            elif dst is None:
                unmapped[("label", labels[0])] += 1
            else:
                flow[(src, dst)] += 1
    return flow, unmapped


def _emotions(record):
    """Judge output parser, same as scripts/annotations_and_activations_merge.py."""
    try:
        for item in record.get("response", {}).get("body", {}).get("output", []):
            if item.get("type") == "message":
                content = item.get("content", [])
                if not content:
                    return None
                text = content[0].get("text")
                if not text:
                    return None
                clean = text.replace("```json", "").replace("```", "").strip()
                return json.loads(clean).get("emotions", [])
    except (json.JSONDecodeError, AttributeError, IndexError):
        return None
    return None


def _ribbon(ax, x0, x1, y0a, y0b, y1a, y1b, color):
    """One flow band: cubic Beziers along the top and bottom edges."""
    cx = (x0 + x1) / 2
    verts = [(x0, y0a),
             (cx, y0a), (cx, y1a), (x1, y1a),      # top edge
             (x1, y1b),
             (cx, y1b), (cx, y0b), (x0, y0b),      # bottom edge back
             (x0, y0a)]
    codes = [Path.MOVETO,
             Path.CURVE4, Path.CURVE4, Path.CURVE4,
             Path.LINETO,
             Path.CURVE4, Path.CURVE4, Path.CURVE4,
             Path.CLOSEPOLY]
    ax.add_patch(PathPatch(Path(verts, codes), facecolor=color, alpha=0.5,
                           edgecolor="none", linewidth=0))


def draw_sankey(ax, flow, gap=0.035, node_w=0.028, x0=0.20, x1=0.80):
    """3 -> 3 polarity flow, node heights proportional to their totals."""
    total = sum(flow.values())
    left = {p: sum(v for (s, _), v in flow.items() if s == p) for p in POLARITY_ORDER}
    right = {p: sum(v for (_, d), v in flow.items() if d == p) for p in POLARITY_ORDER}

    def stack(totals):
        n_used = sum(1 for p in POLARITY_ORDER if totals[p])
        span = 1.0 - gap * max(n_used - 1, 0)
        pos, y = {}, 1.0
        for p in POLARITY_ORDER:
            h = totals[p] / total * span
            pos[p] = (y - h, y)          # (bottom, top)
            if totals[p]:
                y -= h + gap
        return pos

    lpos, rpos = stack(left), stack(right)
    lcur = {p: lpos[p][1] for p in POLARITY_ORDER}
    rcur = {p: rpos[p][1] for p in POLARITY_ORDER}

    for s in POLARITY_ORDER:
        for d in POLARITY_ORDER:
            v = flow.get((s, d), 0)
            if not v:
                continue
            hs = v / total * (1.0 - gap * 2)
            hd = v / total * (1.0 - gap * 2)
            _ribbon(ax, x0 + node_w, x1, lcur[s], lcur[s] - hs,
                    rcur[d], rcur[d] - hd, POLARITY_COLORS[s])
            lcur[s] -= hs
            rcur[d] -= hd

    for p in POLARITY_ORDER:
        for pos, x, ha, tx in ((lpos, x0, "right", x0 - 0.015),
                               (rpos, x1, "left", x1 + node_w + 0.015)):
            bot, top = pos[p]
            if top - bot <= 0:
                continue
            ax.add_patch(Rectangle((x, bot), node_w, top - bot,
                                   facecolor=POLARITY_COLORS[p], edgecolor="none"))
            share = (left if x == x0 else right)[p] / total
            ax.text(tx, (bot + top) / 2, f"{p.split('/')[0]}  {share*100:.0f}%",
                    ha=ha, va="center", fontsize=6.5)

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.02, 1.02)
    ax.axis("off")


def main():
    apply_paper_style()
    os.makedirs(OUT_DIR, exist_ok=True)

    fig, axes = plt.subplots(len(DOMAINS), 1,
                             figsize=(TEXTWIDTH_IN, 1.15 * len(DOMAINS) + 0.35),
                             layout="constrained")

    for ax, (key, pretty), letter in zip(axes, DOMAINS, ["(a)", "(b)", "(c)"]):
        flow, unmapped = load_flow(MODEL, key)
        draw_sankey(ax, flow)
        ax.text(0.0, 1.0, f"{letter} {pretty}", transform=ax.transAxes,
                fontsize=8, fontweight="bold", va="bottom", ha="left")

        total = sum(flow.values())
        kept = {p: sum(v for (s, d), v in flow.items() if s == p and d == p)
                for p in POLARITY_ORDER}
        print(f"{pretty:<22} n={total:<6} "
              f"same-polarity: pos {kept['positive']/max(sum(v for (s,_),v in flow.items() if s=='positive'),1)*100:.0f}% "
              f"neg {kept['negative']/max(sum(v for (s,_),v in flow.items() if s=='negative'),1)*100:.0f}%")
        if unmapped:
            print("   unmapped:", unmapped.most_common(5))

    handles = [mpl.patches.Patch(facecolor=POLARITY_COLORS[p], label=p)
               for p in POLARITY_ORDER]
    fig.legend(handles=handles, loc="outside lower center", ncol=3,
               fontsize=7.5, frameon=False)

    path = os.path.join(OUT_DIR, "figB_intent_polarity_sankey.png")
    fig.savefig(path, dpi=SAVE_DPI)
    plt.close(fig)
    print("\n ", path)




# =============================================================================
# N-gram concentration: is the corpus templated or diverse?
# =============================================================================
# The claim Appendix B makes is that noise injection produced diversity rather
# than a few templates with surface variation. The check that shows this is
# concentration: how much of the corpus the most frequent n-grams account for.
# A templated corpus reaches high coverage within a handful of ranks.
#
# n-grams are counted WITHIN each prompt. Building one flat token stream over
# the whole corpus would invent an n-gram at every prompt boundary (three of
# them per boundary for 4-grams).

import csv  # noqa: E402

CORPORA = {
    "ai_centric": ("AI-centric",
                   "data/01_stimuli/generated_prompts/"
                   "generated_emotional_prompts_batched.csv"),
    "human_3rd": ("human 3rd-person",
                  "data/01_stimuli/generated_human_prompts/"
                  "generated_human_emotional_prompts_batched.csv"),
    "human_conv": ("human conversational",
                   "data/01_stimuli/generated_human_conversation_prompts/"
                   "generated_human_conversation_prompts_batched.csv"),
}
NGRAM_ORDERS = [2, 3, 4]
NGRAM_STYLE = {2: ("#0072B2", "-"), 3: ("#D55E00", "-"), 4: ("#009E73", "-")}


def _tokens(text, stop):
    clean = re.sub(r"[^\w\s]", "", str(text).lower())
    return [w for w in clean.split() if w not in stop]


def ngram_counts(prompts, n, stop):
    c = Counter()
    for p in prompts:
        toks = _tokens(p, stop)
        for i in range(len(toks) - n + 1):
            c[" ".join(toks[i:i + n])] += 1
    return c


def make_ngram_figure():
    import pandas as pd
    from nltk.corpus import stopwords
    stop = set(stopwords.words("english"))

    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH_IN, 2.15),
                             sharey=True, layout="constrained")

    for ax, (tag, (pretty, path)) in zip(axes, CORPORA.items()):
        df = pd.read_csv(os.path.join(REPO, path))
        prompts = [p for p in df["generated_prompt"].astype(str)
                   if not p.startswith("JSON Decode Error")]

        for n in NGRAM_ORDERS:
            counts = ngram_counts(prompts, n, stop)
            freqs = np.array(sorted(counts.values(), reverse=True))
            # Share of prompts containing the rank-k n-gram, read against the
            # 100% line: a corpus built from a few templates would have its
            # top n-grams sitting up at that ceiling.
            share = freqs / len(prompts) * 100
            cum = np.cumsum(freqs) / freqs.sum() * 100
            color, ls = NGRAM_STYLE[n]
            ax.plot(np.arange(1, len(share) + 1), share, color=color,
                    linestyle=ls, linewidth=1.4, label=f"{n}-grams")

            top = counts.most_common(100)
            out = os.path.join(OUT_DIR, f"ngrams_top100_{tag}_{n}gram.csv")
            with open(out, "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow([f"{n}gram", "count", "share_of_prompts_pct"])
                for g, v in top:
                    w.writerow([g, v, f"{v / len(prompts) * 100:.3f}"])
            if n == 4:
                print(f"{pretty:<22} n={len(prompts):<6} most frequent 4-gram "
                      f"'{top[0][0]}' in {top[0][1] / len(prompts) * 100:.2f}% of prompts; "
                      f"top-100 cover {cum[99]:.1f}% of 4-gram tokens")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1, 1e4)
        ax.set_ylim(1e-3, 200)
        ax.axhline(100, color="0.35", linestyle=":", linewidth=0.9)
        ax.grid(True, linestyle="--", alpha=0.5, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.set_xlabel("n-gram rank")
        ax.set_title(pretty, fontsize=8, fontweight="bold")

    axes[0].set_ylabel("share of prompts\ncontaining it (%)")
    axes[0].legend(loc="lower left", fontsize=7)

    path = os.path.join(OUT_DIR, "figB_ngram_concentration.png")
    fig.savefig(path, dpi=SAVE_DPI)
    plt.close(fig)
    print("\n ", path)


if __name__ == "__main__":
    main()
    make_ngram_figure()
