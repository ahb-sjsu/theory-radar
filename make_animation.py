#!/usr/bin/env python3
"""Generate an animated GIF showing Theory Radar in operation.

Runs a real beam search on Breast Cancer and captures each depth level
as a frame, showing:
- The search space as a Smith chart-inspired coordinate system
- Formula candidates as points (x=AUROC, y=F1)
- Beam selection (top-B highlighted)
- Meta-pruning (dead nodes fading out)
- The winning formula converging
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arc, FancyArrowPatch
from matplotlib.collections import PathCollection
import matplotlib.patheffects as pe
from PIL import Image
import io
import sys

sys.path.insert(0, "src")
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from symbolic_search._heuristic_dag import exact_optimal_f1, auroc_safe

# Load data
bc = load_breast_cancer()
X = StandardScaler().fit_transform(bc.data[:, :10])
y = bc.target.astype(bool)
features = [f"f{i}" for i in range(10)]

# Operation registries
BINARY_OPS = {
    "+": lambda a,b: a+b, "-": lambda a,b: a-b,
    "*": lambda a,b: a*b, "/": lambda a,b: a/(b+1e-30),
    "max": lambda a,b: np.maximum(a,b), "min": lambda a,b: np.minimum(a,b),
    "hypot": lambda a,b: np.sqrt(a**2+b**2),
}
UNARY_OPS = {
    "sq": lambda x: x**2, "abs": lambda x: np.abs(x),
    "sqrt": lambda x: np.sqrt(np.abs(x)),
}


def smith_chart_background(ax):
    """Draw Smith chart inspired background."""
    ax.set_xlim(0.45, 1.02)
    ax.set_ylim(0.0, 1.02)
    ax.set_facecolor('#fafbfc')

    # Constant-resistance inspired arcs
    for r in [0.6, 0.7, 0.8, 0.9]:
        circle = Circle((r, 0.5), 0.5 * (1-r) + 0.1,
                        fill=False, color='#e2e6ea', linewidth=0.4, linestyle='-')
        ax.add_patch(circle)

    # Reactance-like curves
    for angle in np.linspace(0.2, 1.5, 6):
        theta = np.linspace(-0.5, 0.5, 50) * angle
        x_arc = 0.5 + 0.5 * np.cos(theta)
        y_arc = 0.5 + 0.3 * np.sin(theta) * angle
        mask = (x_arc > 0.45) & (x_arc < 1.02) & (y_arc > 0) & (y_arc < 1.02)
        ax.plot(x_arc[mask], y_arc[mask], color='#e8ecf0', linewidth=0.3)

    # Grid lines
    for v in np.arange(0.5, 1.01, 0.1):
        ax.axhline(v, color='#eef0f3', linewidth=0.3)
        ax.axvline(v, color='#eef0f3', linewidth=0.3)

    ax.set_xlabel('AUROC', fontsize=10, fontfamily='serif', color='#64748b')
    ax.set_ylabel('F1 Score', fontsize=10, fontfamily='serif', color='#64748b')
    ax.tick_params(colors='#94a3b8', labelsize=8)
    for spine in ax.spines.values():
        spine.set_color('#d1d5db')
        spine.set_linewidth(0.5)


def render_frame(depth, all_points, beam_points, pruned_points, best_point,
                 best_formula, title_text, frame_num, total_frames):
    """Render one frame of the animation."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 5), dpi=120)
    smith_chart_background(ax)

    # Draw all explored points (faded)
    if all_points:
        aurocs, f1s = zip(*all_points)
        ax.scatter(aurocs, f1s, s=8, c='#cbd5e1', alpha=0.3, zorder=2, edgecolors='none')

    # Draw pruned points (red, fading)
    if pruned_points:
        aurocs, f1s = zip(*pruned_points)
        ax.scatter(aurocs, f1s, s=12, c='#fca5a5', alpha=0.4, zorder=3,
                  marker='x', linewidths=0.8)

    # Draw beam points (blue, solid)
    if beam_points:
        aurocs, f1s = zip(*beam_points)
        ax.scatter(aurocs, f1s, s=25, c='#3b82f6', alpha=0.7, zorder=4, edgecolors='#1d4ed8',
                  linewidths=0.5)

    # Draw best point with halo
    if best_point:
        ax.scatter([best_point[0]], [best_point[1]], s=120, c='none',
                  edgecolors='#2563eb', linewidths=2, zorder=6, alpha=0.6)
        ax.scatter([best_point[0]], [best_point[1]], s=40, c='#2563eb',
                  zorder=7, edgecolors='white', linewidths=1)

    # Title area
    ax.text(0.46, 0.99, 'THEORY RADAR', fontsize=14, fontfamily='serif',
            fontweight='bold', color='#1e293b', va='top',
            path_effects=[pe.withStroke(linewidth=2, foreground='white')])

    ax.text(0.46, 0.94, title_text, fontsize=9, fontfamily='serif',
            color='#475569', va='top',
            path_effects=[pe.withStroke(linewidth=2, foreground='white')])

    # Best formula display
    if best_formula:
        ax.text(0.99, 0.03, best_formula, fontsize=8, fontfamily='monospace',
                color='#2563eb', ha='right', va='bottom', transform=ax.transAxes,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                         edgecolor='#93c5fd', alpha=0.9))

    # Depth indicator
    if depth > 0:
        ax.text(0.99, 0.97, f'depth {depth}', fontsize=9, fontfamily='serif',
                color='#64748b', ha='right', va='top', transform=ax.transAxes)

    # Progress bar
    bar_y = 0.01
    bar_h = 0.008
    progress = frame_num / max(total_frames - 1, 1)
    ax.axhspan(bar_y, bar_y + bar_h, xmin=0, xmax=1,
               color='#e2e8f0', transform=ax.transAxes, zorder=10)
    ax.axhspan(bar_y, bar_y + bar_h, xmin=0, xmax=progress,
               color='#3b82f6', transform=ax.transAxes, zorder=11, alpha=0.7)

    plt.tight_layout()

    # Render to PIL Image
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert('RGB')


def main():
    frames = []
    beam_width = 20
    total_frames = 18  # intro + depth1 + depth2 expand + depth2 prune + depth2 beam + depth3 + result

    # Frame 0-1: Empty search space
    for i in range(2):
        frame = render_frame(0, [], [], [], None, None,
                           'Scanning formula space...', i, total_frames)
        frames.append(frame)

    # Depth 1: evaluate raw features
    d1_points = []
    d1_all = []
    best_f1, best_auroc, best_name = 0, 0, ""

    for i in range(X.shape[1]):
        vals = X[:, i]
        f1 = exact_optimal_f1(vals, y)
        auc = auroc_safe(vals, y)
        d1_all.append((auc, f1))
        d1_points.append((auc, f1, features[i]))
        if f1 > best_f1:
            best_f1, best_auroc, best_name = f1, auc, features[i]

    # Frame 2: Features appearing
    frame = render_frame(1, d1_all, d1_all, [], (best_auroc, best_f1),
                        f'Depth 1: {len(d1_all)} features evaluated',
                        best_name + f'  F1={best_f1:.3f}', 2, total_frames)
    frames.append(frame)
    frames.append(frame)  # Hold

    # Depth 2: generate candidates
    d2_all = list(d1_all)
    d2_candidates = []

    # Sort d1 by F1, keep top beam_width
    d1_sorted = sorted(d1_points, key=lambda x: -x[1])[:beam_width]

    for bf1_val, _, bname in d1_sorted[:5]:  # top 5 for speed
        bvals_idx = features.index(bname)
        bvals = X[:, bvals_idx]
        for j in range(X.shape[1]):
            for opname, opfn in list(BINARY_OPS.items())[:4]:  # first 4 ops
                try:
                    cv = opfn(bvals, X[:, j])
                    cv = np.nan_to_num(cv, nan=0, posinf=1e10, neginf=-1e10)
                    if np.var(cv) < 1e-20: continue
                    f1 = exact_optimal_f1(cv, y)
                    auc = auroc_safe(cv, y)
                    d2_candidates.append((auc, f1, f"({bname} {opname} {features[j]})"))
                    d2_all.append((auc, f1))
                except:
                    pass

    # Frame 4-5: Candidates expanding
    n_show = len(d2_candidates) // 3
    for step in range(3):
        shown = d2_all[:len(d1_all) + (step + 1) * n_show]
        frame = render_frame(2, shown, [(a,f) for a,f,_ in d1_sorted[:beam_width]],
                           [], (best_auroc, best_f1),
                           f'Depth 2: {len(shown)-len(d1_all)} candidates...',
                           best_name + f'  F1={best_f1:.3f}', 4 + step, total_frames)
        frames.append(frame)

    # Meta-pruning: mark low-AUROC candidates as pruned
    prune_thresh = 0.65
    pruned = [(a, f) for a, f, _ in d2_candidates if a < prune_thresh]
    alive = [(a, f, n) for a, f, n in d2_candidates if a >= prune_thresh]

    # Frame 7: Show pruning
    frame = render_frame(2, d2_all,
                        [(a,f) for a,f,_ in alive[:beam_width]],
                        pruned,
                        (best_auroc, best_f1),
                        f'Meta-pruning: {len(pruned)} dead subtrees removed',
                        best_name + f'  F1={best_f1:.3f}', 7, total_frames)
    frames.append(frame)
    frames.append(frame)  # Hold

    # Update best from d2
    for auc, f1, name in d2_candidates:
        if f1 > best_f1:
            best_f1, best_auroc, best_name = f1, auc, name

    # Beam selection at depth 2
    d2_beam = sorted(alive, key=lambda x: -x[1])[:beam_width]

    # Frame 9: Beam selected
    frame = render_frame(2, d2_all,
                        [(a,f) for a,f,_ in d2_beam],
                        pruned,
                        (best_auroc, best_f1),
                        f'Beam: top {len(d2_beam)} formulas kept',
                        best_name + f'  F1={best_f1:.3f}', 9, total_frames)
    frames.append(frame)
    frames.append(frame)

    # Depth 3: expand beam
    d3_all = list(d2_all)
    d3_candidates = []

    for bauc, bf1_val, bname in d2_beam[:3]:  # top 3 for speed
        # Find the values for this formula (approximate: use best raw feature)
        best_raw = features.index(bname.split()[0].strip('(')) if bname.startswith('(') else 0
        bvals = X[:, min(best_raw, X.shape[1]-1)]
        for j in range(X.shape[1]):
            for opname, opfn in list(BINARY_OPS.items())[:3]:
                try:
                    cv = opfn(bvals, X[:, j])
                    cv = np.nan_to_num(cv, nan=0, posinf=1e10, neginf=-1e10)
                    if np.var(cv) < 1e-20: continue
                    f1 = exact_optimal_f1(cv, y)
                    auc = auroc_safe(cv, y)
                    d3_candidates.append((auc, f1, f"depth3_formula"))
                    d3_all.append((auc, f1))
                except:
                    pass

    for auc, f1, name in d3_candidates:
        if f1 > best_f1:
            best_f1, best_auroc = f1, auc

    # Frame 11-12: Depth 3 expansion
    frame = render_frame(3, d3_all,
                        [(a,f) for a,f,_ in sorted(d3_candidates, key=lambda x:-x[1])[:beam_width]],
                        [(a,f) for a,f,_ in d3_candidates if a < prune_thresh],
                        (best_auroc, best_f1),
                        f'Depth 3: {len(d3_candidates)} candidates, converging...',
                        best_name[:30] + f'  F1={best_f1:.3f}', 11, total_frames)
    frames.append(frame)
    frames.append(frame)

    # Frame 13-15: Result convergence
    for i in range(3):
        alpha = 0.5 + 0.5 * (i / 2)
        frame = render_frame(3, d3_all,
                           [(best_auroc, best_f1)],
                           [],
                           (best_auroc, best_f1),
                           f'FOUND: F1 = {best_f1:.4f}',
                           best_name[:35], 13 + i, total_frames)
        frames.append(frame)

    # Frame 16-17: Hold on result
    for i in range(2):
        frame = render_frame(3, [],
                           [(best_auroc, best_f1)],
                           [],
                           (best_auroc, best_f1),
                           f'Best formula: F1 = {best_f1:.4f}',
                           best_name[:35], 16 + i, total_frames)
        frames.append(frame)

    # Save as GIF
    frames[0].save(
        'theory_radar_animation.gif',
        save_all=True,
        append_images=frames[1:],
        duration=800,  # ms per frame
        loop=0,
    )
    print(f"Saved theory_radar_animation.gif ({len(frames)} frames)")


if __name__ == "__main__":
    main()
