"""Regenerate the report figures from the adjacent summary; no simulation run."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parents[1]
data = json.loads((root / 'summary.json').read_text())
labels = ['Open', 'Fist', 'Thumb–index pinch', 'Thumb–middle pinch',
          'Intentional crossing', 'Finger spread', 'Synthetic grasp']
for metric, title, filename in [
    ('collision_rate', 'Detected collision rate', 'collision_rate.png'),
    ('margin_violation_rate', 'Safety-margin violation rate (distance < 5 mm)', 'margin_violation_rate.png'),
]:
    fig, ax = plt.subplots(figsize=(10, 5.5), layout='constrained')
    y = np.arange(len(labels))
    for mode, offset, color in [('OFF', -.18, '#bb6644'), ('ON', .18, '#247d91')]:
        values = [100 * m[mode][metric] for m in data['motions'].values()]
        ax.barh(y + offset, values, height=.32, label=mode, color=color)
        for pos, value in zip(y + offset, values):
            ax.text(value + .7, pos, f'{value:.1f}%', va='center', fontsize=9, color=color)
    ax.set(yticks=y, yticklabels=labels, xlim=(0, 70), xlabel='Control-step samples (%)', title=title)
    ax.invert_yaxis()
    ax.legend(loc='lower right')
    ax.grid(axis='x', alpha=.18)
    ax.set_axisbelow(True)
    for side in ['top', 'right']: ax.spines[side].set_visible(False)
    fig.text(.5, -.015, '120 control steps per motion and mode · synthetic robot targets · MuJoCo v1 right hand',
             ha='center', fontsize=9)
    fig.savefig(root / 'figures' / filename, dpi=160, bbox_inches='tight')
    plt.close(fig)
