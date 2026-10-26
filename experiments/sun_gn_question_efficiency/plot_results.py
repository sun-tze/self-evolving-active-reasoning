"""Reproduce the released efficiency figure; requires matplotlib, no model calls."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
data = json.loads((ROOT / 'anonymous-cases.json').read_text(encoding='utf8'))['expected']
conditions, stats = data['conditions'], data['statistics']
arms = ['B0', 'S', 'E0', 'E1', 'E2']
colors = ['#183A44', '#AAB2B7', '#CE785C', '#B69461', '#367E75']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'svg.fonttype': 'none', 'axes.spines.top': False,
                     'axes.spines.right': False})
fig, axes = plt.subplots(1, 2, figsize=(12, 5.8))
fig.subplots_adjust(top=.78, bottom=.28, left=.08, right=.98, wspace=.25)
means = [conditions[a]['mean_capped25'] for a in arms]
axes[0].bar(arms, means, color=colors, width=.65)
for x, a, value in zip(range(5), arms, means):
    axes[0].text(x, value+.35, f'{value:.2f}', ha='center', fontsize=11)
axes[0].set_ylim(0, 26)
axes[0].set_ylabel('Queries to identification, capped at 25')
axes[0].set_title('Lower cost is better', loc='left', fontsize=13)
turns = range(1, 26)
for a, color in zip(arms, colors):
    axes[1].step(turns, [conditions[a]['resolution_curve'][str(t)] for t in turns],
                 where='post', label=f"{a}: {conditions[a]['resolved']}/100", color=color,
                 linewidth=2 if a in ('B0', 'E2') else 1.5)
axes[1].set(xlim=(1, 25), ylim=(0, 100), xlabel='Queries allowed', ylabel='Cases uniquely identified /100')
axes[1].set_title('Coverage must accompany cost', loc='left', fontsize=13)
axes[1].legend(frameon=False, fontsize=9, loc='lower right')
primary = stats['primary']
fig.suptitle('Evolved notes did not improve efficiency over no note', x=.08, ha='left', fontsize=18, weight='bold')
fig.text(.08, .85, f"Primary mean(E0,E1,E2) − B0: +{primary['mean_difference']:.2f} queries; 95% CI [{primary['CI95'][0]:.2f}, {primary['CI95'][1]:.2f}]", fontsize=12)
fig.text(.08, .14, 'B0: no note. S: one rules-only note. E0–E2: all three final adapted notes; none selected after testing.', fontsize=10)
fig.text(.08, .095, 'Fixed internal N100; first public singleton or 25 queries. Failed cases stay in every comparison.', fontsize=10)
fig.text(.08, .05, 'Case-block intervals condition on three realized chains sharing 18 training cases. Outcome-informed follow-up.', fontsize=9, color='#4D5B60')
for extension in ('svg', 'png'):
    fig.savefig(ROOT / f'results.{extension}', dpi=180, metadata={'Creator': 'GN question-efficiency study'})
print('Saved efficiency figure from released anonymous outcomes.')
