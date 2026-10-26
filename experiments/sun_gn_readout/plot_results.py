"""Optional figure reproduction; requires matplotlib, never loads a model."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
result = json.loads((ROOT / 'results.json').read_bytes())['prospective_state_readout']
assert result['status'] == 'AUDITED_COMPLETE' and result['N'] == 50
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                     'svg.fonttype': 'none', 'axes.spines.top': False,
                     'axes.spines.right': False})
fig, ax = plt.subplots(figsize=(9.6, 5.4), constrained_layout=True)
names = ['Baseline model', 'State-assisted model', 'Direct unique-only solver']
correct = [result['conditions']['baseline']['EM'],
           result['conditions']['state_assisted']['EM'], result['symbolic']['correct']]
abstained = [0, 0, result['symbolic']['abstained']]
wrong = [50 - c - a for c, a in zip(correct, abstained)]
y = [2, 1, 0]
ax.barh(y, correct, color='#19766E', label='Correct')
ax.barh(y, wrong, left=correct, color='#CB755C', label='Wrong / invalid')
ax.barh(y, abstained, left=[c + w for c, w in zip(correct, wrong)],
        color='#D5DADC', label='Abstained')
for pos, c in zip(y, correct):
    ax.text(max(c / 2, 2.8), pos, f'{c}/50', va='center', ha='center',
            color='white', weight='bold', fontsize=13)
ax.set_yticks(y, names)
ax.set_xlim(0, 50)
ax.set_xticks(range(0, 51, 10))
ax.set_xlabel('Cases in the fixed prospective internal sample (N = 50)')
ax.set_title('Tool-assisted readout recovers 46 answers', loc='left',
             fontsize=19, weight='bold', pad=43)
ax.text(0, 1.035, 'Paired gain over baseline: +92 pp; bootstrap 95% CI [84, 98] pp',
        transform=ax.transAxes, fontsize=11)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.20), ncols=3, frameon=False)
fig.text(.01, -.06, 'Same 25-turn histories and model final cap. Solver supplies the answer on singleton cases.\n'
         'The model adds no overall EM over the direct solver. Shared interaction cost remains.\n'
         'One model/decoding setup; outcome-informed follow-up, not evidence of learned reasoning.',
         fontsize=9, color='#47555C')
for ext in ('svg', 'png'):
    fig.savefig(ROOT / ('results.' + ext), dpi=180, bbox_inches='tight',
                metadata={'Creator': 'GN public-evidence experiment'})
print('Saved results.svg and results.png from released aggregates.')
