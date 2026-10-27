"""Render the model-free GN reference from reference-policies.json (requires matplotlib)."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

INK, MUTED, SURFACE, BAND = '#0b0b0b', '#52514e', '#fcfcfb', '#d9d8d4'
HIT, IDENTIFIED = '#2a78d6', '#eb6834'
# Hit at any logged guess, 20 final cases per arm (gn-complete/README.md).
LLM_ARMS_HIT_ANY = {'ZeroShot': 0, 'FewShot': 0, 'FewShotInst': 1, 'ToT_parser_repair01': 2,
                    'ProactiveCoT': 1, 'DC_Cumulative': 0, 'ACE_Supervised': 2}


def main():
    base = Path(__file__).absolute().parent
    data = json.loads((base / 'reference-policies.json').read_text(encoding='utf-8'))
    mix = sorted((r for r in data['policies'] if r['policy'].startswith('mixture_')),
                 key=lambda r: r['consistent_choice_probability'])
    p = [r['consistent_choice_probability'] for r in mix]
    hit = [100 * r['hit_rate_within_26'] for r in mix]
    identified = [100 * r['identified_after_25_questions'] / r['secrets'] for r in mix]
    top = 100 * max(LLM_ARMS_HIT_ANY.values()) / 20

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'text.color': INK,
                         'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
                         'axes.edgecolor': MUTED, 'axes.spines.top': False,
                         'axes.spines.right': False, 'svg.hashsalt': 'gn-state-reference-v1'})
    fig, ax = plt.subplots(figsize=(8.6, 4.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.axhspan(0, top, color=BAND, zorder=0)
    ax.text(0.985, top + 1.5, 'Observed LLM arms (Qwen2.5-7B): secret hit in 0–2 of 20 cases',
            ha='right', va='bottom', color=MUTED, fontsize=9, transform=ax.get_yaxis_transform())
    ax.plot(p, identified, color=IDENTIFIED, lw=2, marker='s', ms=8, mec=SURFACE, mew=2,
            label='Secret identified by public feedback after 25 questions', zorder=3)
    ax.plot(p, hit, color=HIT, lw=2, marker='o', ms=8, mec=SURFACE, mew=2,
            label='Secret actually guessed within 26 guesses', zorder=4)
    for x, y in zip(p, hit):
        if x in (0.0, 0.05, 0.1, 0.2):
            ax.annotate(f'{y:.1f}%', (x, y), xytext=(10, 6) if x == 0 else (8, -14),
                        textcoords='offset points', color=INK, fontsize=9)
    ax.text(0.32, 102.5, 'Identified: 100% at every setting, including purely random questions',
            color=INK, fontsize=9, va='bottom')
    ax.set(xlabel='Probability that a guess is drawn from the feedback-consistent secrets '
                  '(otherwise a random unasked guess)',
           ylabel='Share of all 5,040 secrets', xlim=(-0.03, 1.03), ylim=(-3, 106))
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.grid(axis='y', color=BAND, lw=0.8)
    ax.set_title('GN, native 25 questions + final guess: random questions already carry the answer',
                 loc='left', fontsize=11, color=INK)
    ax.legend(loc='center right', frameon=False, bbox_to_anchor=(1.0, 0.45))
    fig.tight_layout()
    for suffix in ('svg', 'png'):
        fig.savefig(base / f'reference-policies.{suffix}', dpi=200, facecolor=SURFACE,
                    metadata={'Date': None} if suffix == 'svg' else None)


if __name__ == '__main__':
    main()
