"""Render the measured GN state metrics from state-tracking-gn-complete.json (requires matplotlib)."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

INK, MUTED, SURFACE, GRID, NEUTRAL = '#0b0b0b', '#52514e', '#fcfcfb', '#d9d8d4', '#8a8984'
DETERMINED, CORRECT = '#eb6834', '#2a78d6'
LEARNERS = {'DC_Cumulative': ('DC', '#1baf7a', 'o'), 'ACE_Supervised': ('ACE', '#6250d6', 's')}
ARMS = [('ZeroShot', 'ZeroShot'), ('FewShot', 'FewShot'), ('FewShotInst', 'FewShotInst'),
        ('ProactiveCoT', 'ProactiveCoT'), ('ToT_parser_repair01', 'ToT (parser repair)'),
        ('DC_Cumulative', 'DC (evolved)'), ('ACE_Supervised', 'ACE (evolved)')]


def style():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'text.color': INK,
                         'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
                         'axes.edgecolor': MUTED, 'axes.spines.top': False,
                         'axes.spines.right': False, 'svg.hashsalt': 'gn-state-traces-v1'})


def save(fig, base, name):
    for suffix in ('svg', 'png'):
        fig.savefig(base / f'{name}.{suffix}', dpi=200, facecolor=SURFACE,
                    metadata={'Date': None} if suffix == 'svg' else None)


def final_arms(groups, base):
    rows = {g['arm']: g for g in groups if g['phase'] == 'final'}
    y = list(range(len(ARMS)))[::-1]
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.6), sharey=True, facecolor=SURFACE,
                             gridspec_kw={'width_ratios': [1.5, 1, 1]})
    ax = axes[0]
    determined = [rows[a]['identified_before_last_guess_episodes'] for a, _ in ARMS]
    correct = [rows[a]['last_guess_correct_episodes'] for a, _ in ARMS]
    ax.barh([v + 0.19 for v in y], determined, height=0.36, color=DETERMINED,
            label='Secret determined by feedback before the final guess')
    ax.barh([v - 0.19 for v in y], correct, height=0.36, color=CORRECT, label='Final guess correct')
    for v, d, c in zip(y, determined, correct):
        ax.text(d + 0.3, v + 0.19, str(d), va='center', fontsize=9, color=INK)
        ax.text(c + 0.3, v - 0.19, str(c), va='center', fontsize=9, color=INK)
    ax.set(xlim=(0, 22), xticks=[0, 5, 10, 15, 20], xlabel='Cases out of 20', title='Information vs. its use')
    fig.legend(*ax.get_legend_handles_labels(), loc='upper left', bbox_to_anchor=(0.01, 0.925),
               ncol=2, frameon=False, fontsize=9.5)
    for ax, key, title in [(axes[1], 'consistency_rate_pooled', 'Guesses consistent with earlier feedback'),
                           (axes[2], 'repeat_rate', 'Guesses that repeat an earlier guess')]:
        values = [100 * rows[a][key] for a, _ in ARMS]
        ax.barh(y, values, height=0.5, color=NEUTRAL)
        top = max(values)
        for v, x in zip(y, values):
            ax.text(x + top * 0.03, v, f'{x:.1f}%', va='center', fontsize=9, color=INK)
        ax.set(xlim=(0, top * 1.3), xlabel='% of guesses', title=title)
    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.grid(axis='x', color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.title.set_fontsize(10.5)
    axes[0].set_yticks(y, [label for _, label in ARMS])
    fig.suptitle('GN final, Qwen2.5-7B, 20 cases per arm: the feedback usually pins down the secret; '
                 'the model almost never uses it', x=0.01, ha='left', fontsize=11.5)
    fig.text(0.01, 0.01, 'Measured offline from the saved traces. "Determined" = exactly one secret '
             'is consistent with all earlier feedback (computed offline; never shown to the model).',
             fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 0.87))
    save(fig, base, 'gn-final-arms-state')


def dev_curves(groups, base):
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.9), facecolor=SURFACE)
    panels = [('identified_before_last_guess_episodes', 'Secret determined before the final guess', 1, 'cases out of 5', (-0.3, 5.6), True),
              ('consistency_rate_pooled', 'Guesses consistent with earlier feedback', 100, '% of guesses', (-2, 52), False),
              ('repeat_rate', 'Guesses that repeat an earlier guess', 100, '% of guesses', (-2, 52), True)]
    for ax, (key, title, scale, unit, ylim, value_labels) in zip(axes, panels):
        lines = {}
        for arm, (label, color, marker) in LEARNERS.items():
            rows = sorted((g for g in groups if g['phase'] == 'dev' and g['arm'] == arm), key=lambda g: g['checkpoint'])
            xs = [g['checkpoint'] for g in rows]
            ys = [scale * g[key] for g in rows]
            lines[label] = (xs, ys)
            ax.plot(xs, ys, color=color, lw=2, marker=marker, ms=8, mec=SURFACE, mew=2)
            if value_labels:
                for x, v in zip(xs, ys):
                    ax.annotate(f'{v:.0f}' if scale == 1 else f'{v:.0f}%', (x, v), xytext=(0, 9),
                                textcoords='offset points', ha='center', fontsize=8.5, color=MUTED)
        close = abs(lines['DC'][1][-1] - lines['ACE'][1][-1]) < 0.06 * (ylim[1] - ylim[0])
        for label, (xs, ys) in lines.items():
            dy = (8 if label == 'DC' else -8) if close else 0
            ax.annotate(label, (xs[-1], ys[-1]), xytext=(8, dy), textcoords='offset points',
                        va='center', fontsize=9.5, color=INK)
        if not value_labels:
            values = '; '.join(f'{label} ' + ', '.join(f'{v:.1f}' for v in ys) + '%' for label, (_, ys) in lines.items())
            ax.text(3, 26, f'At most 1% at every checkpoint\n({values})', ha='center', fontsize=9, color=MUTED)
        ax.set(xticks=[0, 3, 6], xlim=(-0.6, 7.2), ylim=ylim, xlabel='Completed memory updates',
               ylabel=unit, title=title)
        ax.set_facecolor(SURFACE)
        ax.grid(axis='y', color=GRID, lw=0.8)
        ax.title.set_fontsize(10.5)
    fig.suptitle('GN development, the same 5 cases at updates 0, 3 and 6: evolved memories never raise '
                 'consistency and add repeats; final guess correct 0/5 throughout', x=0.01, ha='left', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, base, 'gn-dev-state-curves')


def main():
    base = Path(__file__).absolute().parent
    groups = json.loads((base / 'state-tracking-gn-complete.json').read_text(encoding='utf-8'))['groups']
    style()
    final_arms(groups, base)
    dev_curves(groups, base)


if __name__ == '__main__':
    main()
