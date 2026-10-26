"""Portable paired-case statistics extracted from the independent audit."""

import random

import statistics



def require(condition, message):
    if not condition:
        raise ValueError(message)



def percentile(values, q):
    values = sorted(values)
    point = (len(values)-1)*q
    lo = int(point)
    return values[lo] + (values[min(lo+1, len(values)-1)]-values[lo])*(point-lo)



def paired_statistics(conditions, replicates=10000, bootstrap_seed=62026):
    """Resample shared case blocks, preserving chains and every comparator."""
    names = ('B0', 'S', 'E0', 'E1', 'E2')
    require(set(conditions) == set(names), 'Statistical arm coverage')
    n = len(conditions['B0'])
    require(n > 0 and all(len(conditions[name]) == n for name in names), 'Statistical length mismatch')
    ids = [row['case_id'] for row in conditions['B0']]
    require(len(set(ids)) == n and all([row['case_id'] for row in conditions[name]] == ids for name in names), 'Unpaired statistics')
    delta = {}
    for key in ('capped25', 'penalized26'):
        mean_e = [sum(conditions[name][i][key] for name in ('E0', 'E1', 'E2')) / 3 for i in range(n)]
        for comparison in ('B0', 'S'):
            delta[f'mean_E_minus_{comparison}/{key}'] = [mean_e[i] - conditions[comparison][i][key] for i in range(n)]
        for chain in ('E0', 'E1', 'E2'):
            delta[f'{chain}_minus_B0/{key}'] = [conditions[chain][i][key] - conditions['B0'][i][key] for i in range(n)]
    rng = random.Random(bootstrap_seed)
    samples = {name: [] for name in delta}
    for _ in range(replicates):
        indices = [rng.randrange(n) for _ in range(n)]
        for name, values in delta.items():
            samples[name].append(sum(values[i] for i in indices) / n)
    contrasts = {}
    for name, values in delta.items():
        contrasts[name] = dict(mean_difference=statistics.mean(values),
            CI95=[percentile(samples[name], .025), percentile(samples[name], .975)],
            n=n, wins=sum(x < 0 for x in values), losses=sum(x > 0 for x in values),
            ties=sum(x == 0 for x in values), lower_is_better=True)
    return dict(primary=contrasts.pop('mean_E_minus_B0/capped25'), secondary=contrasts,
                bootstrap_replicates=replicates, bootstrap_seed=bootstrap_seed,
                resampling_unit='paired held-out case; all arms and chains remain together',
                limitation='Conditional on three realized adaptation chains sharing eighteen training cases; not a training-distribution confidence interval.')



def condition_summary(rows):
    queries = sum(row['queries'] for row in rows)
    valid = sum(row['valid_queries'] for row in rows)
    return dict(n=len(rows), resolved=sum(row['resolved'] for row in rows),
        mean_capped25=statistics.mean(row['capped25'] for row in rows),
        mean_penalized26=statistics.mean(row['penalized26'] for row in rows),
        resolution_curve={str(turn):sum(row['first_unique_turn'] is not None and row['first_unique_turn'] <= turn for row in rows) for turn in range(1, 26)},
        actor_calls=queries, malformed_queries=sum(row['invalid'] for row in rows),
        malformed_rate=sum(row['invalid'] for row in rows)/queries,
        truncated_outputs=sum(row['truncated'] for row in rows), valid_queries=valid,
        mean_realized_information_bits_per_valid_query=sum(row['information_bits'] for row in rows)/valid if valid else None,
        input_tokens=sum(row['input_tokens'] for row in rows), output_tokens=sum(row['output_tokens'] for row in rows),
        mechanism_interpretation='Descriptive realized log2(before/after) among valid queries; not a causal conditional effect.')


