"""Independently verify public paired-case arithmetic with the standard library.

Example: python verify_results.py --cases anonymous-cases.json --results results.json
No experiment, producer, audit, model, tokenizer, network or optional package is
imported. This verifies released arithmetic, not private execution provenance.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import time

SCHEMA = 'gn-selected-policy-public-v1'
ARMS = ('B0', 'E0', 'R0', 'E1', 'R1')
CONTRASTS = ('E0-B0', 'R0-B0', 'E1-B0', 'R1-B0', 'Emean-B0', 'Emean-Rmean')
PRIMARY = ('Emean-B0', 'Emean-Rmean')
REPLICATES = 20000


def require(condition, message):
    if not condition:
        raise ValueError(message)


def decode(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'Duplicate JSON key: '+key)
            result[key] = value
        return result
    def reject(value):
        raise ValueError('Nonfinite JSON constant')
    return json.loads(raw, object_pairs_hook=unique, parse_constant=reject)


def integer(x, low, high):
    return type(x) is int and low <= x <= high


def validate_cases(document):
    require(type(document) is dict and set(document) == {'schema','N','cases'}, 'Anonymous document schema')
    require(document['schema'] == SCHEMA and type(document['N']) is int and document['N'] == 200, 'Anonymous denominator')
    rows = document['cases']
    require(type(rows) is list and len(rows) == 200, 'Require exactly 200 paired cases')
    for ordinal, case in enumerate(rows):
        require(type(case) is dict and set(case) == {'case_index','cohort','index','arms'}, 'Anonymous case schema')
        for field, expected in (('case_index',ordinal),('cohort',ordinal//100),('index',ordinal%100)):
            require(type(case[field]) is int and case[field] == expected, 'Case order/pairing: '+field)
        require(type(case['arms']) is dict and set(case['arms']) == set(ARMS), 'Missing or extra paired arm')
        for arm in ARMS:
            r = case['arms'][arm]
            require(type(r) is dict and set(r) == {'capped25','first_unique_turn','invalid','queries','resolved','truncated'}, 'Outcome schema')
            turn = r['first_unique_turn']
            require(turn is None or integer(turn,1,25), 'First singleton domain')
            require(type(r['resolved']) is bool and r['resolved'] == (turn is not None), 'Resolution/turn inconsistency')
            cost = 25 if turn is None else turn
            require(integer(r['capped25'],1,25) and r['capped25'] == cost, 'Capped endpoint inconsistency')
            require(integer(r['queries'],1,25) and r['queries'] == cost, 'Stopping/query inconsistency')
            require(integer(r['invalid'],0,cost) and integer(r['truncated'],0,cost), 'Malformed/truncated count domain')
    return rows


def arm_summary(cases):
    result = {}
    for arm in ARMS:
        rows = [case['arms'][arm] for case in cases]
        n = len(rows)
        queries = sum(row['queries'] for row in rows)
        invalid = sum(row['invalid'] for row in rows)
        resolved = sum(row['resolved'] for row in rows)
        result[arm] = dict(n=n, mean_capped25=sum(row['capped25'] for row in rows)/n,
            resolved=resolved, resolution_rate=resolved/n,
            resolution_curve={str(t):sum(row['first_unique_turn'] is not None and row['first_unique_turn'] <= t for row in rows) for t in range(1,26)},
            queries=queries, invalid=invalid, malformed_rate=invalid/queries,
            truncated=sum(row['truncated'] for row in rows))
    return result


def doubled_contrasts(case):
    """Integer numerator for each difference, with common denominator two."""
    b, e0, r0, e1, r1 = (case['arms'][arm]['capped25'] for arm in ARMS)
    return (2*(e0-b), 2*(r0-b), 2*(e1-b), 2*(r1-b), e0+e1-2*b, e0+e1-r0-r1)


def resample(strata, replicates, seed):
    """One index selects all arms; each stratum retains its original size."""
    require(type(replicates) is int and replicates > 0, 'Invalid replicate count')
    require(strata and all(len(s) > 0 for s in strata), 'Empty resampling stratum')
    rng = random.Random(seed)
    distributions = [[] for _ in CONTRASTS]
    denominator = 2*sum(len(s) for s in strata)
    for _ in range(replicates):
        totals = [0]*6
        for stratum in strata:
            size = len(stratum)
            for _ in range(size):
                selected = stratum[rng.randrange(size)]
                for j in range(6):
                    totals[j] += selected[j]
        for j in range(6):
            distributions[j].append(totals[j]/denominator)
    return [sorted(values) for values in distributions]


def quantile(sorted_values, probability):
    position = (len(sorted_values)-1)*probability
    left = int(position)
    right = min(left+1,len(sorted_values)-1)
    return sorted_values[left]+(sorted_values[right]-sorted_values[left])*(position-left)


def contrast_summary(strata, seed, primary, replicates=REPLICATES):
    n = sum(len(s) for s in strata)
    samples = resample(strata,replicates,seed)
    result = {}
    for j, name in enumerate(CONTRASTS):
        point = sum(row[j] for s in strata for row in s)/(2*n)
        result[name] = dict(n=n,mean_difference=point,
            CI95_descriptive=[quantile(samples[j],.025),quantile(samples[j],.975)])
        if primary and name in PRIMARY:
            result[name]['CI98_75_primary'] = [quantile(samples[j],.00625),quantile(samples[j],.99375)]
    return result


def derive_flags(cohorts, resolved):
    cs = [cohorts[str(c)]['contrasts'] for c in range(2)]
    criteria = dict(
        both_cohort_Emean_B0_primary_upper_below_zero=all(c['Emean-B0']['CI98_75_primary'][1] < 0 for c in cs),
        each_E_run_negative_point_in_each_cohort=all(c[a+'-B0']['mean_difference'] < 0 for c in cs for a in ('E0','E1')),
        each_cohort_observed_mean_gain_at_least_two=all(c['Emean-B0']['mean_difference'] <= -2 for c in cs),
        each_E_pooled_coverage_not_below_B0=all(resolved[a] >= resolved['B0'] for a in ('E0','E1')))
    benefit = all(criteria.values())
    trace = all(c['Emean-Rmean']['CI98_75_primary'][1] < 0 for c in cs)
    return dict(repeated_benefit_criteria=criteria,repeated_observed_practical_benefit=benefit,
        both_cohort_trace_value_primary_upper_below_zero=trace,repeated_benefit_and_trace_value=benefit and trace,
        both_cohort_Emean_B0_primary_upper_below_minus_two=all(c['Emean-B0']['CI98_75_primary'][1] < -2 for c in cs))


def recompute(cases):
    strata = [[doubled_contrasts(row) for row in cases[c*100:(c+1)*100]] for c in range(2)]
    summaries = dict(cohorts={str(c):arm_summary(cases[c*100:(c+1)*100]) for c in range(2)},pooled=arm_summary(cases))
    cohorts = {}
    for c in range(2):
        cohorts[str(c)] = dict(n=100,seed=63026+c,
            contrasts=contrast_summary([strata[c]],63026+c,True),
            arms={a:{k:summaries['cohorts'][str(c)][a][k] for k in ('n','mean_capped25','resolved','resolution_rate')} for a in ARMS})
    resolved = {a:summaries['pooled'][a]['resolved'] for a in ARMS}
    stats = dict(bootstrap_replicates=REPLICATES,cohort_results=cohorts,
        pooled=dict(n=200,seed=63028,method='Stratified paired case blocks; descriptive only',
                    contrasts=contrast_summary(strata,63028,False),resolved_by_arm=resolved),
        primary_family=dict(contrasts=4,interval_level=.9875,
                    method='Central percentile; approximate current-study Bonferroni family 95%'),
        **derive_flags(cohorts,resolved))
    return summaries, stats


def compare_exact(actual, expected, location, differences):
    """No numeric tolerance: independently reproduced public numbers must match."""
    if type(expected) is dict:
        if type(actual) is not dict or set(actual) != set(expected):
            differences.append(location+': object keys differ'); return
        for key in expected:
            compare_exact(actual[key],expected[key],location+'.'+key,differences)
    elif type(expected) is list:
        if type(actual) is not list or len(actual) != len(expected):
            differences.append(location+': list length/type differs'); return
        for i, value in enumerate(expected):
            compare_exact(actual[i],value,location+'['+str(i)+']',differences)
    elif type(actual) is not type(expected) or actual != expected:
        differences.append(location+': expected '+repr(expected)+', received '+repr(actual))


def verify(document, released):
    cases = validate_cases(document)
    require(type(released) is dict and released.get('schema') == SCHEMA, 'Released results schema')
    require(type(released.get('N')) is int and released['N'] == 200, 'Released denominator')
    summaries, stats = recompute(cases)
    differences = []
    compare_exact(released.get('arm_summaries'),summaries,'arm_summaries',differences)
    compare_exact(released.get('statistics'),stats,'statistics',differences)
    resources = released.get('resources',{})
    require(type(resources.get('evaluation_episodes')) is int and resources['evaluation_episodes'] == 1000, 'Released evaluation episode count')
    return dict(status='PASS' if not differences else 'FAIL',differences=differences,
        numeric_comparison='Exact parsed numeric values and types; no tolerance',
        verified=dict(paired_cases=200,cohorts=2,cases_per_cohort=100,arms_per_case=5,evaluation_episodes=1000,
                      primary_contrasts=4,bootstrap_replicates_per_analysis=20000,bootstrap_analyses=3,
                      interval_pairs=22,interval_endpoints=44),
        arm_summaries=summaries,statistics=stats,
        scope='Independent standard-library recomputation from anonymous paired rows. This does not verify private execution provenance, source-audit PASS, training lineage, GPU accounting or the published learning counts.',
        interpretation='The four primary intervals approximate current-study family coverage, not historical multiple-testing control. Pooled intervals are descriptive. Point reduction of two queries is not proof of a true effect of at least two; pooled coverage does not imply each-cohort coverage.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases',type=Path,default=Path('anonymous-cases.json'))
    parser.add_argument('--results',type=Path,default=Path('results.json'))
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    started=time.monotonic()
    case_raw=args.cases.read_bytes(); result_raw=args.results.read_bytes()
    report=verify(decode(case_raw),decode(result_raw))
    report['input_sha256']={'anonymous_cases':hashlib.sha256(case_raw).hexdigest(),'released_results':hashlib.sha256(result_raw).hexdigest()}
    report['verifier_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report['elapsed_seconds']=time.monotonic()-started
    text=json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+'\n'
    if args.output:
        with args.output.open('x',encoding='utf8',newline='\n') as handle:handle.write(text)
        print(json.dumps({k:report[k] for k in ('status','differences','verified','elapsed_seconds')}))
    else:print(text,end='')
    return 0 if report['status']=='PASS' else 1


if __name__=='__main__':
    raise SystemExit(main())
