"""Recompute published paired statistics from anonymous, aggregate case rows.

Usage: python verify_results.py anonymous-cases.json
This verifies arithmetic and pairing, not the provenance of experimental data.
"""
import argparse
import json
import math

from paired_statistics import condition_summary, paired_statistics, require


SCHEMA = 'gn-question-efficiency-public-v1'
ARMS = ('B0', 'S', 'E0', 'E1', 'E2')
FIELDS = {'first_unique_turn', 'queries', 'invalid', 'truncated', 'valid_queries',
          'information_bits', 'input_tokens', 'output_tokens'}


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    def reject_constant(value):
        raise ValueError('Nonfinite JSON number: '+value)
    with open(path, encoding='utf8') as handle:
        return json.load(handle, object_pairs_hook=unique, parse_constant=reject_constant)


def anonymous_conditions(document):
    require(set(document) == {'schema', 'N', 'cases', 'expected'}, 'Public document fields')
    require(document['schema'] == SCHEMA and type(document['N']) is int and document['N'] == 100, 'Public schema/N')
    require(type(document['cases']) is list and len(document['cases']) == document['N'], 'Complete case coverage')
    conditions = {arm:[] for arm in ARMS}
    for index, case in enumerate(document['cases']):
        require(set(case) == {'case_index', 'arms'} and type(case['case_index']) is int and case['case_index'] == index, 'Anonymous row order')
        require(set(case['arms']) == set(ARMS), 'Arm coverage')
        for arm in ARMS:
            row = case['arms'][arm]
            require(set(row) == FIELDS, 'Unexpected public row field')
            turn = row['first_unique_turn']
            require(turn is None or (type(turn) is int and 1 <= turn <= 25), 'First singleton turn')
            resolved = turn is not None
            for key in ('queries', 'invalid', 'truncated', 'valid_queries', 'input_tokens', 'output_tokens'):
                require(type(row[key]) is int and row[key] >= 0, 'Integer count: '+key)
            require(row['queries'] == (turn if resolved else 25), 'First-singleton stopping rule')
            require(row['invalid'] <= row['queries'] and row['truncated'] <= row['queries'], 'Invalid/truncated counts')
            require(row['valid_queries'] == row['queries']-row['invalid'], 'Valid query count')
            require(row['input_tokens'] <= 8064*row['queries'] and row['output_tokens'] <= 128*row['queries'], 'Actor token caps')
            bits = row['information_bits']
            require(type(bits) in (int, float) and math.isfinite(bits) and bits >= 0, 'Information gain')
            require(bits <= math.log2(5040)+1e-9, 'Information exceeds full hypothesis space')
            if resolved:
                require(row['valid_queries'] > 0 and math.isclose(bits, math.log2(5040), abs_tol=1e-9), 'Singleton information total')
            else:
                require(bits <= math.log2(2520)+1e-9, 'Unresolved information total')
            if row['valid_queries'] == 0:
                require(bits == 0, 'Malformed queries cannot add information')
            conditions[arm].append(dict(row, case_id=f'case-{index:03d}', resolved=resolved,
                                        capped25=turn if resolved else 25, penalized26=turn if resolved else 26))
    return conditions


def compare(expected, actual, location='expected'):
    if isinstance(actual, dict):
        require(isinstance(expected, dict) and set(expected) == set(actual), 'Result keys: '+location)
        for key in actual:
            compare(expected[key], actual[key], location+'/'+key)
    elif isinstance(actual, list):
        require(isinstance(expected, list) and len(expected) == len(actual), 'Result list: '+location)
        for index, (a, b) in enumerate(zip(expected, actual)):
            compare(a, b, location+'/'+str(index))
    elif type(actual) in (int, float):
        require(type(expected) in (int, float) and math.isfinite(expected)
                and math.isclose(expected, actual, rel_tol=1e-12, abs_tol=1e-12), 'Result number: '+location)
    else:
        require(type(expected) is type(actual) and expected == actual, 'Result value: '+location)


def verify(document):
    conditions = anonymous_conditions(document)
    computed = dict(statistics=paired_statistics(conditions, replicates=10000, bootstrap_seed=62026),
                    conditions={arm:condition_summary(rows) for arm, rows in conditions.items()})
    compare(document['expected'], computed)
    return dict(arithmetic_verified=True, schema=SCHEMA, N=document['N'], computed=computed,
                scope='Arithmetic, anonymous pairing, endpoint consistency, and case-block resampling only; no execution-provenance claim.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('anonymous_cases', help='Anonymous JSON matching PUBLIC-SCHEMA.json')
    args = parser.parse_args()
    print(json.dumps(verify(read_json(args.anonymous_cases)), indent=2, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
