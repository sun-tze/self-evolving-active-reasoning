"""Pure selection-set gate; preparation only, not experiment authorization.

Input schema
------------
select_policy accepts incumbent_note, challenger_raw, finish_reason and, only
for a mechanically eligible challenger, two complete lists of episode rows.
Each row has exactly: selection_index, first_unique_turn, capped25, resolved,
queries. Each list must cover integer selection_index 0..47 once. It accepts
no labels, original case IDs, evaluation fields, or held-out results.

Output schema
-------------
Every returned decision includes schema, selected_policy, selected_note,
promoted, reason, failed_criteria, mechanical_validation, hashes, pairing,
metrics, and rule. hashes bind both input notes, the normalized challenger,
the selected note, and canonical sorted selection rows when evaluated.
pairing records expected count, whether scoring occurred, completeness,
indices and per-index challenger-minus-incumbent capped25 differences.
metrics is null when mechanical screening prevents evaluation. Invalid or
missing episode data raises ValueError: it never becomes an unresolved case.

The gate requires at least one fewer mean capped query over all 48 pairs AND
no decrease in resolved-by-25 count. It uses integer totals, retains the
incumbent on ties, and defines no held-out statistical success threshold.
"""
import hashlib
import json


SCHEMA = 'paired-selection-policy-v1'
SELECTION_N = 48
NOTE_BYTES = 800
EPISODE_FIELDS = {'selection_index', 'first_unique_turn', 'capped25', 'resolved', 'queries'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def text_hash(text):
    return hashlib.sha256(text.encode('utf8')).hexdigest()


def validate_note(raw, finish_reason):
    require(isinstance(raw, str) and isinstance(finish_reason, str), 'Note and finish reason must be strings')
    clean = raw.replace('\r\n', '\n').replace('\r', '\n').strip()
    if clean == 'KEEP' and finish_reason == 'stop':
        return dict(accepted=True, keep=True, normalized=None, reasons=[])
    reasons = []
    if finish_reason != 'stop': reasons.append('incomplete')
    lines = clean.splitlines()
    if not clean or not 1 <= len(lines) <= 12: reasons.append('line_count_or_empty')
    if any(not line.startswith('- ') or not line[2:].strip() for line in lines): reasons.append('bullet_format')
    if any(ord(char) > 126 or (ord(char) < 32 and char != '\n') for char in clean): reasons.append('ascii')
    if len(clean.encode('utf8')) > NOTE_BYTES: reasons.append('byte_cap')
    if any(marker in clean for marker in ('`', '|', '://', 'www.')): reasons.append('non_note_markup')
    return dict(accepted=not reasons, keep=False, normalized=clean, reasons=reasons)


def validate_selection_episodes(rows):
    """Return new canonical rows sorted by index; never fill missing outcomes."""
    require(isinstance(rows, list) and len(rows) == SELECTION_N, 'Exactly 48 selection episodes required')
    by_index = {}
    for row in rows:
        require(isinstance(row, dict) and set(row) == EPISODE_FIELDS, 'Unexpected selection episode field')
        index, turn = row['selection_index'], row['first_unique_turn']
        require(type(index) is int and 0 <= index < SELECTION_N, 'Selection index must be integer 0..47')
        require(index not in by_index, 'Duplicate selection index')
        require(turn is None or (type(turn) is int and 1 <= turn <= 25), 'Invalid first_unique_turn')
        require(type(row['resolved']) is bool and row['resolved'] == (turn is not None), 'Resolved/first turn mismatch')
        cost = turn if turn is not None else 25
        require(type(row['capped25']) is int and row['capped25'] == cost, 'Capped endpoint mismatch')
        require(type(row['queries']) is int and row['queries'] == cost, 'Query count/stopping mismatch')
        by_index[index] = {name:row[name] for name in EPISODE_FIELDS}
    require(set(by_index) == set(range(SELECTION_N)), 'Missing selection index')
    return [by_index[index] for index in range(SELECTION_N)]


def rows_hash(rows):
    encoded = json.dumps(rows, sort_keys=True, ensure_ascii=True, separators=(',', ':'), allow_nan=False)
    return text_hash(encoded)


def select_policy(*, incumbent_note, challenger_raw, finish_reason,
                  incumbent_episodes=None, challenger_episodes=None):
    """Return an auditable selection decision; perform no I/O or generation.

    Mechanical rejects, KEEP and byte-identical normalized candidates must be
    passed without outcomes. An eligible challenger requires both complete
    selection lists. This separates mechanical screening from outcome choice.
    """
    require(isinstance(incumbent_note, str), 'Incumbent note must be a string')
    if incumbent_note:
        current = validate_note(incumbent_note, 'stop')
        require(current['accepted'] and not current['keep'], 'Invalid incumbent note')
    validation = validate_note(challenger_raw, finish_reason)
    candidate = validation['normalized']
    decision = dict(schema=SCHEMA, selected_policy='incumbent', selected_note=incumbent_note,
        promoted=False, reason=None, failed_criteria=[], mechanical_validation=validation,
        hashes=dict(incumbent_note_sha256=text_hash(incumbent_note),
                    challenger_raw_sha256=text_hash(challenger_raw),
                    challenger_normalized_sha256=text_hash(candidate) if candidate is not None else None,
                    selected_note_sha256=text_hash(incumbent_note),
                    incumbent_selection_sha256=None, challenger_selection_sha256=None),
        pairing=dict(expected_n=SELECTION_N, evaluated=False, complete=None,
                     selection_indices=[], capped25_paired_differences=[]), metrics=None,
        rule=dict(minimum_mean_query_reduction=1, resolution_count_must_not_decrease=True,
                  endpoint='all-case capped first-singleton time; unresolved=25',
                  tie_policy='retain incumbent', scope='selection set only; no held-out success claim'))
    if not validation['accepted']:
        decision['reason'] = 'mechanically_invalid'
    elif validation['keep']:
        decision['reason'] = 'keep_requested'
    elif candidate.encode('utf8') == incumbent_note.encode('utf8'):
        decision['reason'] = 'byte_identical'
    if decision['reason'] is not None:
        require(incumbent_episodes is None and challenger_episodes is None,
                'Mechanically ineligible challenger must not receive episode outcomes')
        return decision

    incumbent = validate_selection_episodes(incumbent_episodes)
    challenger = validate_selection_episodes(challenger_episodes)
    indices = [row['selection_index'] for row in incumbent]
    require(indices == [row['selection_index'] for row in challenger], 'Selection pairing mismatch')
    differences = [b['capped25']-a['capped25'] for a, b in zip(incumbent, challenger)]
    incumbent_total = sum(row['capped25'] for row in incumbent)
    challenger_total = sum(row['capped25'] for row in challenger)
    incumbent_resolved = sum(row['resolved'] for row in incumbent)
    challenger_resolved = sum(row['resolved'] for row in challenger)
    mean_gain_pass = sum(differences) <= -SELECTION_N
    coverage_pass = challenger_resolved >= incumbent_resolved
    decision['pairing'] = dict(expected_n=SELECTION_N, evaluated=True, complete=True,
                              selection_indices=indices, capped25_paired_differences=differences)
    decision['hashes']['incumbent_selection_sha256'] = rows_hash(incumbent)
    decision['hashes']['challenger_selection_sha256'] = rows_hash(challenger)
    decision['metrics'] = dict(paired_n=SELECTION_N, incumbent_capped25_total=incumbent_total,
        challenger_capped25_total=challenger_total, incumbent_mean_capped25=incumbent_total/SELECTION_N,
        challenger_mean_capped25=challenger_total/SELECTION_N,
        paired_difference_total=sum(differences), mean_paired_difference=sum(differences)/SELECTION_N,
        incumbent_resolved25_count=incumbent_resolved, challenger_resolved25_count=challenger_resolved,
        mean_gain_pass=mean_gain_pass, coverage_pass=coverage_pass, equal_mean=sum(differences) == 0)
    if not mean_gain_pass: decision['failed_criteria'].append('mean_reduction_less_than_one_query')
    if not coverage_pass: decision['failed_criteria'].append('resolved25_count_decreased')
    if mean_gain_pass and coverage_pass:
        decision.update(promoted=True, selected_policy='challenger', selected_note=candidate,
                        reason='mean_gain_and_resolution_pass')
        decision['hashes']['selected_note_sha256'] = text_hash(candidate)
    else:
        decision['reason'] = 'selection_criteria_not_met'
    return decision
