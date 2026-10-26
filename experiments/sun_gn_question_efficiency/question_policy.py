"""Public-only question monitoring and text-policy adaptation interfaces."""
from itertools import permutations
import json

UNIVERSE = tuple(''.join(d) for d in permutations('0123456789', 4))


def public_feedback(candidate, guess):
    same = sum(a == b for a, b in zip(candidate, guess))
    # Candidate secrets are distinct-digit; repeated guesses remain legal.
    return same, len(set(candidate) & set(guess)) - same


class PublicMonitor:
    def __init__(self):
        self.candidates = list(UNIVERSE)
        self.records = []

    def observe(self, guess, feedback):
        before = len(self.candidates)
        if guess is None:
            if feedback is not None:
                raise ValueError('Invalid query must carry no feedback')
        else:
            if len(guess) != 4 or any(c not in '0123456789' for c in guess):
                raise ValueError('Invalid query')
            if len(feedback) != 2 or any(type(x) is not int for x in feedback) or min(feedback)<0 or sum(feedback)>4:
                raise ValueError('Invalid feedback')
            self.candidates = [x for x in self.candidates if public_feedback(x, guess) == tuple(feedback)]
            if not self.candidates:
                raise ValueError('Public evidence inconsistent')
        row = dict(turn=len(self.records)+1, guess=guess, feedback=feedback,
                   before=before, after=len(self.candidates))
        self.records.append(row)
        return row

    @property
    def resolved(self):
        return len(self.candidates) == 1


def actor_system(assets, memory):
    # Empty memory is byte-for-byte the existing no-note condition.
    return assets.system('B0') if not memory else assets.system('E', memory)


def curator_messages(memory, episodes, template):
    summaries = []
    for episode in episodes:
        if set(episode) != {'public_history', 'resolved', 'capped25'}:
            raise ValueError('Curator receives only public training summaries')
        for row in episode['public_history']:
            if set(row) != {'turn','guess','feedback','before','after'}:
                raise ValueError('Non-public field in summary')
        summaries.append(episode)
    return [dict(role='user', content=template + '\nCURRENT NOTE:\n' + (memory or '[empty]')
                 + '\nPUBLIC TRAINING EPISODES:\n' + json.dumps(summaries, sort_keys=True, separators=(',', ':')))]


def validate_note(raw, finish_reason, limit=800):
    clean = raw.replace('\r\n','\n').replace('\r','\n').strip()
    if clean == 'KEEP' and finish_reason == 'stop':
        return dict(accepted=True, keep=True, normalized=None, reasons=[])
    reasons = []
    if finish_reason != 'stop': reasons.append('incomplete')
    if not clean or not 1 <= len(clean.splitlines()) <= 12: reasons.append('line_count_or_empty')
    if any(not x.startswith('- ') or not x[2:].strip() for x in clean.splitlines()): reasons.append('bullet_format')
    if any(ord(c)>126 or (ord(c)<32 and c!='\n') for c in clean): reasons.append('ascii')
    if len(clean.encode('utf-8')) > limit: reasons.append('byte_cap')
    if any(x in clean for x in ('`','|','://','www.')): reasons.append('non_note_markup')
    return dict(accepted=not reasons, keep=False, normalized=clean, reasons=reasons)


def adopt(previous, validation):
    return validation['normalized'] if validation['accepted'] and not validation['keep'] else previous
