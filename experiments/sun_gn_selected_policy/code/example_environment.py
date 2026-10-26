"""New public replication sample, NOT the study's restricted internal data.

This deterministic sample draws 332 distinct secrets from all 5,040 ordered
four-digit strings without repeated digits, including leading zeroes. It does
not use the private selection/exclusion procedure or any original case IDs.
Overlap with private study secrets is neither checked nor excluded. It has no
result or performance claim; it supplies a different data distribution and
identifier-dependent model seeds for a new mechanism replication.

Usage with the released protocol and a user-provided callable model adapter:

    repetitions, cohorts, feedback = make_environment()
    result = portable_search.run_experiment(
        protocol, repetitions, cohorts, call=my_model_adapter, feedback=feedback)

Load protocol.json separately. The adapter must honor the requested sampling
parameters, pin the declared model revision, and enforce context/resource caps.
This module itself makes no model or network calls. The environment owns labels;
only feedback strings reach the actor through portable_search.run_episode.
"""
from collections import Counter
import random

from question_policy import UNIVERSE


def number_feedback(secret, guess):
    """Original Counter-based feedback; repeated digits in guesses are legal."""
    if (type(secret) is not str or len(secret) != 4 or len(set(secret)) != 4
            or any(char not in '0123456789' for char in secret)
            or type(guess) is not str or len(guess) != 4
            or any(char not in '0123456789' for char in guess)):
        raise ValueError('Expected a legal secret and a four-digit guess')
    same = sum(a == b for a, b in zip(secret, guess))
    return same, sum((Counter(secret) & Counter(guess)).values()) - same


def make_environment(sample_seed=20261026):
    """Return (repetitions, cohorts, feedback) with fixed disjoint case lists.

    Same seed and Python sampling implementation reproduce this public sample.
    IDs disclose their public-sample origin and seed, never their secret value.
    """
    if type(sample_seed) is not int:
        raise ValueError('Sample seed must be an integer')
    values = random.Random(sample_seed).sample(UNIVERSE, 332)
    ids = [f'public-replication-{sample_seed}-{index:03d}' for index in range(332)]
    labels = dict(zip(ids, values))
    repetitions = {'0':dict(train=ids[:18], selection=ids[18:66]),
                   '1':dict(train=ids[66:84], selection=ids[84:132])}
    cohorts = {'0':ids[132:232], '1':ids[232:332]}

    def feedback(case_id, guess):
        return number_feedback(labels[case_id], guess)

    return repetitions, cohorts, feedback
