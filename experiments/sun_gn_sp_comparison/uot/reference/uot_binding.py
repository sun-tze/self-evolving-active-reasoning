"""Disabled UoT-only role adapter; no transport, tokenizer or process creation.

The injected, pinned QwenBoundary.call is inherited without alteration. This
module adds role classification and a fixed native single-episode invocation,
not execution authority, a controller, a scheduler or a scientific audit.
"""
from copy import deepcopy
import hashlib
import inspect
from pathlib import Path


SOURCE_PINS = {
    'qwen_binding.py': '22615e960b7baa6c84fb07934a4ed47acf0bab57b5fb6b36e648eeda0ebad5fd',
    'arbench_adapter.py': 'd63887f85329b871e9f7d169747c02a290be4e31e3292ac474469a0b3d8288d6',
}
METADATA_KEYS = frozenset({
    'task', 'method', 'call_index', 'temperature', 'top_p', 'json_format',
    'memory_injected', 'phase', 'upstream_caller',
})
ROLES = {
    '_run_uot_evaluation': ('uot_proposal', 512, 0.7, 0.7),
    '_gn_calculate_remaining_with_model': ('uot_simulation', 512, 0.1, 0.9),
}
MAX_TURNS = 25
SIMULATE_DEPTH = 1
EPISODE_CALL_BOUND = 1075  # 25 * (one proposal + three candidates * 14 partitions)
TWENTY_CASE_CALL_BOUND = 21500


def _source_pin(obj, name):
    source = inspect.getsourcefile(obj)
    if source is None or hashlib.sha256(Path(source).read_bytes()).hexdigest() != SOURCE_PINS[name]:
        raise ValueError('Injected source differs from pinned ' + name)


def classify_uot_metadata(metadata):
    """Classify pinned adapter metadata; never infer a role from sampling alone.

Caller names are provenance supplied by trusted Python, not an authentication
mechanism. The outer caller must use the pinned adapter and prevent arbitrary
model output from becoming metadata. call_index resets for each episode.
"""
    if not isinstance(metadata, dict) or set(metadata) != METADATA_KEYS:
        raise ValueError('Exact native UoT metadata schema required')
    if (metadata['task'] != 'GN' or metadata['method'] != 'uot' or
            metadata['phase'] != 'native_algorithm' or
            metadata['json_format'] is not False or metadata['memory_injected'] is not False or
            type(metadata['call_index']) is not int or
            not 1 <= metadata['call_index'] <= EPISODE_CALL_BOUND):
        raise ValueError('Only unmodified native UoT episode calls are supported')
    caller = metadata['upstream_caller']
    if not isinstance(caller, str) or caller not in ROLES:
        raise ValueError('Unrecognized upstream UoT caller')
    role = ROLES[caller]
    for field, wanted in (('temperature', role[2]), ('top_p', role[3])):
        if type(metadata[field]) not in (int, float) or metadata[field] != wanted:
            raise ValueError('Caller and requested UoT sampling disagree')
    return role


def make_uot_boundary_class(base_boundary):
    """Inject the original QwenBoundary class, with its safety method unchanged."""
    if not isinstance(base_boundary, type) or base_boundary.__name__ != 'QwenBoundary':
        raise TypeError('The original QwenBoundary class is required')
    _source_pin(base_boundary, 'qwen_binding.py')

    class UoTBoundary(base_boundary):
        uot_source_pins = dict(SOURCE_PINS)

        def __init__(self, *, execution_enabled=False, **kwargs):
            if type(execution_enabled) is not bool:
                raise TypeError('Explicit Boolean execution switch required')
            # This switch is a prototype lock, not owner or allocation authority.
            self.uot_execution_enabled = execution_enabled
            protocol, plan = kwargs['protocol'], kwargs['plan']
            if (protocol.get('actor') != {'max_tokens': 512, 'temperature': 0.7, 'top_p': 0.7} or
                    type(plan.get('max_calls')) is not int or
                    not 1 <= plan['max_calls'] <= TWENTY_CASE_CALL_BOUND or
                    type(plan.get('max_output_tokens')) is not int or
                    not 1 <= plan['max_output_tokens'] <= TWENTY_CASE_CALL_BOUND * 512):
                raise ValueError('Finite UoT proposal/call/output configuration required')
            super().__init__(**kwargs)

        def _role(self, metadata):
            if not self.uot_execution_enabled:
                raise RuntimeError('UoT prototype execution is disabled')
            return classify_uot_metadata(metadata)

    # There is no overriding call: context, request/token reservations, admission
    # margin, journaling, response integrity and permanent halt remain upstream.
    assert UoTBoundary.call is base_boundary.call
    return UoTBoundary


def run_native_uot_episode(*, adapter, boundary, secret, seed, case_id=None,
                           execution_enabled=False):
    """One native depth-one episode. Defaults disabled; no batch/job machinery.

No memory injection, parser overlay or extra matched-final actor is introduced.
Outer runtime must separately establish actual authorization, source/config
identity, finite allocation deadlines, real token counting and no HTTP retries.
"""
    if execution_enabled is not True or type(execution_enabled) is not bool:
        raise RuntimeError('UoT prototype execution is disabled')
    _source_pin(adapter, 'arbench_adapter.py')
    if (getattr(boundary, 'uot_source_pins', None) != SOURCE_PINS or
            getattr(boundary, 'uot_execution_enabled', None) is not True):
        raise ValueError('Enabled pinned UoT boundary required')
    # Real call implementation and upstream adapter own their original checks.
    result = adapter.run_gn_episode(
        method='uot', secret=secret, chat=boundary.call, seed=seed,
        limits=adapter.CallLimits(EPISODE_CALL_BOUND, 512), max_turn=MAX_TURNS,
        temperature=0.7, top_p=0.7, simulate_depth=SIMULATE_DEPTH,
        memory_context='', case_id=case_id, repair_mode=None,
        matched_finalization=False)
    return result
