"""File-bound UoT controller; no native runtime import before explicit admission.

Missing/disabled scope fails closed. Root owns the real human authorization and
submission/history audit. load_frozen only checks their supplied file bindings;
it cannot authenticate the human or create a stage release. No scheduler here.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import sys
import time


ENTRY_DIR = 'uot-baseline-preview-01'
JOB_NAME = 'gn-uot-depth1-baseline-01'
SECONDS = 7200
PORT = 18000
SOURCE_PINS = {
    'gn-runtime/controller_entry.py': 'cc2f479a8a4922453bd0e5e8883bebd0694943825d2e49a8c5a815f4751d13b5',
    'gn-runtime/launch_binding.py': '4cbf28cef25c30b4af9598360613b06d44922c3bf737b59de63d75f7588cbca4',
    'gn-runtime/qwen_binding.py': '22615e960b7baa6c84fb07934a4ed47acf0bab57b5fb6b36e648eeda0ebad5fd',
    'baseline-adapters/arbench_adapter.py': 'd63887f85329b871e9f7d169747c02a290be4e31e3292ac474469a0b3d8288d6',
    ENTRY_DIR + '/uot_binding.py': 'f92b9c665754be7fafb55ff892bae87df9f2e2f0e820978b3797259fcfc5b072',
    ENTRY_DIR + '/uot_batch.py': 'a2eaa4f244094069154a1bc8307cb39026d29a57b8a62fe7be8888fb6652c26b',
    'protocol.json': 'ad6dcbe18d3c58bbbf480993a63fcfa6d82cc5e5b0db4f9a3a4fa0ab8d06e682',
    'private-cases.json': '27c4550598fc7d306b90a4a087ebcf7e90b34876aae572e85827073145c6e312',
    'vendor/arbench/arbench/reasoner/gn/gn_evaluator.py': 'aa6643d004ed7b510452ce9fe581b371112616a6702dfee3a26132759cb221a9',
    'vendor/arbench/arbench/reasoner/gn/prompt.py': '5dafd196d5542fad163898fbfecca411fb13b1b30b636ebc0577d38a7bbcc713',
    'vendor/arbench/arbench/utils/utils_gn.py': 'f57cc0a77615859ea44fad07a0e5b849aa95e6d428963d72d3c49405c20274e7',
    'vendor/arbench/arbench/utils/utils_sp.py': '03c29648f308b697767009504ea3f9dd01009883e96a33c7e7707aea30125785',
}
REQUIRED = set(SOURCE_PINS) | {'config.json', 'submit_uot.py', 'uot.sbatch', 'local-checks.json', 'history-inventory.json',
    ENTRY_DIR + '/uot_controller_entry.py', ENTRY_DIR + '/uot_allocation_entry.py'}
FIXED_SCOPE = {
    'format': 'gn_uot_execution_scope_v1', 'enabled': True, 'job_name': JOB_NAME,
    'allocation_seconds': SECONDS, 'max_new_jobs': 1,
    'prior_jobs': 13, 'prior_conservative_seconds': 15470,
    'job_cap': 14, 'total_GPU_seconds': 22670,
    'nodes': 1, 'gpus': 1, 'cpus': 8, 'memory_GB': 32, 'API_USD': 0,
    'max_calls': 21500, 'max_output_reserved': 11008000,
    # This entry never runs an audit. Future root management must spend the
    # 600 CPU replay seconds IN AGGREGATE, at most three invocations, with up to
    # five kill-grace seconds each; stop at first PASS. Only mechanical auditor
    # repairs may consume another invocation, preserving sources/raw versions;
    # a scientific, source or data mismatch stops instead of being repaired away.
    'postrun_audit_seconds': 600, 'postrun_audit_kill_grace_seconds': 5,
    'postrun_audit_max_invocations': 3,
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    def unique(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError('Duplicate JSON key')
            out[key] = value
        return out
    return json.loads(Path(path).read_bytes(), object_pairs_hook=unique,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def source_relative(base, name):
    if (not isinstance(name, str) or not name or '\\' in name or ':' in name or
            PurePosixPath(name).is_absolute() or any(x in ('', '.', '..') for x in name.split('/'))):
        raise ValueError('Unsafe package-relative path')
    path = (Path(base) / name).resolve()
    if not path.is_relative_to(Path(base).resolve()):
        raise ValueError('Package path escapes its root')
    return path


def validate_scope(scope):
    if not isinstance(scope, dict) or any(type(scope.get(k)) is not type(v) or scope.get(k) != v
                                          for k, v in FIXED_SCOPE.items()):
        raise ValueError('Enabled exact UoT scope required; absent or disabled is not admission')
    if type(scope.get('max_input_reserved')) is not int or not 0 < scope['max_input_reserved'] <= 165120000:
        raise ValueError('Finite input reservation exceeds 21500 * (8192 - 512)')
    for field in ('manifest_sha256', 'owner_authorization_sha256'):
        if not isinstance(scope.get(field), str) or not re.fullmatch('[0-9a-f]{64}', scope[field]):
            raise ValueError('Actual manifest and owner record hashes required')
    owner = scope.get('owner_authorization_record')
    if not isinstance(owner, str) or not owner:
        raise ValueError('Bound new owner authorization record required')
    return scope


def load_module(base, relative, name):
    """Use only already-manifest-verified source, with fixed critical pins."""
    path = source_relative(base, relative)
    if relative in SOURCE_PINS and sha(path) != SOURCE_PINS[relative]:
        raise ValueError('Pinned reused source changed: ' + relative)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_frozen(base):
    base = Path(base)
    scope = validate_scope(read_json(base / 'execution-scope.json'))
    if sha(base / 'package-manifest.json') != scope['manifest_sha256']:
        raise ValueError('Package manifest binding changed')
    manifest = read_json(base / 'package-manifest.json')
    files = manifest.get('files')
    if not isinstance(files, dict) or not REQUIRED <= files.keys():
        raise ValueError('Complete UoT package source map required')
    if {'execution-scope.json', 'package-manifest.json'} & files.keys():
        raise ValueError('Circular package manifest is forbidden')
    for name, digest in files.items():
        if (not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest) or
                sha(source_relative(base, name)) != digest):
            raise ValueError('Package bytes changed: ' + str(name))
    if any(files.get(name) != digest for name, digest in SOURCE_PINS.items()):
        raise ValueError('Pinned source/reference differs from the reviewed package')
    owner = scope['owner_authorization_record']
    if files.get(owner) != scope['owner_authorization_sha256']:
        raise ValueError('New owner record is not package-bound')
    owner_record = read_json(source_relative(base, owner))
    if not isinstance(owner_record, dict) or owner_record.get('approved') is not True:
        raise ValueError('Actual affirmative owner record required; technical review is not approval')
    for directory in ('gn-runtime', 'baseline-adapters', 'vendor', ENTRY_DIR):
        for path in (base / directory).rglob('*.py'):
            if path.relative_to(base).as_posix() not in files:
                raise ValueError('Unbound executable sibling')
    config, protocol, cases = (read_json(base / name) for name in
                               ('config.json', 'protocol.json', 'private-cases.json'))
    original = load_module(base, 'gn-runtime/controller_entry.py', '_uot_reused_controls')
    # Reuse only path/source/environment identity checks, never its old scope,
    # 6000-second context, old comparison plan or old completion gate.
    original._verify_reused(config)
    batch = load_module(base, ENTRY_DIR + '/uot_batch.py', '_uot_frozen_batch_check')
    batch.validate_reference(protocol, cases)
    return config, protocol, cases, scope


def binding_hashes(base):
    return {name: sha(Path(base) / name) for name in
            ('execution-scope.json', 'package-manifest.json', 'config.json',
             'protocol.json', 'private-cases.json')}


def allocation_context(scope, environment, *, platform, now):
    validate_scope(scope)
    job, started = environment.get('SLURM_JOB_ID', ''), environment.get('SLURM_JOB_START_TIME', '')
    if (platform != 'linux' or environment.get('SLURM_JOB_NAME') != JOB_NAME or
            environment.get('SLURM_JOB_NUM_NODES') != '1' or environment.get('SLURM_GPUS_ON_NODE') != '1' or
            not job.isdigit() or not re.fullmatch(r'[0-9]+(?:\.[0-9]+)?', started)):
        raise ValueError('Actual Slurm allocation identity differs')
    start = float(started)
    if (type(now) not in (int, float) or not math.isfinite(now) or not math.isfinite(start) or
            not start <= now < start + SECONDS - 240):
        raise ValueError('No valid allocation request window remains')
    return dict(job_id=job, job_name=JOB_NAME, start=start,
                request_deadline=start + SECONDS - 240,
                controller_deadline=start + SECONDS - 120,
                cleanup_deadline=start + SECONDS - 2, hard_deadline=start + SECONDS)


def check_completion(output, protocol):
    """Coverage/byte gate only; independent native replay remains a later audit."""
    output = Path(output)
    result, plan = read_json(output / 'result.json'), read_json(output / 'plan.json')
    if (result.get('format') != 'gn_uot_batch_result_v1' or result.get('status') != 'COMPLETE_NATIVE_UOT_BATCH' or
            result.get('failure') is not None or result.get('boundary_halted') is not False or
            result.get('scientific_success_claimed') is not False or
            any(type(result.get(k)) is not int or result[k] != 20 for k in ('expected_count', 'recorded_count', 'completed_count')) or
            result.get('plan_sha256') != sha(output / 'plan.json')):
        raise ValueError('UoT schedule is incomplete or its result binding changed')
    if (plan.get('format') != 'gn_uot_batch_plan_v1' or plan.get('max_calls') != 21500 or
            plan.get('max_output_tokens') != 11008000 or plan.get('simulate_depth') != 1 or
            plan.get('max_turn') != 25 or plan.get('max_tokens') != 512):
        raise ValueError('Native UoT plan changed')
    if (result['reference_protocol_sha256'] != sha(output / 'reference-protocol.json') or
            result['reference_cases_sha256'] != sha(output / 'private-cases.json')):
        raise ValueError('Batch reference copies changed')
    batch = load_module(Path(__file__).parent.parent, ENTRY_DIR + '/uot_batch.py', '_uot_completion_batch')
    _, _, expected = batch.validate_reference(protocol, read_json(output / 'private-cases.json'))
    records, previous = result['records'], 0
    if len(records) != 20 or plan['schedule'] != expected:
        raise ValueError('Missing or reordered UoT case schedule')
    for row, identity in zip(records, expected):
        name = 'episodes/' + str(identity['index']).zfill(5) + '.json'
        end = row['call_end_inclusive']
        if (any(row.get(k) != v or type(row.get(k)) is not type(v) for k, v in identity.items()) or
                row.get('status') != 'complete' or row.get('file') != name or
                row.get('call_start_exclusive') != previous or type(end) is not int or
                not previous < end <= previous + 1075 or row.get('sha256') != sha(output / name)):
            raise ValueError('UoT episode coverage/hash/call range differs')
        episode = read_json(output / name)
        if (episode.get('status') != 'complete' or episode.get('method') != 'uot' or
                episode.get('case_id') != identity['case_id'] or episode.get('seed') != identity['seed'] or
                episode.get('variant_suffix') != '' or episode.get('parser_repair') is not None):
            raise ValueError('Native episode identity differs')
        previous = end
    accounting = result['accounting']
    if (set(accounting) != set(batch.COUNTERS) or any(type(v) is not int or v < 0 for v in accounting.values()) or
            accounting['calls'] != previous or not 0 < previous <= 21500 or
            accounting['reserved_output'] != previous * 512 or
            accounting['reserved_input'] != accounting['actual_input'] or
            accounting['reserved_input'] > plan['max_input_reserved'] or
            accounting['actual_output'] > accounting['reserved_output']):
        raise ValueError('UoT runtime accounting differs')
    return dict(completed=True, controller_status=result['status'],
                coverage={'expected_count': 20, 'recorded_count': 20, 'completed_count': 20},
                result_sha256=sha(output / 'result.json'), model_calls=previous,
                scientific_success_claimed=False)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', '--package', dest='root', type=Path, required=True)
    base = parser.parse_args(argv).root.absolute()  # Keep lexical /home argv identity.
    config, protocol, cases, scope = load_frozen(base)
    context = allocation_context(scope, os.environ, platform=sys.platform, now=time.time())
    bindings = binding_hashes(base)
    recorded = read_json(base / 'allocation-context.json')
    if recorded != dict(context, bindings=bindings):
        raise ValueError('Recorded allocation/source identity changed')
    sys.path.insert(0, config['old_code'])
    from experiments.sun_gn_context.runtime import PinnedTokenizer, LoopbackTransport
    from experiments.sun_gn_context.journal import Journal
    from experiments.sun_gn_context.allocation import write_exclusive
    from experiments.sun_gn_context.sampling import validate_environment
    validate_environment()
    tokenizer = PinnedTokenizer(Path(config['tokenizer']))
    runtime = read_json(config['runtime'])
    if hashlib.sha256(tokenizer.tokenizer.chat_template.encode()).hexdigest() != runtime['server_chat_template_sha256']:
        raise ValueError('Existing tokenizer/server template changed')
    adapter = load_module(base, 'baseline-adapters/arbench_adapter.py', '_uot_live_adapter')
    original = load_module(base, 'gn-runtime/qwen_binding.py', '_uot_live_qwen')
    binding = load_module(base, ENTRY_DIR + '/uot_binding.py', '_uot_live_roles')
    batch = load_module(base, ENTRY_DIR + '/uot_batch.py', '_uot_live_batch')
    journal = Journal(base / 'allocation/runtime-journal.jsonl', recorded)
    def admit(plan):
        if load_frozen(base) != (config, protocol, cases, scope) or binding_hashes(base) != bindings:
            raise ValueError('Admission package/source binding changed')
        allocation_context(scope, os.environ, platform=sys.platform, now=time.time())
        return dict(admitted=True, bindings=bindings, job_id=context['job_id'],
                    max_calls=21500, max_output_tokens=11008000,
                    max_input_reserved=scope['max_input_reserved'])
    try:
        Boundary = binding.make_uot_boundary_class(original.QwenBoundary)
        boundary = Boundary(execution_enabled=True,
            transport=LoopbackTransport('http://127.0.0.1:18000/v1', timeout=120),
            tokenizer=tokenizer, journal=journal,
            protocol={'actor': {'max_tokens': 512, 'temperature': 0.7, 'top_p': 0.7}},
            plan={'max_calls': 21500, 'max_output_tokens': 11008000},
            deadline=context['request_deadline'], max_input_reserved=scope['max_input_reserved'])
        batch.run_batch(protocol, cases, adapter=adapter, boundary=boundary, admission=admit,
            output_dir=base / 'allocation/controller', execution_enabled=True)
        complete = check_completion(base / 'allocation/controller', protocol)
        journal.append('controller_completed', complete)
        write_exclusive(base / 'allocation/controller-complete.json', dict(complete, bindings=bindings))
    except BaseException as exc:
        journal.append('controller_failed', {'error_type': type(exc).__name__, 'message': str(exc)})
        raise
    finally:
        journal.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
