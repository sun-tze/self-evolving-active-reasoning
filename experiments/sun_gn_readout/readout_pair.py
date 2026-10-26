"""Portable final-readout adapter extracted from the evaluated runner.

Caller supplies public messages/records, frozen prompts and a boundary exposing
call(messages, max_tokens, seed, metadata). It must enforce resource limits and
record the unmodified response. This adapter neither sees labels nor scores.
"""
import hashlib
import json
import time
from constraint_readout import public_state

def seed(case_id, slot, purpose="actor"):
    raw = f"sun-gn-v1\n{case_id}\n{purpose}\n{slot}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % 2147483647

def branch_readout(cid, index, messages, history, boundary, protocol, prompts):
    # Neither the answer nor its hash is an argument. This function does not score.
    started = time.thread_time()
    state = public_state(history, protocol['list_cap'])
    resolver_cpu_seconds = time.thread_time() - started
    ordinary = prompts['final_guess_prompt']
    assisted = protocol['state_prefix'] + json.dumps(state, sort_keys=True, separators=(',', ':')) + protocol['state_suffix'] + '\n\n' + ordinary
    results = {}
    order = ('baseline', 'state_assisted') if index % 2 == 0 else ('state_assisted', 'baseline')
    for arm in order:
        chain = messages + [dict(role='user', content=ordinary if arm == 'baseline' else assisted)]
        results[arm] = boundary.call(chain, 128, seed(cid, 26), dict(case_id=cid, arm=arm, development=False))
    return dict(case_id=cid, index=index, results=results, public_history=history, public_state=state,
                resolver_cpu_seconds=resolver_cpu_seconds,
                symbolic_answer=state['candidates'][0] if state['candidate_count']==1 else None,
                shared_messages_sha256=hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest())

