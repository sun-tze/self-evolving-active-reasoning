"""Per-case native UoT episode for run_cases; same invocation as the pinned serial binding.

uot_binding.run_native_uot_episode passes the shared boundary.call. Here each case
gets its own chat from ConcurrentBoundary.chat_for, so the journal can tag calls
with the case. Every other argument, and the constants below, are the pinned
binding's (checked by test_concurrent_boundary.py). No parser repair, memory or
extra final actor call.
"""

MAX_TURNS = 25
SIMULATE_DEPTH = 1
EPISODE_CALL_BOUND = 1075  # 25 * (one proposal + three candidates * 14 partitions)


def native_uot_runner(adapter):
    def run(case, chat):
        return adapter.run_gn_episode(
            method='uot', secret=case['secret'], chat=chat, seed=case['seed'],
            limits=adapter.CallLimits(EPISODE_CALL_BOUND, 512), max_turn=MAX_TURNS,
            temperature=0.7, top_p=0.7, simulate_depth=SIMULATE_DEPTH,
            memory_context='', case_id=case['case_id'], repair_mode=None,
            matched_finalization=False)
    return run
