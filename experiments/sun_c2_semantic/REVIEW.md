# Independent code and method review

Date: 2026-10-15. Reviewer: the independent C2 code-review agent, separate from both fresh-context blind semantic reviewers. This reviewer did not read semantic ratings or contact those reviewers. It inspected the new extraction, rubric, confirmatory proposal, analyzer and tests; it did not rerun old suites or invoke GPU/API services.

Two P2 findings were reported and fixed:

1. Validation initially enforced score-to-issue consistency only. A central contradiction issue could coexist with full consistency/faithfulness scores. `audit.py` now rejects the reverse inconsistency as well. Ratings are never silently changed. The regression reproduces the original invalid combination; both real raw reviews passed the strengthened validator unchanged.
2. Original extraction checked a reference against its self-declared text hash but did not compare the full targets file with the pinned manifest or both source record identities. `provenance.py` now verifies that full file hash, the selected text hashes, both historical dataset bindings and every frozen reference. The actual private-bundle check passed. `prepare_verified.py` performs target verification before future extraction. The original `prepare.py` and all rubric/packet/freeze bytes remain unchanged as historical freeze evidence; the verified entry is the supported route.

The reviewer re-inspected these fixes and reported: both findings closed for the current audit, no residual actionable concerns in that limited recheck. It found no issue with mean-of-two aggregation, normalized paired contrasts, weighted kappa or tie handling. It did not independently validate the substantive semantic judgments, inspect the remote freeze commit or run tests during the recheck. The root agent separately verified the remote branch head through the GitHub connector.

Validation evidence: five initial tests passed; two new regression tests and the affected original validator test passed after fixes. Seven distinct tests, eight executions. Actual pinned-target provenance and both completed reviews passed validation. No historical scores or blind ratings were modified.
