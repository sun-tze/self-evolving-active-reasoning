import copy
import unittest
import hashlib
import json
import tempfile
from pathlib import Path
from .audit import agreement, contrasts, rank_pairs, validate_review, verify_inputs
from .provenance import verified_targets


class AuditTests(unittest.TestCase):
    def fixture(self):
        packet = {'reviewer_id': 'r1', 'rubric_sha256': 'rubric', 'items': [
            {'item_id': 'opaque', 'answer': 'A hidden passage.', 'history': [{'ref': 'H01'}]}]}
        rating = {'item_id': 'opaque', 'semantic_correctness': 0, 'referee_consistency': 0,
                  'unsupported_severity': 2, 'evidence_faithfulness': 0, 'rationale': 'Denied mechanism.',
                  'issues': [{'kind': 'contradicted', 'role': 'central', 'answer_quote': 'hidden passage',
                              'evidence_refs': ['H01'], 'reason': 'Explicit denial.'}]}
        return packet, {'reviewer_id': 'r1', 'rubric_sha256': 'rubric', 'packet_sha256': 'packet',
                        'independence_statement': 'Only assigned inputs.', 'ratings': [rating]}

    def test_frozen_source_and_blinding(self):
        source, packets, mappings = verify_inputs()
        self.assertEqual(len(source), 20)
        allowed = {'item_id', 'surface', 'history', 'reference_solution', 'semantic_anchors', 'answer'}
        for packet in packets.values():
            self.assertTrue(all(set(item) == allowed for item in packet['items']))
        self.assertTrue(set(mappings['r1']).isdisjoint(mappings['r2']))

    def test_valid_review_and_reject_corruption(self):
        packet, review = self.fixture()
        validate_review(review, packet, 'packet')
        mutations = [lambda x: x.update(packet_sha256='changed'),
                     lambda x: x['ratings'].append(copy.deepcopy(x['ratings'][0])),
                     lambda x: x['ratings'][0].update(semantic_correctness=True),
                     lambda x: x['ratings'][0].update(evidence_faithfulness=2),
                     lambda x: x['ratings'][0]['issues'][0].update(answer_quote='not present'),
                     lambda x: x['ratings'][0]['issues'][0].update(evidence_refs=['H26']),
                     lambda x: x['ratings'][0].update(issues=[])]
        for mutate in mutations:
            bad = copy.deepcopy(review)
            mutate(bad)
            with self.assertRaises(ValueError):
                validate_review(bad, packet, 'packet')

    def test_contrast_signs_and_interaction(self):
        result = contrasts(dict(zip('ABCD', [0, 1, 0, 0])))
        self.assertEqual(result['writer_marginal'], 0.5)
        self.assertEqual(result['dialogue_marginal'], -0.5)
        self.assertEqual(result['interaction'], -1)

    def test_issue_cannot_coexist_with_inflated_scores(self):
        packet, review = self.fixture()
        review['ratings'][0].update(referee_consistency=2, unsupported_severity=0, evidence_faithfulness=2)
        with self.assertRaises(ValueError):
            validate_review(review, packet, 'packet')
        review['ratings'][0].update(unsupported_severity=2, evidence_faithfulness=0)
        with self.assertRaises(ValueError):
            validate_review(review, packet, 'packet')

    def test_target_binding_rejects_self_consistent_forgery(self):
        target = {'index': 7, 'bottom': 'truth', 'target_sha256': hashlib.sha256(b'truth').hexdigest(), 'dataset_record_sha256': 'record'}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'targets.jsonl'
            raw = (json.dumps(target)+'\n').encode()
            path.write_bytes(raw)
            manifest = {'targets_sha256': hashlib.sha256(raw).hexdigest(), 'case_bindings': {'7': {'bare': {'dataset_record_sha256': 'record'}, 'first': {'dataset_record_sha256': 'record'}}}}
            self.assertEqual(verified_targets(path, manifest, [7])[7], target)
            target.update(bottom='forged', target_sha256=hashlib.sha256(b'forged').hexdigest())
            path.write_text(json.dumps(target)+'\n', encoding='utf-8')
            with self.assertRaises(ValueError):
                verified_targets(path, manifest, [7])
            path.write_bytes(raw)
            manifest['case_bindings']['7']['first']['dataset_record_sha256'] = 'different'
            with self.assertRaises(ValueError):
                verified_targets(path, manifest, [7])

    def test_reversals_exclude_ties(self):
        result = rank_pairs(dict(zip('ABCD', [0, 1, 2, 3])), dict(zip('ABCD', [1, 0, 0, 1])))
        self.assertEqual(sum(x['classification'] == 'reversal' for x in result), 2)
        self.assertEqual(sum(x['classification'] == 'metric_tie' for x in result), 2)

    def test_kappa_handles_degeneracy(self):
        self.assertIsNone(agreement([0, 0], [0, 0])['quadratic_weighted_kappa'])
        self.assertEqual(agreement([0, 1, 2], [0, 1, 2])['quadratic_weighted_kappa'], 1)
        self.assertEqual(agreement([0, 2], [2, 0])['quadratic_weighted_kappa'], -1)


if __name__ == '__main__':
    unittest.main()
