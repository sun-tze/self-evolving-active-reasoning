"""Offline regressions for selection, blinding, provenance and authorization."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from . import study as S, semantic as M, submit_once as U, guard as G
from experiments.sun_c2_real.tests.test_real import fixture


class ConfirmatoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.manifest, self.rows, self.msha = fixture(self.root)
        self.doc, self.selected, self.counts = S.R.plan(
            self.root, self.msha, self.root/'budget.json', 'confirmatory',
            'http://127.0.0.1:18000/v1', 260, 26000, 104000, 'test-marker')

    def test_selection_preserves_original_payloads_and_excludes_pilot(self):
        self.assertEqual(len(self.selected), 260)
        self.assertEqual(len({r['index'] for r in self.selected}), 65)
        self.assertFalse(set(self.manifest['pilot_indices']) & {r['index'] for r in self.selected})
        originals = {r['request_id']:r for r in self.rows}
        self.assertTrue(all(r == originals[r['request_id']] for r in self.selected))
        self.assertEqual(len(self.manifest['unseen_indices']), 70)
        with self.assertRaises(ValueError):
            S.C.scope_indices(self.manifest, 'unknown')

    def test_each_budget_cap_is_enforced(self):
        for caps in [(259,26000,104000),(260,25999,104000),(260,26000,103999)]:
            with self.subTest(caps=caps), self.assertRaisesRegex(ValueError, 'caps'):
                S.R.plan(self.root,self.msha,self.root/'budget.json','confirmatory',
                         'http://127.0.0.1:18000/v1',*caps)

    def test_review_assignments_cover_each_answer_twice(self):
        assignments = S.assignments(self.selected)
        self.assertEqual(assignments, S.assignments(self.selected))
        self.assertEqual(len(assignments), 26)
        self.assertTrue(all(len(v)==20 for v in assignments.values()))
        self.assertEqual(len({r['item_id'] for v in assignments.values() for r in v}), 520)
        for stream in ('r1','r2'):
            ids = [r['request_id'] for k,v in assignments.items() if k.startswith(stream) for r in v]
            self.assertEqual(len(ids),len(set(ids)))
            self.assertEqual(set(ids),{r['request_id'] for r in self.selected})

    def test_reference_tampering_is_rejected(self):
        path=self.root/'private/targets.jsonl'
        path.write_bytes(path.read_bytes()+b' ')
        with self.assertRaisesRegex(ValueError,'target-file'):
            S.targets(self.root,self.manifest,S.C.scope_indices(self.manifest,'confirmatory'))

    def test_analysis_is_rebuilt_from_raw_journal(self):
        run=self.root/'run'
        run.mkdir()
        S.C.write_json(run/'run.json',{'plan':self.doc,'plan_sha256':S.C.digest(self.doc)})
        request=self.selected[0]
        raw={'model':S.C.MODEL,'choices':[{'message':{'content':'Synthetic test answer'},'finish_reason':'stop'}],
             'usage':{'prompt_tokens':100,'completion_tokens':3,'total_tokens':103}}
        base={'request_id':request['request_id'],'payload_sha256':request['payload_sha256']}
        S.C.write_jsonl(run/'journal.jsonl',[
            dict(base,event='started'),dict(base,event='result',status='completed',raw=raw,
                 parsed=S.R.parse_completion(raw,request),error=None,elapsed_s=1)])
        path=self.root/'analysis.json'
        original=M.A.analyze(self.root,self.msha,run,path)
        with patch.object(M,'plan',return_value=(self.doc,self.selected,self.counts)):
            self.assertEqual(M.verified_analysis(self.root,path,'freeze',run),original)
            altered=copy.deepcopy(original)
            altered['responses'][0]['response']['text']='Substituted answer'
            S.C.write_json(path,altered)
            with self.assertRaisesRegex(ValueError,'journal'):
                M.verified_analysis(self.root,path,'freeze',run)
            S.C.write_json(path,original)
            changed=copy.deepcopy(self.doc)
            changed['totals']['calls']=261
            S.C.write_json(run/'run.json',{'plan':changed,'plan_sha256':S.C.digest(changed)})
            with self.assertRaisesRegex(ValueError,'frozen execution'):
                M.verified_analysis(self.root,path,'freeze',run)

    def test_packet_schema_hides_condition_identity_and_scores(self):
        here=self.root/'design'
        here.mkdir()
        S.C.write_json(here/'assignments.json',S.assignments(self.selected))
        S.C.write_json(here/'anchors.json',{'cases':{str(i):{'required':['test mechanism']} for i in self.doc['indices']}})
        S.C.write_json(here/'review_protocol.json',{})
        analysis={'scope':'confirmatory','provenance':{'manifest_sha256':self.msha},
            'responses':[{'request_id':r['request_id'],'index':r['index'],'cell':r['cell'],
                          'status':'completed','score':.25,'response':{'text':'test answer'}} for r in self.selected]}
        with patch.object(M,'HERE',here), patch.object(M,'verify',return_value=({'manifest_sha256':self.msha},self.manifest,self.selected)), patch.object(M,'verified_analysis',return_value=analysis):
            packets,_,_=M.packets(self.root,'analysis','freeze','run')
        self.assertEqual(sum(len(p['items']) for p in packets.values()),520)
        for packet in packets.values():
            for item in packet['items']:
                self.assertEqual(set(item),{'item_id','surface','history','reference_solution','semantic_anchors','answer'})
                self.assertEqual(set(item['history'][0]),{'ref','question','reply'})

    def test_positive_paired_effect_and_missing_withholding(self):
        rows=[{'index':i,'cell':c,'semantic_correctness':2 if c in 'BD' else 0,
               'evidence_faithfulness':1} for i in range(2) for c in 'ABCD']
        result=M.summarize(rows,[0,1],{0})
        self.assertEqual(result['primary_status'],'positive_effect_supported')
        self.assertEqual(result['paired_contrasts_normalized']['semantic_correctness']['writer_marginal']['ci95'],[1,1])
        self.assertEqual(result['paired_contrasts_normalized']['semantic_correctness']['dialogue_marginal']['mean'],0)
        self.assertEqual(result['cell_means']['B']['semantic_correctness']['fully_scored_2'],2)
        rows[0]['semantic_correctness']=None
        result=M.summarize(rows,[0,1],{0})
        self.assertEqual(result['primary_status'],'incomplete_withheld')
        self.assertIsNone(result['paired_contrasts_normalized']['semantic_correctness']['writer_marginal']['ci95'])

    def test_authorization_binds_freeze_commit_and_limits(self):
        here=self.root/'design'
        here.mkdir()
        S.C.write_json(here/S.FREEZE_NAME,{'fixed':True})
        S.C.write_json(here/'protocol.json',{'gpu_limits':{'jobs':1,'completion_calls':260}})
        freeze=S.file_sha(here/S.FREEZE_NAME)
        record={'approved':True,'freeze_sha256':freeze,'limits':{'jobs':1,'completion_calls':260},
                'authorization_id':'test','code_commit':'a'*40}
        path=self.root/'approval.json'
        with patch.object(S,'HERE',here):
            S.C.write_json(path,record)
            self.assertEqual(S.authorization(path,freeze),record)
            for field,value in [('approved',False),('freeze_sha256','b'*64),('limits',{'jobs':2}),('code_commit','branch')]:
                altered=dict(record)
                altered[field]=value
                S.C.write_json(path,altered)
                with self.subTest(field=field), self.assertRaises(ValueError):
                    S.authorization(path,freeze)

    def test_no_approval_means_no_scheduler_calls(self):
        args=['submit','--bundle',str(self.root),'--freeze-sha256','bad','--authorization-file',str(self.root/'absent'),
              '--private-root',str(self.root/'private'),'--runtime-python','python','--cache',str(self.root)]
        with patch('sys.argv',args), patch.object(U.subprocess,'run') as run, patch.object(U.subprocess,'check_output') as read:
            with self.assertRaises(FileNotFoundError):
                U.main()
            run.assert_not_called()
            read.assert_not_called()

    def test_previous_submission_claim_prevents_scheduler_calls(self):
        private=self.root/'submission'
        private.mkdir()
        approved={'code_commit':'a'*40,'authorization_id':'one-approval'}
        with patch.object(G,'LEDGER_ROOT',self.root/'ledger'):
            G.claim(G.ledger(approved)/'SUBMISSION_ATTEMPTED.json',{})
        args=['submit','--bundle',str(self.root),'--freeze-sha256','f','--authorization-file','a',
              '--private-root',str(private),'--runtime-python',str(Path.home()/'code/ar_harness/conda_env/bin/python'),
              '--cache',str(Path.home()/'orcd/pool/ar_harness/hf_cache')]
        with patch('sys.argv',args), patch.object(G,'LEDGER_ROOT',self.root/'ledger'), patch.object(U,'authorization',return_value=approved), patch.object(U,'verify'), patch.object(U.subprocess,'check_output',side_effect=['a'*40,'']), patch.object(U.subprocess,'run') as run:
            with self.assertRaisesRegex(ValueError,'already attempted'):
                U.main()
            run.assert_not_called()

    def test_launch_and_inference_bound_to_one_job(self):
        record={'authorization_id':'one-approval','approved':True}
        with patch.object(G,'LEDGER_ROOT',self.root/'ledger'), patch.dict('os.environ',{'SLURM_JOB_ID':'123'}):
            G.claim(G.ledger(record)/'SUBMISSION_ATTEMPTED.json',{'approval_sha256':S.C.digest(record)})
            G.receipt(record,'123')
            G.launch(record)
            with self.assertRaises(FileExistsError):
                G.launch(record)
            G.inference(record)
            with self.assertRaises(FileExistsError):
                G.inference(record)
            with patch.dict('os.environ',{'SLURM_JOB_ID':'124'}), self.assertRaisesRegex(ValueError,'another Slurm job'):
                G.allocated_job(record)

    def test_preflight_must_use_launcher_environment(self):
        args=['submit','--bundle',str(self.root),'--freeze-sha256','f','--authorization-file','a',
              '--private-root',str(self.root/'out'),'--runtime-python','wrong-python','--cache','wrong-cache']
        with patch('sys.argv',args), patch.object(U,'authorization',return_value={}), patch.object(U.subprocess,'run') as run:
            with self.assertRaisesRegex(ValueError,'runtime path'):
                U.main()
            run.assert_not_called()

    def test_full_offline_aggregation_retains_disagreement_and_reversals(self):
        here, packets, reviews = [self.root/n for n in ('design','packets','reviews')]
        for directory in (here,packets,reviews):
            directory.mkdir()
        assignment=S.assignments(self.selected)
        S.C.write_json(here/'assignments.json',assignment)
        S.C.write_json(here/'anchors.json',{'cases':{str(i):{} for i in self.doc['indices']}})
        by_id={r['request_id']:r for r in self.selected}
        expected, hashes = {}, {}
        for name, entries in assignment.items():
            packet={'reviewer_id':name,'rubric_sha256':'test','items':[{'item_id':r['item_id'],'answer':'synthetic','history':[]} for r in entries]}
            expected[name]=packet
            S.C.write_json(packets/f'{name}.json',packet)
            hashes[name]=S.file_sha(packets/f'{name}.json')
            ratings=[{'item_id':r['item_id'],'semantic_correctness':2 if by_id[r['request_id']]['cell'] in 'BD' else 0,
                      'referee_consistency':2,'unsupported_severity':0,'evidence_faithfulness':2,
                      'rationale':'Synthetic regression fixture.','issues':[]} for r in entries]
            if name=='r2_b01':
                ratings[0]['semantic_correctness']=1
            S.C.write_json(reviews/f'{name}.json',{'reviewer_id':name,'rubric_sha256':'test',
                         'packet_sha256':hashes[name],'independence_statement':'Synthetic fixture; no model review.', 'ratings':ratings})
        analysis={'responses':[{'request_id':r['request_id'],'status':'completed','score':1 if r['cell'] in 'AC' else 0} for r in self.selected],
                  'cell_means':{c:{'available':65,'mean':1 if c in 'AC' else 0} for c in 'ABCD'},
                  'contrasts':{},'new_inference_cost':{},'truncated_completions':0}
        path=self.root/'analysis.json'
        S.C.write_json(path,analysis)
        S.C.write_json(packets/'binding.json',{'freeze_sha256':'test','analysis_sha256':S.file_sha(path),'packets':hashes})
        with patch.object(M,'HERE',here), patch.object(M,'packets',return_value=(expected,analysis,self.selected)):
            result=M.aggregate(self.root,path,'test','run',packets,reviews,self.root/'result.json')
        self.assertEqual(result['annotation_count'],520)
        self.assertEqual(len(result['disagreements']),1)
        self.assertEqual(result['primary_status'],'positive_effect_supported')
        self.assertEqual(result['condition_mean_ranking_pairs']['semantic_correctness'][0]['classification'],'reversal')
        self.assertEqual(result['ordinal_counts_by_reviewer_and_cell']['r1']['B']['semantic_correctness']['2'],65)


if __name__ == '__main__':
    unittest.main()
