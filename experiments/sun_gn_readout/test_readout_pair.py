import json
from pathlib import Path
import unittest
from readout_pair import branch_readout, seed
from paired_statistics import paired

P=json.loads((Path(__file__).parent/'readout_settings.json').read_bytes())
class Boundary:
    def __init__(self):self.requests=[]
    def call(self,messages,cap,call_seed,metadata):
        self.requests.append(dict(messages=messages,cap=cap,seed=call_seed,metadata=metadata))
        return dict(text='Guess: 9999')

class PairTests(unittest.TestCase):
    def test_same_history_seed_cap_no_answer_substitution(self):
        for index in (0,1):
            b=Boundary();messages=[dict(role='system',content='synthetic rules')]
            r=branch_readout('synthetic',index,messages,[dict(turn=1,guess='0123',same=4,different=0)],b,P,{'final_guess_prompt':'Final instruction'})
            self.assertEqual(len(b.requests),2)
            self.assertEqual([q['cap'] for q in b.requests],[128,128])
            self.assertEqual([q['seed'] for q in b.requests],[seed('synthetic',26)]*2)
            self.assertEqual(b.requests[0]['messages'][:-1],b.requests[1]['messages'][:-1])
            self.assertEqual(b.requests[index]['metadata']['arm'],'baseline')
            self.assertTrue(b.requests[1-index]['messages'][-1]['content'].endswith('Final instruction'))
            self.assertEqual(r['symbolic_answer'],'0123')
            self.assertEqual(r['results']['state_assisted']['text'],'Guess: 9999')
    def test_sparse_interval_does_not_claim_equivalence(self):
        r=paired([False]*50,[False]*50)
        self.assertEqual(r['delta'],0)
        self.assertEqual(r['bootstrap_CI95'],[0,0])
        self.assertLess(r['conservative_simultaneous_binomial_CI95'][0],0)
        self.assertGreater(r['conservative_simultaneous_binomial_CI95'][1],0)

if __name__=='__main__':unittest.main()
