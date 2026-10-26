"""Small synthetic checks; no experiment files, models or external packages."""
from copy import deepcopy
import unittest
import verify_results as v


def document():
    rows=[]
    for i in range(200):
        arms={}
        for arm,cost in zip(v.ARMS,(10,7,9,7,9)):
            arms[arm]=dict(capped25=cost,queries=cost,first_unique_turn=cost,resolved=True,invalid=0,truncated=0)
        rows.append(dict(case_index=i,cohort=i//100,index=i%100,arms=arms))
    return dict(schema=v.SCHEMA,N=200,cases=rows)


class PublicArithmeticTests(unittest.TestCase):
    def test_exact_denominator_order_and_boolean_types(self):
        good=document();self.assertEqual(len(v.validate_cases(good)),200)
        bad=deepcopy(good);bad['cases'][0]['case_index']=False
        with self.assertRaises(ValueError):v.validate_cases(bad)
        bad=deepcopy(good);bad['cases'][0],bad['cases'][1]=bad['cases'][1],bad['cases'][0]
        with self.assertRaises(ValueError):v.validate_cases(bad)
        bad=deepcopy(good);bad['cases'][0]['arms']['E0']['resolved']=1
        with self.assertRaises(ValueError):v.validate_cases(bad)

    def test_unresolved_at_25_is_distinct_from_resolved_at_25(self):
        data=document()
        data['cases'][0]['arms']['B0'].update(capped25=25,queries=25,first_unique_turn=None,resolved=False)
        data['cases'][1]['arms']['B0'].update(capped25=25,queries=25,first_unique_turn=25,resolved=True)
        rows=v.validate_cases(data);summary=v.arm_summary(rows)['B0']
        self.assertEqual(summary['resolved'],199)
        self.assertEqual(summary['resolution_curve']['24'],198)
        self.assertEqual(summary['resolution_curve']['25'],199)

    def test_paired_integer_contrasts(self):
        self.assertEqual(v.doubled_contrasts(document()['cases'][0]),(-6,-2,-6,-2,-6,-4))
        sample=v.resample([[(-6,-2,-6,-2,-6,-4)]*3],20,63026)
        self.assertEqual(sample[4],[-3]*20)
        self.assertEqual(sample[5],[-2]*20)

    def test_stratified_fixed_sizes_not_pooled_draws(self):
        samples=v.resample([[(0,)*6]*2,[(20,)*6]*2],50,63028)
        self.assertTrue(all(values==[5]*50 for values in samples))

    def test_frozen_percentile_interpolation(self):
        x=list(range(20000))
        self.assertEqual(v.quantile(x,.00625),124.99375)
        self.assertEqual(v.quantile(x,.99375),(len(x)-1)*.99375)
        self.assertEqual(v.quantile(x,.025),499.975)

    def test_point_two_does_not_imply_interval_below_two(self):
        cohorts={}
        for c in ('0','1'):
            cohorts[c]=dict(contrasts={'Emean-B0':dict(mean_difference=-2,CI98_75_primary=[-3,-.5]),
                'Emean-Rmean':dict(mean_difference=-1,CI98_75_primary=[-2,-.1]),
                'E0-B0':dict(mean_difference=-2),'E1-B0':dict(mean_difference=-2)})
        counts=dict(B0=180,E0=190,E1=190,R0=180,R1=180)
        flags=v.derive_flags(cohorts,counts)
        self.assertTrue(flags['repeated_benefit_and_trace_value'])
        self.assertFalse(flags['both_cohort_Emean_B0_primary_upper_below_minus_two'])
        counts['E1']=179
        self.assertFalse(v.derive_flags(cohorts,counts)['repeated_observed_practical_benefit'])

    def test_exact_comparison_detects_numeric_and_flag_tampering(self):
        diff=[];v.compare_exact({'n':True,'flag':False,'ci':[1,2.01]}, {'n':1,'flag':True,'ci':[1,2.0]},'test',diff)
        self.assertEqual(len(diff),3)

    def test_duplicate_and_nonfinite_json_rejected(self):
        for text in ('{"N":200,"N":100}','{"x":NaN}'):
            with self.assertRaises(ValueError):v.decode(text)


if __name__=='__main__':unittest.main()
