"""Recalculate released paired statistics; no models or private inputs."""
import json
from pathlib import Path
from paired_statistics import paired

def check(study, a, b):
    rows=study['cases']
    assert len(rows)==study['N'] and len({r['pair'] for r in rows})==len(rows)
    actual=paired([r['arms'][a]['EM'] for r in rows],[r['arms'][b]['EM'] for r in rows])
    assert actual==study['primary'], (actual,study['primary'])
    for arm in study['conditions']:
        for key in ('EM','consistent','invalid','truncated','input_tokens','output_tokens'):
            assert sum(r['arms'][arm][key] for r in rows)==study['conditions'][arm][key]
    return actual

if __name__=='__main__':
    r=json.loads((Path(__file__).parent/'results.json').read_bytes())
    print('Limited feedback:',check(r['limited_feedback'],'R3','R2'))
    if r['prospective_state_readout']['status']=='AUDITED_COMPLETE':
        print('Prospective state readout:',check(r['prospective_state_readout'],'state_assisted','baseline'))
    else:print('Prospective study has no released result yet.')
