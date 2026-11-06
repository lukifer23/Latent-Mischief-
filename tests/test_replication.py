from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from replicate_offsets import select_units


class ReplicationSelectionTests(unittest.TestCase):
    def test_controls_exclude_target_capped_and_mismatched_strength_channels(self):
        cfg={'seed':123,'targets':[{'layer':4,'neuron':1}],'strengths':[.05,.2],'random_controls_per_layer':2}
        rows=[]
        for neuron in range(1,6):
            for target in cfg['strengths']:
                rows.append({'candidate':f'L04N{neuron:04d}','layer':4,'neuron':neuron,'sign':1,'target_fraction':target,
                    'hit_cap':neuron==4,'mean_actual_fraction':target*(2 if neuron==5 else 1),'mean_kl':.1})
        units=select_units(cfg,rows)
        self.assertEqual({x['neuron'] for x in units if x['role']=='same-layer-random'},{2,3})
        self.assertEqual(units,select_units(cfg,rows))
        self.assertEqual(len(units),6)
