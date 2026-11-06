from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from campaign_report import replication_criteria


class CriteriaTests(unittest.TestCase):
    def fixture(self):
        targets=[{'layer':4,'candidate':'L04N5848'},{'layer':20,'candidate':'L20N0472'}]
        selected=[{'layer':layer,'candidate':f'L{layer}C{control}','role':'same-layer-random','target_fraction':.2}
                  for layer in [4,20] for control in [1,2]]
        conditions=['baseline',*[x['candidate']+'@0.2' for x in targets+selected]]
        replies=[]
        for condition in conditions:
            for i in range(8):
                replies.append({'condition':condition,'prompt_id':str(i),'cohort':'fresh-behavior' if i<6 else 'fresh-check',
                    'tokens':100 if condition=='baseline' else (80 if condition.startswith('L04N') else 90),
                    'eos':True,'exact_check':True if i>=6 else None})
        return {'config':{'targets':targets},'selected':selected,'replies':replies}

    def test_exact_check_loss_blocks_compression_even_when_aggregate_length_improves(self):
        data=self.fixture()
        self.assertEqual(replication_criteria(data)['L04N5848@0.2']['status'],'met')
        next(x for x in data['replies'] if x['condition']=='L04N5848@0.2' and x['prompt_id']=='6')['exact_check']=False
        result=replication_criteria(data)['L04N5848@0.2']
        self.assertEqual(result['status'],'not met')
        self.assertEqual(result['lost_exact_checks'],['6'])

    def test_no_repair_opportunity_is_inconclusive_and_new_caps_stay_visible(self):
        data=self.fixture()
        next(x for x in data['replies'] if x['condition']=='L20N0472@0.2' and x['prompt_id']=='0')['eos']=False
        result=replication_criteria(data)['L20N0472@0.2']
        self.assertIn('inconclusive',result['status'])
        self.assertEqual(result['added_cap_failures'],['0'])
