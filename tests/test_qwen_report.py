import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import lab
import qwen_report
from storage import write_json


class FollowupReportTests(unittest.TestCase):
    def test_resource_refusal_exports_zero_rows_and_both_missing_pairs(self):
        actual_root=lab.ROOT
        cfg=json.loads((actual_root/'configs/qwen-mac-followup.json').read_text())
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);folder=root/'runs'/'refused';folder.mkdir(parents=True)
            grader=(actual_root/'scripts/math_grade.py').read_bytes()
            for path in [root/'scripts/math_grade.py',folder/'source/scripts/math_grade.py']:
                path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(grader)
            manifest={'problems':[{'unique_id':key} for key in cfg['selected_problem_ids']]}
            write_json(folder/'source'/cfg['manifest'],manifest)
            write_json(folder/'audit.json',{'checks':{'identity':True}})
            cfg['manifest_sha256']=qwen_report.sha(folder/'source'/cfg['manifest'])
            cfg['audit_sha256']=qwen_report.sha(folder/'audit.json')
            write_json(folder/'metadata.json',{'experiment':'QWN-003','experiment_config':cfg,'model':{},'environment':{},
                'math_verify_version':'0.9.0','source_hashes':{'scripts/math_grade.py':hashlib.sha256(grader).hexdigest()}})
            write_json(folder/'outputs.json',[])
            write_json(folder/'interruption.json',{'reason':'pressure','active_trial':None})
            (folder/'resources.jsonl').write_text(json.dumps({'available_bytes':3*2**30,'pressure_level':2,'thermal_report':'private telemetry'})+'\n')
            with patch.object(lab,'ROOT',root):data=qwen_report.collect('refused')
            self.assertEqual(data['experiment'],'QWN-003')
            self.assertEqual(data['rows'],[])
            self.assertEqual(data['pairs'],[])
            self.assertEqual(data['missing_or_wall_interrupted_pairs'],cfg['selected_problem_ids'])
            self.assertNotIn('thermal_report',data['resources'][0])
            self.assertIsNone(data['summary'])
