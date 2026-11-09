from pathlib import Path
import sys
import unittest
import json
import tempfile
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from catalog import recipe_id
import catalog


class RecipeIdentityTests(unittest.TestCase):
    def test_fresh_clone_keeps_public_recipes_and_merge_does_not_duplicate_observations(self):
        recipe={'family':'channel-gain','model':{'revision':'pinned'}}
        identifier=recipe_id(recipe)
        entry={'id':identifier,'recipe':recipe,'methods':[{'run':'historical'}],
               'observations':[{'run':'historical','cohort':'fresh','n':3}]}
        public={'recipes':[entry],'runs':{'historical':{'experiment':'MAP-001'}},'run_refs':{'historical':{'0':identifier}}}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'experiments/personas/registry.json';path.parent.mkdir(parents=True)
            path.write_text(json.dumps(public))
            (root/'runs').mkdir()
            with patch.object(catalog,'ROOT',root):
                built=catalog.build()
                self.assertEqual(built['recipes'],[entry])
                rebuilt=catalog.merge_public(built)
                self.assertEqual(rebuilt,built)
                self.assertIn('historical',rebuilt['runs'])
                self.assertEqual(rebuilt['run_refs']['historical']['0'],identifier)

    def test_identity_is_key_order_stable_but_exact_change_sensitive(self):
        a={'model':{'revision':'abc','dtype':'bf16'},'operations':[{'layer':4,'offsets':[.2],'channels':[10]}]}
        reordered={'operations':a['operations'],'model':{'dtype':'bf16','revision':'abc'}}
        self.assertEqual(recipe_id(a),recipe_id(reordered))
        b={'model':a['model'],'operations':[{'layer':4,'offsets':[.20001],'channels':[10]}]}
        self.assertNotEqual(recipe_id(a),recipe_id(b))
        with self.assertRaises(ValueError): recipe_id({'offset':float('nan')})

    def test_publication_excludes_interactive_conversations(self):
        entry={'id':'LM-test','codename':'Mischief-TEST','observed_name':'','method_label':'test','name_status':'unreviewed','naming_notes':'',
               'recipe':{},'methods':[],'observations':[{'run':'private','system':'PRIVATE CHAT CONTENT'}]}
        data={'recipes':[entry],'runs':{'private':{'experiment':'LIVE','config':{'prompt':'PRIVATE CHAT CONTENT'}}},'run_refs':{'private':{'1':'LM-test'}}}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'experiments').mkdir()
            with patch.object(catalog,'ROOT',root):catalog.publish(data)
            public=(root/'experiments/personas/registry.json').read_text(encoding='utf-8')
            self.assertNotIn('PRIVATE CHAT CONTENT',public)
            self.assertFalse(json.loads(public)['runs'])
            self.assertTrue(entry['observations'],'Publishing must not mutate the local catalog')
