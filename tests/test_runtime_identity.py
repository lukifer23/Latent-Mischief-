from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from runtime_identity import inference_lock_identity


class RuntimeIdentityTests(unittest.TestCase):
    def test_optional_tools_do_not_mask_a_real_inference_dependency_change(self):
        lock=b'''version = 1
requires-python = ">=3.11,<3.12"
[[package]]
name = "latent-mischief"
version = "0.1"
source = { virtual = "." }
dependencies = [{name="torch"}]
[[package]]
name = "torch"
version = "2.14.1"
source = {registry="official"}
dependencies = [{name="sympy"}]
[[package]]
name = "sympy"
version = "1.14"
source = {registry="official"}
'''
        extra=b'''\n[[package]]
name = "optional-grader"
version = "0.9"
source = {registry="official"}
'''
        original=inference_lock_identity(lock)['sha256']
        self.assertEqual(original,inference_lock_identity(lock+extra)['sha256'])
        self.assertNotEqual(original,inference_lock_identity(lock.replace(b'2.14.1',b'2.14.2'))['sha256'])
        self.assertNotEqual(original,inference_lock_identity(lock.replace(b'1.14',b'1.15'))['sha256'])
