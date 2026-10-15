"""Focused CPU regressions for static CLI inspection and freeze lineage."""
import copy
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from . import cli_preflight as P, study as S, submit_once as U, guard as G

REGISTERED = '''class EngineArgs:
    def add_cli_args(parser):
        attention_group.add_argument("--attention-backend", **attention_kwargs["backend"])
'''


class StaticCliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.exe = self.root/'bin/vllm'
        self.python = self.root/'bin/python'
        self.exe.parent.mkdir()
        self.exe.write_text('#!' + str(self.python) + '\nfrom vllm.entrypoints.cli.main import main\nmain()\n')
        self.exe.chmod(0o755)
        self.package = self.root/'site-packages'
        for name, text in [('vllm/engine/arg_utils.py', REGISTERED),
                           ('vllm/entrypoints/cli/main.py', 'def main(): pass\n'),
                           ('vllm/__init__.py', 'raise RuntimeError("must not import")\n')]:
            path = self.package/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        self.dist = SimpleNamespace(version='0.31.0', entry_points=[SimpleNamespace(
            name='vllm', group='console_scripts', value=P.ENTRYPOINT)],
            locate_file=lambda rel: self.package/rel)

    def inspect(self):
        with patch.object(P.metadata, 'distribution', return_value=self.dist), \
             patch.object(P.sys, 'prefix', str(self.root)), patch.object(P.sys, 'executable', str(self.python)):
            return P.inspect_cli(self.exe)

    def test_valid_static_inspection_never_imports_package(self):
        # A real import would execute the sentinel in the synthetic package.
        with patch.object(P.sys, 'path', [str(self.package), *P.sys.path]):
            result = self.inspect()
        self.assertEqual(result['status'], 'passed')
        self.assertFalse(result['imports_vllm'])
        self.assertEqual(len(result['source_sha256']), 2)

    def test_unreviewed_version_and_entrypoint_rejected(self):
        self.dist.version = '0.32.0'
        with self.assertRaisesRegex(ValueError, 'version'):
            self.inspect()
        self.dist.version = '0.31.0'
        self.dist.entry_points[0].value = 'other:main'
        with self.assertRaisesRegex(ValueError, 'entrypoint'):
            self.inspect()

    def test_executable_interpreter_and_permissions_checked(self):
        self.exe.write_text('#!/usr/bin/python\nfrom vllm.entrypoints.cli.main import main\n')
        with self.assertRaisesRegex(ValueError, 'interpreter'):
            self.inspect()
        with patch.object(P.os, 'access', return_value=False), self.assertRaisesRegex(ValueError, 'unavailable'):
            self.inspect()
        self.exe.unlink()
        with self.assertRaisesRegex(ValueError, 'unavailable'):
            self.inspect()

    def test_mentions_wrong_scope_dead_code_and_wrong_binding_rejected(self):
        invalid = [REGISTERED.replace('attention_group.add_argument', '# attention_group.add_argument') + '        pass\n',
                   REGISTERED.replace('EngineArgs', 'Other'),
                   REGISTERED.replace('        attention_group', '        if False:\n            attention_group'),
                   REGISTERED.replace('"backend"', '"other"'),
                   REGISTERED.replace('add_argument', 'remove_argument')]
        for text in invalid:
            with self.subTest(text=text), self.assertRaises(ValueError):
                P.registered_option(text)

    def test_missing_and_oversized_source_rejected(self):
        path = self.package/'vllm/engine/arg_utils.py'
        path.write_bytes(b' ' * (P.MAX_SOURCE_BYTES + 1))
        with self.assertRaisesRegex(ValueError, 'oversized'):
            self.inspect()
        path.unlink()
        with self.assertRaisesRegex(ValueError, 'missing'):
            self.inspect()

    def test_source_outside_runtime_rejected(self):
        self.dist.locate_file = lambda rel: self.root.parent/'outside.py'
        with self.assertRaisesRegex(ValueError, 'outside'):
            self.inspect()

    def test_failed_or_timed_out_static_probe_never_reaches_slurm(self):
        approved = {'code_commit': 'a'*40, 'authorization_id': 'unit-test'}
        args = ['submit', '--bundle', str(self.root), '--freeze-sha256', 'f',
                '--authorization-file', 'approval', '--private-root', str(self.root/'private'),
                '--runtime-python', str(Path.home()/'code/ar_harness/conda_env/bin/python'),
                '--cache', str(Path.home()/'orcd/pool/ar_harness/hf_cache')]
        for error in [subprocess.TimeoutExpired('probe', 30), subprocess.CalledProcessError(1, 'probe')]:
            def run(cmd, **kwargs):
                self.assertNotEqual(cmd[0], 'sbatch')
                if 'cli_preflight.py' in ' '.join(cmd):
                    self.assertEqual(kwargs['timeout'], 30)
                    self.assertIn('-I', cmd)
                    raise error
            with patch('sys.argv', args), patch.object(U, 'authorization', return_value=approved), \
                 patch.object(U, 'verify'), patch.object(U.subprocess, 'check_output', side_effect=['a'*40, '']), \
                 patch.object(U.subprocess, 'run', side_effect=run) as calls, \
                 patch.object(G, 'LEDGER_ROOT', self.root/'ledger'):
                with self.assertRaises(type(error)):
                    U.main()
                self.assertEqual(calls.call_count, 3)
                self.assertFalse((G.ledger(approved)/'SUBMISSION_ATTEMPTED.json').exists())


class AmendmentTests(unittest.TestCase):
    def document(self):
        original = S.C.read_json(S.HERE/'freeze.json')
        amended = dict(original, format='c2_confirmatory_freeze_v2', parent_freeze_sha256=S.PARENT_SHA)
        amended['files'] = dict(original['files'])
        return amended

    def test_original_freeze_and_protected_file_changes_rejected(self):
        amended = self.document()
        S.check_amendment(amended)
        for rel in set(amended['files']) - S.AMENDED_FILES:
            changed = copy.deepcopy(amended)
            changed['files'][rel] = '0'*64
            with self.subTest(path=rel), self.assertRaisesRegex(ValueError, 'protected'):
                S.check_amendment(changed)
        with patch.object(S, 'file_sha', return_value='0'*64), self.assertRaisesRegex(ValueError, 'original freeze'):
            S.check_amendment(amended)

    def test_design_metadata_and_parent_identity_cannot_change(self):
        for key in ['manifest_sha256', 'budget_sha256', 'targets_sha256', 'rubric_sha256',
                    'reference_bindings', 'parent_freeze_sha256']:
            amended = self.document()
            amended[key] = 'changed'
            with self.subTest(key=key), self.assertRaises(ValueError):
                S.check_amendment(amended)

    def test_old_authorization_cannot_authorize_amended_freeze(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            S.C.write_json(root/S.FREEZE_NAME, {'new': True})
            S.C.write_json(root/'approval.json', {'approved': True, 'freeze_sha256': S.PARENT_SHA})
            with patch.object(S, 'HERE', root), self.assertRaisesRegex(ValueError, 'another freeze'):
                S.authorization(root/'approval.json', S.PARENT_SHA)


if __name__ == '__main__':
    unittest.main()
