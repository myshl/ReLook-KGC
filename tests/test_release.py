import contextlib
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('paper_run', ROOT / 'scripts/paper_run.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ReleaseTests(unittest.TestCase):
    def test_historical_source_checksums(self):
        with contextlib.redirect_stdout(io.StringIO()):
            module.verify(SimpleNamespace(assets=None, dataset=None))

    def dry_command(self, dataset, stage, previous=Path('/example/incoming.ckpt')):
        capture = io.StringIO()
        args = SimpleNamespace(dataset=dataset, stage=stage, previous=previous,
                               assets=Path('/example/assets'), gpu=0,
                               output=Path('/example/output'), execute=False)
        with contextlib.redirect_stdout(capture):
            module.stage(args)
        return json.loads(capture.getvalue())

    def test_all_archived_training_commands_are_portable(self):
        for file in sorted((ROOT / 'configs').glob('*/*/command.json')):
            dataset, stage = file.parent.parent.name, file.parent.name
            with self.subTest(dataset=dataset, stage=stage):
                command = self.dry_command(dataset, stage)
                self.assertFalse(any('${' in item for item in command))
                expected = 'db15k' if dataset == 'DB15K' or stage == 'csprom' else 'localplus'
                self.assertEqual(command[1], str(ROOT / 'versions' / expected / 'main.py'))
                self.assertEqual(command[command.index('-dataset') + 1], dataset)
                self.assertEqual(command[command.index('-eval_split') + 1], 'valid')

    def test_missing_incoming_checkpoint_fails(self):
        with self.assertRaisesRegex(ValueError, 'requires --previous'):
            self.dry_command('YAGO15K', 'lar_refine', previous=None)

    def test_paper_deployment_not_new_rerun(self):
        manifest = module.read_json(ROOT / 'release_manifest.json')
        expected = [('DB15K', .5, 0., 'joint'), ('YAGO15K', .5, 0., 'visual'),
                    ('FB15k-237', .25, .1, 'prior')]
        for dataset, blend, threshold, stage in expected:
            policy = manifest['datasets'][dataset]['deployment']
            self.assertEqual((policy['blend'], policy['threshold'], policy['stage']),
                             (blend, threshold, stage))
        self.assertIn('argsort', manifest['evaluation'])


if __name__ == '__main__':
    unittest.main()
