"""Exercise text inference without optional injection modules or a GPU backend."""
import builtins
from contextlib import nullcontext, redirect_stdout
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
import warnings
from unittest.mock import patch

from PIL import Image

from scripts.configure import configure
from scripts.prepare_data import prepare
from tests.test_release_data import make_pairs
from wgsr_v9.executor_launcher import build_plan
from wgsr_v9.executor_inference.cli import parse_args
from wgsr_v9.executor_inference.config import validate_config
from wgsr_v9.executor_inference import runner
from wgsr_v9.io import read_jsonl, write_json, write_jsonl, sha256_path, canonical_sha256

ROOT = Path(__file__).resolve().parents[1]


class CpuGenerator:
    def __init__(self, *, device):
        self.device = device

    def manual_seed(self, seed):
        self.seed = seed
        return self


class TextExecutorReleaseTests(unittest.TestCase):
    def test_numeric_training_config_cannot_silently_run_text_training(self):
        config = json.loads((ROOT / 'configs/recipes/rice/executor_teacher_cot.json').read_text())
        config['numeric_condition'] = {'enabled': True}
        try:
            with self.assertRaisesRegex(ValueError, 'Numeric injection is not included'):
                build_plan(config, 'teacher_cot', 'train')
        except FileNotFoundError:
            self.fail('Numeric config reached model loading instead of being rejected')

    def test_numeric_inference_config_cannot_silently_run_text_inference(self):
        config = json.loads((ROOT / 'configs/recipes/rice/executor_infer.json').read_text())
        config.update(world_size=1, physical_gpu_ids='0')
        for field, value in (
            ('numeric_bundle', 'old-bundle'),
            ('numeric_condition', {'enabled': True}),
            ('condition_dropout', 'drop_numeric'),
        ):
            with self.subTest(field=field):
                changed = {**config, field: value}
                with self.assertRaisesRegex(ValueError, 'Numeric injection is not included'):
                    validate_config(changed)

    def test_mainline_workers_and_completed_resume_without_numeric_module(self):
        # Only the GPU/model boundary is substituted. Preparation, CLI, manifest
        # validation, image I/O, ledgers and resume checks execute normally.
        original_import = builtins.__import__

        def without_numeric(name, *args, **kwargs):
            if 'executor_numeric' in name.split('.'):
                raise ModuleNotFoundError('optional injection module is unavailable')
            return original_import(name, *args, **kwargs)

        torch_backend = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: True, set_device=lambda _: None),
            Generator=CpuGenerator, inference_mode=nullcontext,
        )
        for crop, view in (('rice', 'front'), ('rice', 'top'), ('maize', 'sv1')):
            with self.subTest(crop=crop, view=view), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                pairs = make_pairs(root, crop)
                size = (672, 896) if crop == 'rice' else (768, 1280)
                if view == 'top':
                    size = (896, 672)
                    for row in pairs:
                        row['view'] = 'top'
                        for role in ('source', 'target'):
                            Image.new('RGB', size, 'white').save(root / row[role + '_image_path'])
                        row['teacher_cot'] = '\n'.join(
                            line.replace('Plant density:', 'Canopy:').replace('Metric = Solidity;', 'Metric = Canopy density;')
                            for line in row['teacher_cot'].splitlines() if not line.startswith('Culm:'))
                write_jsonl(root / 'pairs.jsonl', pairs)
                package = root / 'data' / crop
                prepare(crop, root / 'pairs.jsonl', package)
                write_json(root / 'settings.json', {'data_root': str(root / 'data'), 'run_root': str(root / 'runs')})
                configure(root / 'settings.json', root / 'configs')
                model = root / 'model'
                (model / 'transformer').mkdir(parents=True)
                write_json(model / 'transformer/config.json', {})
                (model / 'transformer/diffusion_pytorch_model.safetensors').write_bytes(b'base-test-weights')
                checkpoint = root / 'text.safetensors'
                checkpoint.write_bytes(b'text-test-weights')
                predicted = copy.deepcopy(read_jsonl(package / 'executor/teacher_cot/val.jsonl')[0])
                predicted.update(executor_variant='predicted_cot', cot_source='planner_prediction', adapter_sha256='test-planner')
                predicted['prompt'] = predicted['prompt'].replace('Taller culms.', 'Predicted taller culms.').replace('The visible plant develops.', 'Predicted plant development.')
                write_jsonl(root / 'predicted.jsonl', [predicted])
                # A missing target must not prevent source-only image generation.
                (root / 'val_target.png').unlink()
                for variant, pretrained in (('simple', False), ('teacher_cot', False), ('predicted_cot', False),
                                            ('simple', True), ('predicted_cot', True)):
                    with self.subTest(variant=variant, pretrained=pretrained):
                        config_path = root / 'configs' / crop / ('executor_base.json' if pretrained else 'executor_infer.json')
                        config = json.loads(config_path.read_text())
                        config['expected_rows'] = 1
                        write_json(config_path, config)
                        output = root / (variant + ('_base' if pretrained else '_trained'))
                        command = ['infer', '--config', str(config_path), '--variant', variant,
                                   '--model-path', str(model), '--output-root', str(output), '--world-size', '1', '--gpu-ids', '0']
                        if variant == 'predicted_cot':
                            command += ['--input-manifest', str(root / 'predicted.jsonl')]
                        if not pretrained:
                            command += ['--checkpoint', str(checkpoint)]
                        else:
                            command += ['--pretrained']
                        with patch.object(sys, 'argv', command):
                            args = parse_args()
                        input_rows, input_sha = runner.load_input_rows(args.input_manifest, None)
                        expected_prompt = input_rows[0]['prompt']
                        calls = []

                        def image_backend(**kwargs):
                            calls.append(kwargs)
                            return SimpleNamespace(images=[Image.new('RGB', (kwargs['width'], kwargs['height']),
                                                                     (kwargs['generator'].seed % 256, 0, 0))])

                        def worker_process(command, **kwargs):
                            with patch.object(sys, 'argv', command[1:]):
                                worker_args = parse_args()
                            status = runner.worker_main(worker_args)
                            return SimpleNamespace(pid=1234, poll=lambda: status)

                        with patch.object(builtins, '__import__', without_numeric), \
                             patch.dict(sys.modules, {'torch': torch_backend}), \
                             patch.object(runner, 'load_pipeline', return_value=(image_backend, {}, 'bf16')), \
                             patch.object(runner, 'nvidia_memory', return_value={'0': 0}), \
                             patch.object(runner.subprocess, 'Popen', side_effect=worker_process), \
                             warnings.catch_warnings(record=True) as notices, \
                             redirect_stdout(io.StringIO()):
                            warnings.simplefilter('always', ResourceWarning)
                            try:
                                self.assertEqual(runner.supervisor_main(args), 0)
                            except ModuleNotFoundError as exc:
                                self.fail(f'Text worker depends on an optional injection module: {exc}')
                            self.assertEqual(len(calls), 1)
                            self.assertEqual(calls[0]['prompt'], expected_prompt)
                            self.assertEqual(calls[0]['image'].size, size)
                            self.assertEqual(calls[0]['num_inference_steps'], 20)
                            self.assertNotIn('attention_kwargs', calls[0])
                            result = read_jsonl(output / 'rank-00/results.jsonl')[0]
                            self.assertEqual(result['seed'], 917)
                            self.assertEqual(result['output_size'], list(size))
                            self.assertEqual(result['prompt'], expected_prompt)
                            with Image.open(result['output_path']) as generated:
                                self.assertEqual(generated.getpixel((0, 0)), (149, 0, 0))
                            # Mainline identities written before module removal include this null field.
                            previous_identity = {
                                'variant': variant, 'input_manifest_sha256': input_sha,
                                'checkpoint_sha256': None if pretrained else sha256_path(checkpoint),
                                **({'pretrained_transformer': runner.pretrained_transformer_identity(model)} if pretrained else {}),
                                'model_path': str(model), 'selected_indices': [0], 'steps': 20,
                                'dtype': args.dtype, 'use_kv_cache': args.use_kv_cache,
                                'seed_base': args.seed_base, 'seed_manifest_sha256': sha256_path(args.seed_manifest),
                                'numeric_condition': None,
                                'executor_prompt_protocol_version': config['executor_prompt_protocol_version'],
                            }
                            self.assertEqual(result['run_identity_sha256'], canonical_sha256(previous_identity))
                            write_json(output / 'inference_manifest.json', {'status': 'complete', 'run_identity': previous_identity})
                            self.assertEqual(runner.supervisor_main(args), 0)
                            self.assertEqual(runner.worker_main(args), 0)
                            self.assertEqual(len(calls), 1)
                            self.assertEqual(len(read_jsonl(output / 'rank-00/results.jsonl')), 1)
                            self.assertFalse([n for n in notices if issubclass(n.category, ResourceWarning)])
