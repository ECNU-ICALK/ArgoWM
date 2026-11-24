"""Catch source/target leakage, lost seeds, split mixing and nonportable image paths."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def make_pairs(root, crop):
    size = (672, 896) if crop == 'rice' else (768, 1280)
    cot = (ROOT / 'tests/fixtures' / ('rice_front_cot.txt' if crop == 'rice' else 'maize_cot.txt')).read_text().strip()
    rows = []
    for split, plant, seed in [('train', 'plantA', 13), ('val', 'plantB', 917)]:
        Image.new('RGB', size, 'white').save(root / (split+'_source.png'))
        Image.new('RGB', size, 'green').save(root / (split+'_target.png'))
        rows.append(dict(pair_id=split, pair_group_id=split, plant_id=plant,
                         accession_id='A1', split=split, view='front' if crop == 'rice' else 'sv1',
                         actual_interval_days=18, source_image_path=split+'_source.png', target_image_path=split+'_target.png',
                         source_stage_id='S1_tillering' if crop == 'rice' else 'S2',
                         target_stage_id='S2_jointing' if crop == 'rice' else 'S9',
                         source_image_id=plant+'_source', target_image_id=plant+'_target',
                         seed=seed, teacher_cot=cot, horizon_bucket='mid'))
    return rows


class ReleaseDataTests(unittest.TestCase):
    def run_prepare(self, root, crop, rows):
        path = root / 'pairs.jsonl'
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        return subprocess.run([sys.executable, str(ROOT/'scripts/prepare_data.py'), '--crop', crop,
                               '--pairs', str(path), '--output', str(root/'prepared')],
                              capture_output=True, text=True)

    def test_round_trip_both_crops_preserves_teacher_seeds_and_source_only_inputs(self):
        for crop in ('rice', 'maize'):
            with self.subTest(crop=crop), tempfile.TemporaryDirectory(prefix='agrowm space ') as d:
                root=Path(d).resolve(); rows=make_pairs(root,crop)
                result=self.run_prepare(root,crop,rows)
                self.assertEqual(result.returncode,0,result.stderr)
                p=root/'prepared'
                planner=json.loads((p/'planner/val.jsonl').read_text())
                self.assertEqual(planner['messages'][2]['content'],rows[1]['teacher_cot'])
                inputs=planner['messages'][:2]
                self.assertNotIn('target.png',json.dumps(inputs))
                self.assertNotIn(rows[1]['teacher_cot'],json.dumps(inputs))
                self.assertEqual(planner['images'],[str(root/'val_source.png')])
                user=inputs[1]['content']
                self.assertIn('3 weeks' if crop=='rice' else '18 days',user)
                self.assertNotIn('18 days' if crop=='rice' else '3 weeks',user)
                ev=json.loads((p/'evaluation/manifest.jsonl').read_text())
                self.assertEqual(ev['seed'],917)
                self.assertEqual(ev['target_stage'],rows[1]['target_stage_id'])
                self.assertEqual(ev['target_image_path'],str(root/'val_target.png'))
                manifest=json.loads((p/'executor/package_manifest.json').read_text())
                self.assertEqual(manifest['algorithm_version'],'wgsr_v9_qwen_image21_executor_sft_v1')
                self.assertEqual(manifest['variants']['simple']['rows_by_split'],{'train':1,'val':1})
                simple=json.loads((p/'executor/simple/val.jsonl').read_text())
                self.assertNotIn('GROWTH_COT',simple['prompt'])
                from wgsr_v9.executor_inference.manifest import validate_prompt_rows
                for variant in ('simple','teacher_cot'):
                    r=json.loads((p/f'executor/{variant}/val.jsonl').read_text())
                    validate_prompt_rows([r],variant,expected_protocol=r['protocol_version'])
                self.assertNotEqual(self.run_prepare(root,crop,rows).returncode,0)

    def test_plant_overlap_is_rejected_before_writing_package(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d).resolve();rows=make_pairs(root,'maize'); rows[1]['plant_id']='plantA'
            result=self.run_prepare(root,'maize',rows)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('overlap',result.stderr.lower())
            self.assertFalse((root/'prepared/package_manifest.json').exists())

    def test_missing_seed_is_rejected_instead_of_changing_comparison(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d).resolve();rows=make_pairs(root,'rice'); del rows[1]['seed']
            result=self.run_prepare(root,'rice',rows)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('seed',result.stderr.lower())

    def test_unknown_crop_view_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d).resolve();rows=make_pairs(root,'maize'); rows[1]['view']='top'
            result=self.run_prepare(root,'maize',rows)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('view',result.stderr.lower())

    def test_evaluation_stage_labels_are_preserved_separately_from_script_stages(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d).resolve();rows=make_pairs(root,'rice')
            rows[1].update(source_stage='Tillering',target_stage='Grain filling')
            result=self.run_prepare(root,'rice',rows)
            self.assertEqual(result.returncode,0,result.stderr)
            ev=json.loads((root/'prepared/evaluation/manifest.jsonl').read_text())
            self.assertEqual(ev['target_stage'],'Grain filling')

    def test_truncated_image_is_rejected_before_writing_package(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d).resolve();rows=make_pairs(root,'rice')
            source=root/'val_source.png'
            source.write_bytes(source.read_bytes()[:100])
            result=self.run_prepare(root,'rice',rows)
            self.assertNotEqual(result.returncode,0)
            self.assertFalse((root/'prepared/package_manifest.json').exists())

    def test_maize_requires_original_horizon_bucket(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d).resolve();rows=make_pairs(root,'maize')
            for row in rows: row.pop('horizon_bucket',None)
            result=self.run_prepare(root,'maize',rows)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('horizon_bucket',result.stderr)

    def test_prepared_maize_source_copy_completes_phenotype_evaluation(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d).resolve();rows=make_pairs(root,'maize')
            result=self.run_prepare(root,'maize',rows)
            self.assertEqual(result.returncode,0,result.stderr)
            package=root/'prepared'
            result=subprocess.run([sys.executable,str(ROOT/'scripts/evaluate_maize_phenotypes.py'),
                '--dataset-manifest',str(package/'evaluation/manifest.jsonl'),
                '--result','source_copy='+str(package/'evaluation/source_copy'),
                '--output-dir',str(root/'phenotypes'),'--workers','1'],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertTrue((root/'phenotypes/summary.json').is_file())
