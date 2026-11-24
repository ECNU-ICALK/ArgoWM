"""Catch ignored user paths, invalid GPU topology and checkpoint mix-ups."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]

class ConfigurationTests(unittest.TestCase):
    def test_released_settings_select_matching_planner_and_renderer(self):
        with tempfile.TemporaryDirectory(prefix='agrowm release ') as d:
            p=Path(d).resolve()
            self.assertTrue((ROOT/'configs/released.example.json').is_file(),
                            'A published-model settings example must ship with the code')
            settings=json.loads((ROOT/'configs/released.example.json').read_text())
            for crop in ('rice','maize'):
                self.assertEqual(settings[crop+'_planner_adapter'],f'checkpoints/{crop}/phenoclock')
                self.assertEqual(settings[crop+'_renderer_checkpoint'],
                                 f'checkpoints/{crop}/rooted_renderer/diffusion_pytorch_model.safetensors')
                settings[crop+'_planner_adapter']=str(p/'model release'/crop/'phenoclock')
                settings[crop+'_renderer_checkpoint']=str(p/'model release'/crop/'renderer.safetensors')
            settings_path=p/'settings.json';settings_path.write_text(json.dumps(settings))
            result=subprocess.run([sys.executable,str(ROOT/'scripts/configure.py'),'--settings',str(settings_path),
                                   '--output',str(p/'out')],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            from wgsr_v9.executor_inference.config import load_config,default_checkpoint
            for crop in ('rice','maize'):
                planner=json.loads((p/'out'/crop/'planner_infer.json').read_text())
                self.assertEqual(planner['adapter'],settings[crop+'_planner_adapter'])
                renderer=load_config(p/'out'/crop/'executor_infer.json')
                selected=Path(settings[crop+'_renderer_checkpoint'])
                self.assertEqual(default_checkpoint(renderer,'predicted_cot'),selected)
                self.assertEqual(default_checkpoint(renderer,'teacher_cot'),selected)
                self.assertNotEqual(default_checkpoint(renderer,'simple'),selected)
                base=load_config(p/'out'/crop/'executor_base.json')
                self.assertNotIn('checkpoints',base)
                self.assertNotIn('executor_training_root',base)

    def test_renderer_only_override_preserves_other_crop_and_base_controls(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d).resolve()
            settings_path=p/'settings.json'
            settings_path.write_text(json.dumps({'rice_renderer_checkpoint':'downloaded/rice.safetensors'}))
            result=subprocess.run([sys.executable,str(ROOT/'scripts/configure.py'),'--settings',str(settings_path),
                                   '--output',str(p/'out')],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            from wgsr_v9.executor_inference.config import load_config,default_checkpoint
            rice=load_config(p/'out/rice/executor_infer.json')
            maize=load_config(p/'out/maize/executor_infer.json')
            self.assertEqual(default_checkpoint(rice,'predicted_cot'),ROOT/'downloaded/rice.safetensors')
            self.assertEqual(default_checkpoint(maize,'predicted_cot'),
                             ROOT/'runs/maize/executor_training/teacher_cot/checkpoints/epoch-1.safetensors')

    def test_settings_relocate_both_crops_and_keep_ours_on_text_checkpoint(self):
        with tempfile.TemporaryDirectory(prefix='agrowm config ') as d:
            p=Path(d).resolve()
            settings={'data_root':str(p/'data'), 'run_root':str(p/'runs'),
                      'planner_model':str(p/'models/planner'), 'image_model':str(p/'models/renderer'),
                      'diffsynth_repo':str(p/'vendor'), 'planner_env':str(p/'p-env'),
                      'renderer_env':str(p/'r-env'), 'rice_planner_adapter':str(p/'rice-adapter'),
                      'maize_planner_adapter':str(p/'maize-adapter')}
            config=p/'settings.json'; config.write_text(json.dumps(settings))
            result=subprocess.run([sys.executable,str(ROOT/'scripts/configure.py'),'--settings',str(config),
                                   '--output',str(p/'local')],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            from wgsr_v9.executor_inference.config import load_config,default_checkpoint
            for crop,count in [('rice',2000),('maize',1600)]:
                cfg=load_config(p/'local'/crop/'executor_infer.json')
                self.assertEqual(cfg['executor_data_root'],str(p/'data'/crop/'executor'))
                self.assertEqual(default_checkpoint(cfg,'predicted_cot'),p/'runs'/crop/'executor_training/teacher_cot/checkpoints/epoch-1.safetensors')
                self.assertEqual(cfg['expected_rows'],count)
                planner=json.loads((p/'local'/crop/'planner_infer.json').read_text())
                self.assertEqual(planner['adapter'],str(p/(crop+'-adapter')))
                self.assertEqual(planner['evaluation_manifest'],str(p/'data'/crop/'evaluation/manifest.jsonl'))
                z=load_config(p/'local'/crop/'executor_base.json')
                self.assertNotIn('checkpoints',z)
                self.assertNotIn('executor_training_root',z)
            self.assertNotEqual(subprocess.run([sys.executable,str(ROOT/'scripts/configure.py'),'--settings',str(config),
                                    '--output',str(p/'local')],capture_output=True).returncode,0)

    def test_gpu_count_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d); cfg=p/'settings.json';cfg.write_text(json.dumps({'renderer_train_gpus':'0,1'}))
            result=subprocess.run([sys.executable,str(ROOT/'scripts/configure.py'),'--settings',str(cfg),'--output',str(p/'out')],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('8',result.stderr)
