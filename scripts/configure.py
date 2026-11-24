#!/usr/bin/env python3
"""Materialize both crop recipes using one local settings file; never launch jobs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PATH_KEYS = ('data_root','run_root','planner_model','image_model','diffsynth_repo',
             'planner_env','renderer_env','cache_tmpfs_root','rice_planner_adapter','maize_planner_adapter',
             'rice_renderer_checkpoint','maize_renderer_checkpoint')


def configure(settings_path: Path, output: Path) -> dict:
    output=output.resolve()
    if output.exists():
        raise FileExistsError(f'Output already exists; choose a new directory: {output}')
    settings=json.loads((ROOT/'configs/local.example.json').read_text())
    overrides=json.loads(settings_path.read_text())
    unknown=set(overrides)-set(settings)
    if unknown: raise ValueError(f'Unknown settings: {sorted(unknown)}')
    settings.update(overrides)
    for key in ('renderer_train_gpus','renderer_infer_gpus','rice_planner_gpus','maize_planner_gpus'):
        value=settings[key]
        if not isinstance(value,str) or not re.fullmatch(r'\d+(,\d+)*',value):
            raise ValueError(f'{key} must contain comma-separated GPU indices')
        ids=value.split(',')
        expected={'renderer_train_gpus':8,'rice_planner_gpus':2,'maize_planner_gpus':8}.get(key)
        if len(set(ids))!=len(ids) or (expected is not None and len(ids)!=expected):
            raise ValueError(f'{key} requires {expected or "unique"} GPU indices for this recipe')
    for key in PATH_KEYS:
        value=settings[key]
        if value is not None:
            path=Path(value).expanduser()
            settings[key]=str((ROOT/path).resolve())
        elif not key.endswith(('_adapter','_checkpoint')):
            raise ValueError(f'{key} cannot be null')
    settings['repo_root']=str(ROOT)

    def expand(value):
        if isinstance(value,dict): return {k:expand(v) for k,v in value.items()}
        if isinstance(value,list): return [expand(v) for v in value]
        if not isinstance(value,str): return value
        match=re.fullmatch(r'\$\{(\w+)\}',value)
        if match: return settings[match.group(1)]
        return re.sub(r'\$\{(\w+)\}',lambda m:str(settings[m.group(1)]),value)

    generated={}
    for source in sorted((ROOT/'configs/recipes').rglob('*.json')):
        value=expand(json.loads(source.read_text()))
        if source.stem in ('executor_infer','executor_base'):
            value['world_size']=len(settings['renderer_infer_gpus'].split(','))
        if source.stem == 'executor_infer':
            checkpoint=settings[source.parent.name+'_renderer_checkpoint']
            if checkpoint is not None:
                # Ours and Tutor share the text Renderer; Simple stays independent.
                value['checkpoints']['teacher_cot']=checkpoint
        generated[source.relative_to(ROOT/'configs/recipes').as_posix()]=value
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='agrowm-config-',dir=output.parent) as d:
        stage=Path(d)/'config';stage.mkdir()
        for name,value in generated.items():
            path=stage/name;path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps(value,indent=2)+'\n')
        if output.exists(): raise FileExistsError(output)
        stage.rename(output)
    return {'status':'complete','output':str(output),'configs':sorted(generated),
            'missing_adapters':[c for c in ('rice','maize') if settings[c+'_planner_adapter'] is None]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=ROOT/'configs/local')
    args=parser.parse_args()
    print(json.dumps(configure(args.settings,args.output),indent=2))

if __name__=='__main__': main()
