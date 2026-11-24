#!/usr/bin/env python3
"""Prepare portable, frozen rice/maize pairs for Planner, Renderer and evaluation.

This CPU-only step preserves split membership, row order, teachers and seeds.
It never generates annotations, changes image geometry, or calls an API.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys
import tempfile

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from wgsr_v9.io import read_jsonl, sha256_path, write_json, write_jsonl
from wgsr_v9.executor_training import (
    NAMED_WEEKS_PROTOCOL_VERSION, executor_prompt, growth_cot_body, prediction_interval_weeks,
)
from wgsr_v9.planner_direct import PROTOCOL_VERSION as RICE_PLANNER, validate_direct_body
from wgsr_v9.maize import experiment as maize

ALGORITHM = 'agrowm_portable_data_v1'
RICE_PROTOCOL = ROOT / 'protocols/qwen_image21_executor_v9_v3_metric_named_weeks'
RICE_PLANNER_ROOT = ROOT / 'protocols/planner_v3_direct_metric_named_weeks'
MAIZE_TEACHER = ROOT / 'protocols/maize_sv1_v1/cot_v2_en_v2'


def prepare(crop: str, pairs: Path, output: Path, *, paper_split: bool = False) -> dict:
    pairs, output = pairs.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(f'Output already exists; choose a new directory: {output}')
    rows = read_jsonl(pairs)
    if not rows:
        raise ValueError('Pairs manifest is empty')
    required = ('pair_id', 'pair_group_id', 'plant_id', 'split', 'view', 'actual_interval_days',
                'source_image_path', 'target_image_path',
                'source_image_id', 'target_image_id', 'seed')
    planner = {s: [] for s in ('train', 'val')}
    executor = {v: {s: [] for s in ('train', 'val')} for v in ('simple', 'teacher_cot')}
    evaluation, seeds, copies = [], [], []
    plants = {s: set() for s in planner}
    ids, groups = set(), {s: set() for s in planner}
    image_checks, metadata_by_image = {}, {}
    protocol = RICE_PLANNER if crop == 'rice' else maize.PLANNER_PROTOCOL
    executor_protocol = NAMED_WEEKS_PROTOCOL_VERSION if crop == 'rice' else maize.EXECUTOR_PROTOCOL

    def provenance(src: str, dest: Path, version: str) -> dict:
        return dict(algorithm_version=ALGORITHM, protocol_version=version, input_path=src,
                    output_path=str(dest), status='ready', error=None, retry_count=0)

    for row in rows:
        missing = [key for key in required if row.get(key) is None or row.get(key) == '']
        if missing:
            raise ValueError(f'Missing required fields: {missing}')
        key, split = row['pair_id'], row['split']
        if not isinstance(key, str) or key in ids:
            raise ValueError(f'Duplicate or invalid pair_id: {key}')
        ids.add(key)
        if split not in planner:
            raise ValueError(f'Unsupported split {split!r}; use train or val')
        if crop == 'maize' and split == 'val' and row.get('horizon_bucket') not in ('short', 'mid', 'long'):
            raise ValueError('Maize validation requires original horizon_bucket: short, mid or long')
        days = row['actual_interval_days']
        if isinstance(days, bool) or not isinstance(days, int) or days <= 0:
            raise ValueError('actual_interval_days must be a positive integer')
        if isinstance(row['seed'], bool) or not isinstance(row['seed'], int) or row['seed'] < 0:
            raise ValueError('seed must be a nonnegative integer')
        view = row['view']
        if crop == 'rice':
            if view not in ('front', 'side0', 'top'):
                raise ValueError(f'Unsupported rice view: {view}')
            if not re.fullmatch(r'[A-Za-z0-9_-]+', str(row.get('accession_id', ''))):
                raise ValueError('Rice requires a valid accession_id')
            size = (896, 672) if view == 'top' else (672, 896)
        else:
            if view not in ('sv1', 'side0'):
                raise ValueError(f'Unsupported maize view: {view}')
            view, size = 'side0', maize.SIZE
        plants[split].add(row['plant_id']); groups[split].add(row['pair_group_id'])
        paths = {}
        for side in ('source', 'target'):
            p = (pairs.parent / row[f'{side}_image_path']).resolve(strict=True)
            if p not in image_checks:
                with Image.open(p) as im:
                    im.load()  # Decode now: valid headers do not imply intact image data.
                    image_checks[p] = (im.size, im.mode, sha256_path(p))
            if image_checks[p][:2] != (size, 'RGB'):
                raise ValueError(f'{key}: expected RGB {size} image: {p}')
            stage = row.get(f'{side}_stage') or row.get(f'{side}_stage_id')
            if not isinstance(stage, str) or not stage.strip():
                raise ValueError(f'{key}: missing {side}_stage evaluation label')
            identity = (row['plant_id'], view, stage, row[f'{side}_image_id'],
                        row.get(f'{side}_capture_date'))
            if p in metadata_by_image and metadata_by_image[p] != identity:
                raise ValueError(f'Conflicting plant/view/stage/image metadata for {p}')
            metadata_by_image[p] = identity
            paths[side] = str(p)
        cot = row.get('teacher_cot')
        if not cot and row.get('teacher_cot_path'):
            cot = (pairs.parent / row['teacher_cot_path']).read_text()
        if not isinstance(cot, str):
            raise ValueError(f'{key}: teacher_cot or teacher_cot_path is required')
        cot = cot.strip()
        index = len(planner[split])
        meta = dict(request_id=key, pair_id=key, pair_group_id=row['pair_group_id'],
                    plant_id=row['plant_id'], split=split, species=crop, view=view,
                    actual_interval_days=days, source_size=list(size), original_row_index=index,
                    time_condition_unit='weeks' if crop == 'rice' else 'days')
        if crop == 'rice':
            body = growth_cot_body(cot, key)
            validate_direct_body(body, view)
            meta.update(accession_id=row['accession_id'], prediction_interval_weeks=prediction_interval_weeks(row),
                        numeric_evidence_format='metric_named_v1')
            common = (RICE_PLANNER_ROOT / 'SYSTEM_COMMON.txt').read_text().strip()
            normalized = 'top' if view == 'top' else 'front'
            system = common + '\n\n' + (RICE_PLANNER_ROOT / f'SYSTEM_{normalized.upper()}.txt').read_text().strip()
            user = (RICE_PLANNER_ROOT / 'USER_TEMPLATE.txt').read_text().strip().format(
                accession_id=meta['accession_id'], view_label=normalized+' view',
                prediction_interval_weeks=meta['prediction_interval_weeks'])
            messages = [dict(role='system', content=system), dict(role='user', content=user)]
            prompts = {v: executor_prompt(meta,
                prompt_template=(RICE_PROTOCOL / 'EXECUTOR_PROMPT_TEMPLATE.txt').read_text(),
                cot_suffix=(RICE_PROTOCOL / 'TEACHER_COT_SUFFIX.txt').read_text(), variant=v,
                teacher_cot=body if v == 'teacher_cot' else None, protocol_version=executor_protocol)
                for v in executor}
        else:
            maize.validate_growth_cot(cot)
            meta.update(native_view='sv1', numeric_evidence_format='maize_sv1_v1')
            messages = maize.planner_messages(meta, MAIZE_TEACHER)
            prompts = {v: maize.executor_prompt(meta, cot if v == 'teacher_cot' else None) for v in executor}
        planner[split].append({**provenance(paths['source'], output/'planner'/f'{split}.jsonl', protocol),
            'id': key, 'messages': [*messages, dict(role='assistant', content=cot)],
            'images': [paths['source']], 'planner_metadata': meta})
        for variant in executor:
            executor[variant][split].append({**provenance(paths['source'], Path(paths['target']), executor_protocol),
                **meta, 'image': paths['target'], 'edit_image': paths['source'], 'prompt': prompts[variant],
                'executor_variant': variant, 'cot_source': 'teacher_annotation' if variant == 'teacher_cot' else None})
        if split == 'val':
            ev = {**provenance(paths['source'], output/'evaluation/manifest.jsonl', protocol), **meta,
                  'row_index': index, 'seed': row['seed'], 'target_usage': 'eval_reference_only'}
            for side in ('source', 'target'):
                for field in ('stage_id', 'image_id', 'capture_date'):
                    if f'{side}_{field}' in row: ev[f'{side}_{field}'] = row[f'{side}_{field}']
                ev[f'{side}_stage'] = row.get(f'{side}_stage') or row.get(f'{side}_stage_id')
                ev[f'{side}_image_path'] = paths[side]
                ev[f'{side}_image_sha256'] = image_checks[Path(paths[side])][2]
            if 'horizon_bucket' in row: ev['horizon_bucket'] = row['horizon_bucket']
            evaluation.append(ev)
            seeds.append({'pair_id': key, 'seed': row['seed'], **provenance(paths['source'], output/'evaluation/seeds.jsonl', protocol)})
            copies.append({**ev, 'status': 'completed', 'model': 'source_copy', 'output_path': paths['source']})
    if plants['train'] & plants['val'] or groups['train'] & groups['val']:
        raise ValueError('Train/val plant or pair-group overlap detected')
    counts = {s: len(v) for s, v in planner.items()}
    if not all(counts.values()):
        raise ValueError('Both train and val splits must be nonempty')
    if paper_split:
        expected = {'train': 20000, 'val': 2000} if crop == 'rice' else {'train': 6400, 'val': 1600}
        expected_plants = 79 if crop == 'rice' else 84
        if counts != expected or len(plants['val']) != expected_plants:
            raise ValueError(f'Paper split mismatch: {counts}, {len(plants["val"])} validation plants')
        if crop == 'rice':
            pair_views = {}
            for row in rows:
                pair_views.setdefault((row['split'], row['pair_group_id']), []).append(
                    'top' if row['view'] == 'top' else 'front')
            if any(sorted(v) != ['front', 'top'] for v in pair_views.values()):
                raise ValueError('Paper rice split requires one front and one top query per pair group')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='agrowm-prepare-', dir=output.parent) as staging:
        temp = Path(staging)/'package'; temp.mkdir()
        for split, records in planner.items(): write_jsonl(temp/'planner'/f'{split}.jsonl', records)
        manifest = dict(algorithm_version=ALGORITHM, protocol_version=protocol, status='complete',
                        planner_variant='proposed_cot', active_no_cot_planner=False, cot_version='v3',
                        numeric_evidence_format='metric_named_v1' if crop == 'rice' else 'maize_sv1_v1',
                        time_condition_unit='weeks' if crop == 'rice' else 'days', enable_thinking=False,
                        output_wrapper='GROWTH_COT', rows_by_split=counts, source_image_only=True,
                        input_path=str(pairs), output_path=str(output/'planner'), error=None, retry_count=0)
        write_json(temp/'planner/package_manifest.json', manifest)
        for variant, splits in executor.items():
            for split, records in splits.items(): write_jsonl(temp/'executor'/variant/f'{split}.jsonl', records)
        write_json(temp/'executor/package_manifest.json', {
            **provenance(str(pairs), output/'executor', executor_protocol),
            'algorithm_version': 'wgsr_v9_qwen_image21_executor_sft_v1',
            'status': 'complete', 'species': crop,
            'variants': {variant: {'status': 'complete', 'rows_by_split': counts,
                'generated_file_sha256': {f'{split}.jsonl': sha256_path(temp/'executor'/variant/f'{split}.jsonl')
                                           for split in ('train', 'val')}}
                         for variant in executor}})
        for name, records in [('manifest',evaluation),('seeds',seeds),('source_copy/results',copies)]:
            write_jsonl(temp/'evaluation'/f'{name}.jsonl',records)
        summary = dict(algorithm_version=ALGORITHM, crop=crop, rows_by_split=counts,
                       plants_by_split={s:len(p) for s,p in plants.items()}, paper_split_checked=paper_split,
                       input_path=str(pairs), output_path=str(output), input_sha256=sha256_path(pairs),
                       status='complete', error=None, retry_count=0,
                       generated_file_sha256={p.relative_to(temp).as_posix():sha256_path(p) for p in sorted(temp.rglob('*.json*'))})
        write_json(temp/'package_manifest.json',summary)
        if output.exists(): raise FileExistsError(output)
        temp.rename(output)
    return summary


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--crop', choices=['rice','maize'], required=True)
    parser.add_argument('--pairs',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--paper-split',action='store_true',help='Require the published split counts and validation plant counts.')
    args=parser.parse_args()
    print(json.dumps(prepare(args.crop,args.pairs,args.output,paper_split=args.paper_split),indent=2))

if __name__=='__main__': main()
