#!/usr/bin/env python3
"""Standard-library integrity check for the public UI-KB release."""
import json, hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CORPUS=ROOT/'data/corpus'
GT=ROOT/'data/ground_truth'
EXPECTED_COUNTS={'dev_150.json':150,'test_60.json':60,'legacy_dev_130.json':130}
EXPECTED_FINAL_HASHES={
    'dev_150.json':'8d78ace5d05fbf5c987ac38e8ba3d696d670726a6604c3942672150c5d0fd864',
    'test_60.json':'a5c5d38ef2b1b4418ec95ad58a2f00271c72a06c0e58e60a67b566f313cfd290',
}

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
ids=set(); ordered_ids=[]; chunks=0; duplicates=[]
for p in sorted(CORPUS.glob('*.jsonl')):
    for ln,line in enumerate(p.read_text(encoding='utf-8').splitlines(),1):
        if not line.strip(): continue
        rec=json.loads(line)
        if not (rec.get('text') or '').strip(): continue
        chunks+=1; rid=str(rec.get('id'))
        if rid in ids: duplicates.append((p.name,ln,rid))
        ids.add(rid); ordered_ids.append(rid)
print(f'corpus files : {len(list(CORPUS.glob("*.jsonl")))}')
print(f'corpus chunks: {chunks}')
print(f'unique IDs   : {len(ids)}')
errors=[]
if chunks!=377: errors.append(f'expected 377 corpus chunks, got {chunks}')
if duplicates: errors.append(f'duplicate corpus IDs: {duplicates[:5]}')
for name,n in EXPECTED_COUNTS.items():
    p=GT/name; data=json.loads(p.read_text(encoding='utf-8'))
    relevant={str(x) for q in data for x in q.get('relevant_answer',[])}
    missing=sorted(relevant-ids)
    print(f'{name:22s}: {len(data):3d} questions; missing IDs={len(missing)}; sha256={sha(p)}')
    if len(data)!=n: errors.append(f'{name}: expected {n}, got {len(data)}')
    if missing: errors.append(f'{name}: {len(missing)} missing relevant IDs')
    if name in EXPECTED_FINAL_HASHES and sha(p)!=EXPECTED_FINAL_HASHES[name]:
        errors.append(f'{name}: final frozen SHA256 changed')
for profile in ['baseline','hazm']:
    cfg=json.loads((ROOT/'configs'/f'{profile}.json').read_text(encoding='utf-8'))
    index_root=ROOT/'indices'/cfg['indices_subdir']
    for meta in sorted(index_root.glob('*/all_meta.json')):
        rows=json.loads(meta.read_text(encoding='utf-8'))
        if len(rows)!=377: errors.append(f'{meta}: expected 377 metadata rows, got {len(rows)}')
        meta_ids=[str(r.get('id')) for r in rows]
        if meta_ids != ordered_ids: errors.append(f'{meta}: metadata order/IDs do not match corpus order')
print('\nstatus:', 'FAIL' if errors else 'PASS')
for e in errors: print(' -',e)
raise SystemExit(1 if errors else 0)
