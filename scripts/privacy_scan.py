#!/usr/bin/env python3
"""Heuristic release-time privacy scan. Human review is still required."""
import json, re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
EMAIL=re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}')
PHONE=re.compile(r'(?<!\d)(?:\+?98|0)?9\d{9}(?!\d)|(?<!\d)0\d{2,3}[- ]?\d{7,8}(?!\d)')
hits=[]
for p in sorted((ROOT/'data/corpus').glob('*.jsonl')):
    for ln,line in enumerate(p.read_text(encoding='utf-8').splitlines(),1):
        try: obj=json.loads(line)
        except json.JSONDecodeError: continue
        text=json.dumps(obj,ensure_ascii=False)
        for kind,rx in [('email',EMAIL),('phone',PHONE)]:
            for value in rx.findall(text): hits.append((p.name,ln,kind,value))
if not hits:
    print('No email/phone-like strings detected. Manual review is still required.')
else:
    print(f'Detected {len(hits)} email/phone-like strings requiring human review:')
    for row in hits: print(f'{row[0]}:{row[1]} [{row[2]}] {row[3]}')
