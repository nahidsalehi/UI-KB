import json, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class RepositoryTests(unittest.TestCase):
    def test_profiles(self):
        for name in ('baseline','hazm'):
            cfg=json.loads((ROOT/'configs'/f'{name}.json').read_text(encoding='utf-8'))
            self.assertEqual(cfg['name'],name)
    def test_ground_truth_counts_and_ids(self):
        ids=set()
        for p in (ROOT/'data/corpus').glob('*.jsonl'):
            for line in p.read_text(encoding='utf-8').splitlines():
                if line.strip():
                    rec=json.loads(line)
                    if (rec.get('text') or '').strip(): ids.add(str(rec['id']))
        self.assertEqual(len(ids),377)
        for name,n in [('dev_150.json',150),('test_60.json',60),('legacy_dev_130.json',130)]:
            gt=json.loads((ROOT/'data/ground_truth'/name).read_text(encoding='utf-8'))
            self.assertEqual(len(gt),n)
            missing={str(x) for q in gt for x in q['relevant_answer']}-ids
            self.assertFalse(missing)
if __name__=='__main__': unittest.main()
