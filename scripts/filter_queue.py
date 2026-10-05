"""Writes a copy of a queue YAML with every BIG2015 job removed.

Usage: python scripts/filter_queue.py experiments/queue.yaml experiments/queue_malimg.yaml
"""

import sys

import yaml

src, dst = sys.argv[1], sys.argv[2]
q = yaml.safe_load(open(src, encoding="utf-8"))
dropped = 0
for tier in q["tiers"].values():
    kept = [j for j in tier["jobs"] if "big2015" not in f"{j.get('name', '')} {j.get('config', '')}"]
    dropped += len(tier["jobs"]) - len(kept)
    tier["jobs"] = kept
yaml.safe_dump(q, open(dst, "w", encoding="utf-8"), sort_keys=False)
print(f"Wrote {dst}: dropped {dropped} BIG2015 job(s)")
