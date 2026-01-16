import os, time
from pathlib import Path

roots = [
    Path(os.path.expanduser(r"~\.cache\torch")),
    Path(os.path.expanduser(r"~\AppData\Local\torch")),
    Path(os.path.expanduser(r"~\AppData\Roaming\torch")),
]

now = time.time()
hits = []

for root in roots:
    if root.exists():
        for ext in ("*.pth", "*.pt"):
            for p in root.rglob(ext):
                try:
                    age_min = (now - p.stat().st_mtime) / 60
                    hits.append((age_min, str(p), p.stat().st_size))
                except:
                    pass

hits.sort(key=lambda x: x[0])

print("Most recent torch weight files (last 7 days):")
for age, path, size in hits:
    if age <= 7 * 24 * 60:
        mb = size / (1024 * 1024)
        print(f"{age:8.1f} min | {mb:8.1f} MB | {path}")
