import os
nc = 20
lab_dir = r"E:\Program\phase-1 project\EXPERIMENTAL-1\test\dataset\valid\labels"
bad = []
empty = []
for f in os.listdir(lab_dir):
    if not f.endswith('.txt'): continue
    p = os.path.join(lab_dir, f)
    with open(p, 'r', encoding='utf-8') as fh:
        lines = [L.strip() for L in fh if L.strip()]
    if not lines:
        empty.append(p)
        continue
    for L in lines:
        parts = L.split()
        try:
            cid = int(parts[0])
            if cid < 0 or cid >= nc:
                bad.append((p, L))
        except Exception:
            bad.append((p, L))
print("Empty label files:", empty)
print("Files with invalid label lines:", bad)