"""Validate question bank JSON files in data/src/.

Usage: python tools/validate.py [file.json ...]   (default: all files in data/src)
"""
import json, sys, glob, os, re
sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TYPES = {"single", "multi", "numeric"}


def check(path):
    errs, warns = [], []
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
    except Exception as e:
        return [f"JSON parse error: {e}"], []
    for k in ("id", "name", "icon", "notes", "questions"):
        if k not in d:
            errs.append(f"missing top-level key '{k}'")
    if errs:
        return errs, warns
    if not isinstance(d["notes"], list) or not d["notes"]:
        errs.append("notes must be a non-empty list")
    for i, n in enumerate(d["notes"]):
        if not isinstance(n, dict) or not n.get("title") or not n.get("md"):
            errs.append(f"notes[{i}] needs title + md")
    seen = set()
    stats = {"single": 0, "multi": 0, "numeric": 0, "trap": 0, 1: 0, 2: 0, 3: 0}
    for i, q in enumerate(d["questions"]):
        tag = q.get("id", f"#{i}")
        if tag in seen:
            errs.append(f"{tag}: duplicate id")
        seen.add(tag)
        if not str(tag).startswith(d["id"] + "-"):
            errs.append(f"{tag}: id must start with '{d['id']}-'")
        t = q.get("type")
        if t not in TYPES:
            errs.append(f"{tag}: bad type {t!r}")
            continue
        stats[t] += 1
        for k in ("sub", "q", "explain"):
            if not isinstance(q.get(k), str) or not q[k].strip():
                errs.append(f"{tag}: missing/empty '{k}'")
        lv = q.get("level")
        if lv not in (1, 2, 3):
            errs.append(f"{tag}: level must be 1/2/3")
        else:
            stats[lv] += 1
        if q.get("trap"):
            stats["trap"] += 1
        if t in ("single", "multi"):
            opts = q.get("options")
            if not isinstance(opts, list) or not (3 <= len(opts) <= 6):
                errs.append(f"{tag}: options must be a list of 3-6 strings")
                continue
            if len(set(opts)) != len(opts):
                errs.append(f"{tag}: duplicate options")
            a = q.get("answer")
            if t == "single":
                if not isinstance(a, int) or not (0 <= a < len(opts)):
                    errs.append(f"{tag}: single answer must be a valid index")
            else:
                if (not isinstance(a, list) or not a or len(set(a)) != len(a)
                        or any(not isinstance(x, int) or not (0 <= x < len(opts)) for x in a)):
                    errs.append(f"{tag}: multi answer must be a non-empty list of valid indices")
            for o in opts:
                if re.match(r"^\s*[A-Fa-f][\.\)]\s", o):
                    warns.append(f"{tag}: option starts with a letter label (UI adds labels): {o[:30]!r}")
        else:
            a = q.get("answer")
            if not isinstance(a, (int, float)) or isinstance(a, bool):
                errs.append(f"{tag}: numeric answer must be a number")
            tol = q.get("tolerance", 0)
            if not isinstance(tol, (int, float)) or tol < 0:
                errs.append(f"{tag}: tolerance must be a non-negative number")
    return errs, warns, stats, len(d["questions"])


def main():
    files = sys.argv[1:] or sorted(glob.glob(os.path.join(ROOT, "data", "src", "*.json")))
    bad = False
    for p in files:
        res = check(p)
        errs, warns = res[0], res[1]
        name = os.path.basename(p)
        if errs:
            bad = True
            print(f"[FAIL] {name}")
            for e in errs:
                print("   -", e)
        else:
            stats, n = res[2], res[3]
            print(f"[OK]   {name}: {n} questions | single={stats['single']} multi={stats['multi']} "
                  f"numeric={stats['numeric']} | L1={stats[1]} L2={stats[2]} L3={stats[3]} | trap={stats['trap']}")
        for w in warns:
            print("   ~", w)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
