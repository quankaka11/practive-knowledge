"""Heuristic audit: answer-position spread, 'correct = longest option' bias, positional references."""
import glob, json, os, re, collections, sys
sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POS_RE = re.compile(r"(phương án|đáp án|lựa chọn|option|câu)\s*\(?[A-F1-6]\)?(?![\w])|\b[A-F]\)\s|\b(cả|tất cả)\s+(các\s+)?(ý|phương án|đáp án)\s+trên", re.I)
files = sys.argv[1:] or sorted(glob.glob(os.path.join(ROOT, "data", "src", "*.json")))
for p in files:
    d = json.load(open(p, encoding="utf-8"))
    pos = collections.Counter(); longest = 0; singles = 0; refs = []; lc = []; lw = []
    for q in d["questions"]:
        if q["type"] == "single":
            singles += 1; pos[q["answer"]] += 1
            L = [len(o) for o in q["options"]]
            if L[q["answer"]] == max(L) and L.count(max(L)) == 1: longest += 1
        if q["type"] in ("single", "multi"):
            ans = {q["answer"]} if q["type"] == "single" else set(q["answer"])
            for i, o in enumerate(q["options"]):
                (lc if i in ans else lw).append(len(o))
        for k in ("q", "explain"):
            text = re.sub(r"```.*?```", "", q.get(k, ""), flags=re.S)
            m = POS_RE.search(text)
            if m: refs.append(f'{q["id"]}.{k}: ...{text[max(0,m.start()-25):m.end()+15]}...')
    print(f"{os.path.basename(p)}: single pos={dict(sorted(pos.items()))} correct-is-longest={longest}/{singles} ({100*longest//max(1,singles)}%) | avg len correct={sum(lc)/max(1,len(lc)):.0f} wrong={sum(lw)/max(1,len(lw)):.0f}")
    for r in refs: print("   ref?", r.replace("\n", " "))
