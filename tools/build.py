"""Build the quiz from data/src/*.json.

Outputs:
  data/bank.js        -> loaded by index.html (works when opened via file://)
  dist/ai-quiz.html   -> single self-contained page (CSS/JS/data inlined) for sharing/publishing
  KIEN_THUC.md        -> all knowledge notes in one Markdown file for reading/printing

Usage: python tools/build.py
"""
import glob
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read(p):
    with open(os.path.join(ROOT, p), encoding="utf-8") as f:
        return f.read()


def write(p, s):
    full = os.path.join(ROOT, p)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w", encoding="utf-8", newline="\n") as f:
        f.write(s)


def main():
    rc = subprocess.call([sys.executable, os.path.join(ROOT, "tools", "validate.py")])
    if rc != 0:
        print("Validation failed - fix data/src first.")
        sys.exit(rc)

    topics = []
    for p in sorted(glob.glob(os.path.join(ROOT, "data", "src", "*.json"))):
        with open(p, encoding="utf-8") as f:
            topics.append(json.load(f))
    topics.sort(key=lambda t: (t.get("order", 99), t["id"]))

    payload = json.dumps(topics, ensure_ascii=False, separators=(",", ":"))
    # keep "</script>" sequences from closing an inline script tag
    payload = payload.replace("</", "<\\/")
    bank_js = "window.QUIZ_BANK = " + payload + ";\n"
    write("data/bank.js", bank_js)

    # single-file page: head/body regions of index.html + inlined css/js/data
    html = read("index.html")
    head = re.search(r"<!-- ARTIFACT:HEAD:START -->(.*?)<!-- ARTIFACT:HEAD:END -->", html, re.S).group(1)
    body = re.search(r"<!-- ARTIFACT:BODY:START -->(.*?)<!-- ARTIFACT:BODY:END -->", html, re.S).group(1)
    css = read("assets/app.css")
    js = read("assets/app.js")
    single = (head.strip() + "\n<style>\n" + css + "\n</style>\n" + body.strip() + "\n<script>\n"
              + bank_js + "</script>\n<script>\n" + js + "\n</script>\n")
    write("dist/ai-quiz.html", single)

    # knowledge summary
    total = sum(len(t["questions"]) for t in topics)
    lines = ["# Tổng hợp kiến thức ôn phỏng vấn AI Engineer", "",
             f"{len(topics)} chủ đề · {total} câu hỏi trong web quiz. File này được sinh tự động từ `data/src/*.json`.", "",
             "## Mục lục", ""]
    for t in topics:
        lines.append(f"- {t['icon']} {t['name']}")
    for t in topics:
        lines += ["", "---", "", f"## {t['icon']} {t['name']}", ""]
        for n in t["notes"]:
            body_md = re.sub(r"^(#{1,4}) ", lambda m: "#" * min(6, len(m.group(1)) + 3) + " ", n["md"], flags=re.M)
            lines += [f"### {n['title']}", "", body_md.strip(), ""]
    write("KIEN_THUC.md", "\n".join(lines) + "\n")

    print(f"Built {len(topics)} topics / {total} questions -> data/bank.js, dist/ai-quiz.html "
          f"({len(single.encode('utf-8')) // 1024} KB), KIEN_THUC.md")


if __name__ == "__main__":
    main()
