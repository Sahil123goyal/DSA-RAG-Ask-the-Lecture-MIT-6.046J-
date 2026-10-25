"""
build_index.py - build the RAG knowledge base from data/sources.json

Usage:
    python build_index.py             # text + tables only (free, fast)
    python build_index.py --vision    # also describe diagram/handwritten pages
                                      # with Groq Qwen (needs GROQ_API_KEY env variable)

Outputs: index.faiss, chunks.pkl, chunks.json, data/pages/*.png
"""
import argparse
import base64
import io
import json
import os
import pickle
import re
import time
from pathlib import Path

import faiss
import pymupdf as fitz
from sentence_transformers import SentenceTransformer


DATA = Path("data")
PAGES = DATA / "pages"
VISION_CACHE = DATA / "vision_cache.json"
EMB_MODEL = "BAAI/bge-small-en-v1.5"
CHUNK_WORDS, OVERLAP = 350, 60  # roughly 500 tokens, ~85 token overlap

VISION_PROMPT = (
    "This is a page from MIT 6.046J algorithms course material. Transcribe all "
    "text on the page (including handwriting). Write math in plain text or LaTeX. "
    "Convert any table to Markdown, keeping every number exactly. For each diagram, "
    "describe what it shows: nodes, arrows, labels and what it illustrates. "
    "Do not add anything that is not on the page."
)

TS = re.compile(r"(\d+):(\d+):(\d+)\.\d+\s+-->")
TAG = re.compile(r"<[^>]+>")


# ---------- transcript (.vtt) ----------
def parse_vtt(path):
    """Return a list of (start_seconds, text) with tags and duplicates removed."""
    cues, start = [], None
    for line in path.read_text(encoding="utf-8").splitlines():
        m = TS.match(line)
        if m:
            h, mi, s = map(int, m.groups())
            start = h * 3600 + mi * 60 + s
            continue
        if not line.strip():
            start = None
            continue
        if start is None or line.startswith(("WEBVTT", "Kind:", "Language:")):
            continue
        text = TAG.sub("", line).strip()
        if text:
            cues.append((start, text))
    out, last = [], None
    for t, txt in cues:  # YouTube auto-captions repeat each line twice
        if txt != last:
            out.append((t, txt))
            last = txt
    return out


def fmt_time(sec):
    return f"{sec // 60}:{sec % 60:02d}"


def yt_link(url, sec):
    url = re.sub(r"[&?]t=\d+s?", "", url)
    return f"{url}{'&' if '?' in url else '?'}t={sec}s"


def windows(items, size=CHUNK_WORDS, overlap=OVERLAP):
    step = size - overlap
    for i in range(0, len(items), step):
        yield items[i:i + size]
        if i + size >= len(items):
            break


def process_vtt(entry):
    cues = parse_vtt(DATA / entry["file"])
    words = [(w, t) for t, txt in cues for w in txt.split()]
    chunks = []
    for win in windows(words):
        sec = win[0][1]
        chunks.append({
            "text": " ".join(w for w, _ in win),
            "locator": f"~{fmt_time(sec)}",
            "link": yt_link(entry["url"], sec),
            "kind": "transcript",
        })
    return chunks


# ---------- PDFs ----------
def table_to_md(rows):
    rows = [[(c or "").replace("\n", " ").strip() for c in r] for r in rows if r]
    if len(rows) < 2:
        return ""
    md = ["| " + " | ".join(rows[0]) + " |",
          "|" + "---|" * len(rows[0])]
    md += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(md)


THINK = re.compile(r"<think>.*?</think>", re.S)
# remembers the request size that worked, so later pages skip the failed attempt
VISION = {"side": int(os.getenv("VISION_MAX_SIDE", "1000")), "max_tokens": 1200}


def encode_image(png_path, max_side):
    """Downscale the page and send it as JPEG: far fewer tokens than full-size PNG."""
    from PIL import Image

    img = Image.open(png_path).convert("RGB")
    img.thumbnail((max_side, max_side))  # only ever shrinks
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=75)
    return base64.standard_b64encode(buf.getvalue()).decode()


def describe_page(png_path, retries=7):
    from groq import Groq  # pip install groq

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise SystemExit("Set the GROQ_API_KEY environment variable first.")
    client = Groq(api_key=api_key)
    model = os.getenv("VISION_MODEL", "qwen/qwen3.8-27b")
    side, max_tokens = VISION["side"], VISION["max_tokens"]

    for attempt in range(retries):
        b64 = encode_image(png_path, side)
        kw = dict(model=model, max_tokens=max_tokens, temperature=0.2, messages=[{
            "role": "user", "content": [
                {"type": "text", "text": VISION_PROMPT},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ]}])
        try:
            try:  # thinking off: faster and saves the free token budget
                resp = client.chat.completions.create(
                    extra_body={"reasoning_effort": "none"}, **kw)
            except Exception as e:
                if "429" in str(e):
                    raise
                resp = client.chat.completions.create(  # fallback: hide reasoning
                    extra_body={"reasoning_format": "hidden"}, **kw)
            text = THINK.sub("", resp.choices[0].message.content or "").strip()
            VISION["side"], VISION["max_tokens"] = side, max_tokens  # remember
            time.sleep(float(os.getenv("VISION_PAUSE", "12")))  # stay under the TPM cap
            return text
        except Exception as e:
            msg = str(e)
            if "too large" in msg.lower():  # waiting won't help: shrink the request
                side = int(side * 0.75)
                max_tokens = max(600, int(max_tokens * 0.8))
                print(f"  request too large, retrying with image {side}px / "
                      f"{max_tokens} output tokens")
                continue
            wait = 20 * (attempt + 1)
            print(f"  API error ({msg[:90]}...), retrying in {wait}s")
            time.sleep(wait)
    raise SystemExit("Vision model kept failing. Re-run later; finished pages are cached.")


def process_pdf(entry, use_vision, cache):
    pdf_path = DATA / entry["file"]
    doc = fitz.open(pdf_path)
    chunks, stats = [], {"text": 0, "vision": 0, "flagged": 0}
    PAGES.mkdir(parents=True, exist_ok=True)

    for pno, page in enumerate(doc, 1):
        png = PAGES / f"{pdf_path.stem}_p{pno}.png"
        page.get_pixmap(dpi=120).save(png)  # shown under answers in the app
        text = page.get_text().strip()
        needs_vision = (len(text) < 200 or bool(page.get_images())
                        or len(page.get_drawings()) > 20)
        loc = f"page {pno}"

        if needs_vision:
            stats["flagged"] += 1
        if needs_vision and use_vision:
            key = f"{entry['file']}:{pno}"
            if key not in cache:
                print(f"  vision: {key}")
                cache[key] = describe_page(png)
                VISION_CACHE.write_text(json.dumps(cache, indent=1), encoding="utf-8")
            for part in windows(cache[key].split()):
                chunks.append({"text": " ".join(part), "locator": loc,
                               "kind": "AI-generated page description",
                               "image": str(png)})
            stats["vision"] += 1
            continue

        for tbl in page.find_tables().tables:  # each table = one chunk
            md = table_to_md(tbl.extract())
            if md:
                chunks.append({"text": md, "locator": loc, "kind": "table",
                               "image": str(png)})
        for part in windows(text.split()):
            if len(part) > 15:
                chunks.append({"text": " ".join(part), "locator": loc,
                               "kind": "text", "image": str(png)})
                stats["text"] += 1

    print(f"  pages flagged for vision: {stats['flagged']}/{len(doc)}"
          f"{'' if use_vision else '  (run with --vision to cover them)'}")
    return chunks


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vision", action="store_true")
    args = ap.parse_args()

    sources = json.loads((DATA / "sources.json").read_text(encoding="utf-8"))
    cache = json.loads(VISION_CACHE.read_text()) if VISION_CACHE.exists() else {}

    all_chunks = []
    for entry in sources:
        print(f"Processing {entry['file']}")
        path = DATA / entry["file"]
        if not path.exists():
            print("  MISSING FILE, skipped")
            continue
        if path.suffix == ".vtt":
            chunks = process_vtt(entry)
        elif path.suffix == ".pdf":
            chunks = process_pdf(entry, args.vision, cache)
        else:
            print("  unsupported type, skipped")
            continue
        for c in chunks:
            c.update(source=entry["file"], type=entry["type"],
                     lecture=entry["lecture"], url=c.get("link", entry["url"]))
        print(f"  {len(chunks)} chunks")
        all_chunks += chunks

    print(f"\nTotal chunks: {len(all_chunks)}. Embedding...")
    model = SentenceTransformer(EMB_MODEL)
    vecs = model.encode([c["text"] for c in all_chunks],
                        normalize_embeddings=True, show_progress_bar=True)
    index = faiss.IndexFlatIP(vecs.shape[1])  # cosine similarity (vectors normalized)
    index.add(vecs)

    faiss.write_index(index, "index.faiss")
    with open("chunks.pkl", "wb") as f:
        pickle.dump(all_chunks, f)
    Path("chunks.json").write_text(json.dumps(all_chunks, indent=1), encoding="utf-8")
    print("Saved index.faiss, chunks.pkl, chunks.json")


if __name__ == "__main__":
    main()