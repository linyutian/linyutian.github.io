#!/usr/bin/env python3
"""Lint a blog post (Markdown page bundle) before preview/publish.

Usage:
  python3 scripts/check-post.py content/posts/<slug>       # a post in the site
  python3 scripts/check-post.py drafts/<slug>              # a draft (draft rules)
  python3 scripts/check-post.py --publish content/posts/<slug>   # also require publish-ready

Exit code 1 if any ERROR is found. Standard library only (macOS `sips` used for image sizes).
"""
import re
import subprocess
import sys
from pathlib import Path

RASTER = {".png", ".jpg", ".jpeg", ".webp"}
PASSTHROUGH = {".svg", ".gif"}
UNSUPPORTED = {".heic", ".heif", ".tif", ".tiff", ".bmp", ".avif"}
MAX_PX = 2400          # longer side; larger originals just bloat the repo
MAX_BYTES = 2_000_000
SLUG_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")

errors, warnings, notes = [], [], []


def err(msg): errors.append(msg)
def warn(msg): warnings.append(msg)
def note(msg): notes.append(msg)


def split_front_matter(text):
    if not text.startswith("---\n"):
        return None, text
    end = text.find("\n---", 4)
    if end == -1:
        return None, text
    return text[4:end], text[end + 4:].lstrip("\n")


def strip_code(body):
    """Remove fenced code blocks and inline code so prose checks don't trip on code."""
    body = re.sub(r"^(```|~~~).*?^\1[ \t]*$", "", body, flags=re.S | re.M)
    return re.sub(r"`[^`\n]+`", "", body)


def image_size(path):
    try:
        out = subprocess.run(["sips", "-g", "pixelWidth", "-g", "pixelHeight", str(path)],
                             capture_output=True, text=True, timeout=10).stdout
        w = int(re.search(r"pixelWidth: (\d+)", out).group(1))
        h = int(re.search(r"pixelHeight: (\d+)", out).group(1))
        return w, h
    except Exception:
        return None


def main(argv):
    publish = "--publish" in argv
    args = [a for a in argv if not a.startswith("--")]
    if len(args) != 1:
        print(__doc__)
        return 2
    target = Path(args[0])
    md = target if target.suffix == ".md" else target / "index.md"
    if not md.exists():
        print(f"ERROR: {md} not found")
        return 1
    folder = md.parent
    in_site = "content/posts" in md.as_posix()

    # --- folder / slug ---
    if md.name != "index.md":
        err(f"post file must be named index.md (page bundle), got {md.name}")
    if not SLUG_RE.match(folder.name):
        err(f"folder name '{folder.name}' is the URL slug: use lowercase-kebab-case (a-z, 0-9, -)")
    elif len(folder.name) > 60:
        warn(f"slug is {len(folder.name)} chars; shorter URLs age better")

    text = md.read_text(encoding="utf-8")
    fm, body = split_front_matter(text)

    # --- front matter ---
    if fm is None:
        err("missing YAML front matter (--- block at top)")
        fm = ""
    def fm_val(key):
        m = re.search(rf"^{key}:[ \t]*(.*)$", fm, flags=re.M)
        return None if m is None else m.group(1).strip().strip('"').strip("'")
    title = fm_val("title")
    if not title:
        err("front matter: title is missing or empty")
    date = fm_val("date")
    if not date:
        err("front matter: date is missing")
    elif not re.match(r"^\d{4}-\d{2}-\d{2}", date):
        err(f"front matter: date '{date}' should start YYYY-MM-DD")
    draft = fm_val("draft")
    if draft not in ("true", "false"):
        err("front matter: draft must be true or false")
    tags = fm_val("tags")
    if tags is None:
        err("front matter: tags is missing (use tags: [] if none)")
    else:
        for t in re.findall(r"[\w\-./ ]+", tags.strip("[]")):
            t = t.strip()
            if t and not SLUG_RE.match(t):
                warn(f"tag '{t}' should be lowercase-kebab-case so tag URLs stay consistent")
    summary = fm_val("summary")
    if publish:
        if draft != "false":
            err("--publish: draft must be false")
        if not summary:
            warn("no summary: the list page will show the first ~70 words instead")
    cover = re.search(r"^\s+image:[ \t]*[\"']?([^\"'\n]+)", fm, flags=re.M)

    # --- body structure ---
    prose = strip_code(body)
    for line in prose.splitlines():
        if re.match(r"^# \S", line):
            err(f"H1 in body ('{line[:50]}'): the title comes from front matter; start sections at ##")
            break
    for m in re.finditer(r"^(```|~~~)[ \t]*$", body, flags=re.M):
        # bare fence: only an *opening* fence matters, count bare fences before this one
        before = body[:m.start()]
        if len(re.findall(r"^(```|~~~)", before, flags=re.M)) % 2 == 0:
            warn("code block without a language (```python, ```bash, ```text...) won't be highlighted")
            break

    # --- math ---
    inline_dollar = re.findall(r"(?<![\\$])\$(?=[^\s$])([^$\n]*?[^\s\\$])\$(?!\$)", prose)
    if inline_dollar:
        msg = (f"single-$ inline math found ({', '.join('$'+x+'$' for x in inline_dollar[:3])}...): "
               "the site only renders \\( ... \\) inline")
        (err if in_site else note)(msg if in_site else msg + " (fine in a draft; converted on import)")
    if re.search(r"\$\$|\\\(|\\\[", prose):
        note("contains math: KaTeX will load on this page")

    # --- links ---
    for bad in re.findall(r"\]\(((?:file:|https?://localhost|https?://127\.0\.0\.1)[^)]*)\)", body):
        err(f"link points to a local-only address: {bad}")
    for m in re.findall(r"\b(TODO|TK|XXX|FIXME)\b", prose):
        (err if publish else warn)(f"leftover marker '{m}' in text")
        break

    # --- images ---
    refs = re.findall(r"!\[([^\]]*)\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)", body)
    refs += [("(html)", s) for s in re.findall(r"<img[^>]+src=[\"']([^\"']+)", body)]
    used = set()
    for alt, src in refs:
        if re.match(r"^[a-z]+://", src):
            warn(f"remote image {src}: may disappear or expire; prefer downloading it into the folder")
            continue
        rel = src.split("#")[0].split("?")[0].removeprefix("./")
        if "%20" in rel or " " in rel:
            err(f"image '{src}': rename without spaces (kebab-case)")
            continue
        p = folder / rel
        used.add(p.name)
        if not p.exists():
            err(f"image '{src}' not found in {folder}/")
            continue
        if alt.strip() == "":
            warn(f"image '{src}' has empty alt text (describe it for screen readers / broken loads)")
        ext = p.suffix.lower()
        if ext in UNSUPPORTED:
            err(f"image '{src}': {ext} isn't web-safe; convert to .png/.jpg (sips -s format png)")
        elif ext in RASTER:
            size = image_size(p)
            if size and max(size) > MAX_PX:
                warn(f"image '{src}' is {size[0]}x{size[1]}; downscale originals to <= {MAX_PX}px "
                     f"(sips -Z {MAX_PX} '{p}') to keep the repo small")
            if p.stat().st_size > MAX_BYTES:
                warn(f"image '{src}' is {p.stat().st_size/1e6:.1f} MB")
        elif ext not in PASSTHROUGH:
            warn(f"image '{src}': unusual extension {ext}")
    if cover:
        c = cover.group(1).strip()
        if c and not (folder / c).exists():
            err(f"cover image '{c}' not found in {folder}/")
        used.add(c)

    # --- unused files ---
    for f in sorted(folder.iterdir()):
        if f.name in ("index.md", ".DS_Store") or f.name in used or f.is_dir():
            continue
        if f.suffix.lower() in (".excalidraw", ".drawio", ".fig", ".py", ".ipynb"):
            note(f"source file kept alongside (not published): {f.name}")
        else:
            note(f"unreferenced file (not published; delete if unneeded): {f.name}")

    # --- report ---
    words = len(re.findall(r"\w+", prose))
    print(f"{md}  |  title: {title!r}  |  draft: {draft}  |  ~{words} words, ~{max(1, round(words/220))} min read")
    for label, items in (("ERROR", errors), ("WARN ", warnings), ("note ", notes)):
        for i in items:
            print(f"  {label} {i}")
    print("  OK: no errors" if not errors else f"  {len(errors)} error(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
