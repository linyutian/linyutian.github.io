# linyutian.github.io

Source for [linyutian.github.io](https://linyutian.github.io/), built with Hugo + PaperMod (as a Hugo Module) and deployed by GitHub Actions on push to `main`.

## Project layout

```
Builds the public site
  hugo.yaml            site config: title, landing-page intro, social links, math, highlighting
  content/             posts (one folder per post: index.md + its images)
  layouts/             overrides for PaperMod (image resizing, math, analytics, landing intro)
  assets/css/          custom CSS (code colors, landing-page text)

Repo tooling (in the repo, not part of the site)
  archetypes/posts.md  front-matter template used by `hugo new`
  .github/workflows/   GitHub Actions: build and deploy on push to main
  go.mod, go.sum       pins the PaperMod theme version (Hugo Module)
  scripts/check-post.py  post linter (front matter, images, math, links)
  README.md            this file

Local only (gitignored, never pushed)
  CLAUDE.md, .claude/  context and writing-workflow skills for Claude Code
  drafts/              work-in-progress posts
  content/posts/hello-world/   feature test post
  public/, resources/  build output
```

## Write a post

Posts are Markdown page bundles: `content/posts/<slug>/index.md` with images in the same folder.

```sh
hugo new posts/my-post-slug/index.md                 # or move in a finished draft folder
python3 scripts/check-post.py content/posts/my-post-slug   # lint: front matter, images, math
hugo server -D                                       # preview with drafts at http://localhost:1313
```

Set `draft: false`, commit, and push to `main` to publish.

- **Images**: `![alt](diagram.png)` with the file in the post folder. Resized to 1200px max and converted to WebP automatically; a missing image fails the build.
- **Math**: inline `\( ... \)`, display `$$ ... $$` or `\[ ... \]`. KaTeX loads only on pages with math. A single `$` is plain text, so prices are safe.
- **Nav menu**: hidden for now; uncomment `menu:` in `hugo.yaml` to show Posts / Tags / Search.
- **Cover image**: set `cover.image: cover.png` in front matter.

## Maintenance

- Update PaperMod: `hugo mod get -u && hugo mod tidy`
- Upgrade Hugo/Go: bump `HUGO_VERSION` / `GO_VERSION` in `.github/workflows/hugo.yaml` (and locally via `brew upgrade hugo go`)
- Code highlighting: `assets/css/includes/chroma-styles.css` (GitHub light/dark)
- Analytics: GoatCounter code `linyutian` in `hugo.yaml` (production builds only)
