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
  README.md            this file

Local only (gitignored, never pushed)
  CLAUDE.md            working notes/context for Claude Code
  content/posts/hello-world/   feature test post
  public/, resources/  build output
```

## Write a post

```sh
hugo new posts/my-post-slug/index.md   # page bundle; drop images in the same folder
hugo server -D                         # preview with drafts at http://localhost:1313
```

Set `draft: false` when it's ready, then commit and push.

- **Images**: `![alt](diagram.png)` with the file in the post folder. It's resized to 1200px max and converted to WebP automatically.
- **Math**: inline `\( ... \)`, display `$$ ... $$` or `\[ ... \]`. KaTeX loads only on pages that contain math. A single `$` is plain text, so prices are safe.
- **Nav menu**: hidden for now; uncomment `menu:` in `hugo.yaml` to show Posts / Tags / Search.
- **Cover image**: set `cover.image: cover.png` in front matter.

## Maintenance

- Update PaperMod: `hugo mod get -u && hugo mod tidy`
- Upgrade Hugo/Go: bump `HUGO_VERSION` / `GO_VERSION` in `.github/workflows/hugo.yaml` (and locally via `brew upgrade hugo go`)
- Code highlighting: `assets/css/includes/chroma-styles.css` (GitHub light/dark)
- Analytics: GoatCounter code `linyutian` in `hugo.yaml` (production builds only)
