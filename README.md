# jakobj.dev

Personal website, built with [Hugo](https://gohugo.io/) and the [PaperMod](https://github.com/adityatelange/hugo-PaperMod) theme.

## Local development

```sh
git submodule update --init   # fetch the theme
hugo server -D                # live preview, including drafts
```

New posts: `hugo new posts/<slug>/index.md`. Images go in the post's folder next to `index.md`. Posts start as drafts, so set `draft: false` before merging or they won't be published.

## CI and deployment

`.github/workflows/ci.yml` runs on every PR and push to `main`:

1. Builds the site (any Hugo warning fails the build), checks that key pages exist, and checks internal links.
2. Computes a version number with GitVersion.
3. Builds the Docker image (Hugo build stage, then nginx serving on port 8080).

On `main` it also pushes the image to `ghcr.io/jakobottar/jakobjdev` as `:<version>` and `:stable`, and tags the commit. PRs stop before pushing.

The server runs `compose.yml`, which pulls the `:stable` image.

## Reverse proxy (Traefik)

The site sits behind [Traefik](https://traefik.io/), which handles HTTPS and also routes other services on the same server. Traefik is **not** part of this repo: it runs as its own compose project (`~/traefik` on the server), with its own `traefik.yml` and `compose.yml`.

`compose.yml` here relies on these names from Traefik's setup:

| Name | What it is |
|---|---|
| `traefik_web` | Docker network Traefik creates; the website container joins it as an external network |
| `websecure` | Traefik entrypoint on port 443 (port 80 redirects to it) |
| `mycerts` | Let's Encrypt certificate resolver (HTTP challenge) |

If any of these are renamed in Traefik's config, update the labels in `compose.yml` to match.
