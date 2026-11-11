# Maintaining and Publishing Documentation

The site is generated from Markdown files in `docs/` with Sphinx and PyTorch
Sphinx Theme 2. It is published at
[zhiqwang.github.io/huo](https://zhiqwang.github.io/huo/). The site language,
page titles, header, navigation, and search interface are English.
The same deployment includes the
[interactive browser demo](https://zhiqwang.github.io/huo/demo/).

## Local preview and strict build

From the repository root, install the documentation dependencies with uv and
Python 3.12:

```bash
uv sync --locked --only-group docs --python 3.12
uv run --locked --only-group docs sphinx-autobuild -a --host 0.0.0.0 --port 8000 docs site
```

Open [http://127.0.0.1:8000/](http://127.0.0.1:8000/) locally, or use the host
machine's IP address and port 8000 from another device. The preview rebuilds
when source files change. The `-a` flag also refreshes the output when only CSS
or templates change, so the browser receives the updated layout.

Before publishing, run the same strict build used in CI:

```bash
uv run --locked --only-group docs sphinx-build -b html -W --keep-going docs site
```

The output is written to `site/`. Both that directory and `.venv/` are ignored
by Git. Documentation dependencies are defined in the `docs` group in
`pyproject.toml` and locked in `uv.lock`; run `uv lock` after changing them.

To include the browser demo in a local preview of the complete Pages site,
run these commands from the repository root after the Sphinx build above
(Node.js 22 and npm are required):

```bash
npm --prefix packages ci
npm --prefix packages run build
cp -R packages/demo packages/src site/
python -m http.server 8000 --bind 0.0.0.0 --directory site
```

Open `http://127.0.0.1:8000/demo/`. The `demo/` and `src/` directories stay
next to each other so the demo's relative JavaScript imports also work under
the `/huo/` GitHub Pages prefix. The demo loads jax-js from `esm.sh`; visitors
need access to that CDN. Rebuild and copy the demo again after changing its
TypeScript sources.

## Adding articles and mathematical notation

Place new Markdown articles in `docs/` and add their navigation entries to
the `toctree` in `docs/index.md`. Use concise English navigation titles and
relative Markdown links within the site, such as `[Tutorial](tutorial.md)`.
For source files outside `docs/`, use GitHub file links so deployed links do
not point to nonexistent site paths.

Use `$...$` for inline mathematics and `$$...$$` on separate lines for display
mathematics. The MyST `dollarmath` extension parses both forms; MathJax renders
matrices, cases, and explicit `\tag{...}` equation numbers. Configuration is
in `docs/conf.py`. Headings generate the article's table of contents in the
right sidebar, while the `toctree` supplies the left navigation.

Inside mathematics, write code identifiers with `\mathtt{img\_end}`. This
renders the underscore correctly with the site's MathJax configuration. Use
ordinary Markdown backticks for identifiers outside mathematics.

For a tagged equation containing matrix rows or other `\\` line breaks, use
an explicit equation environment with `:nowrap:`. This prevents Sphinx from
placing the tag inside a `split` environment, which MathJax rejects:

````markdown
```{math}
:nowrap:

\begin{equation}
A=\begin{pmatrix}1&2\\2&1\end{pmatrix}.
\tag{1}
\end{equation}
```
````

## Checking the Lean proofs

The proof project is in `lean/` and pins Lean and Mathlib to v4.24.0. Install
Lean's `elan` toolchain manager, then run:

```bash
cd lean
lake exe cache get Mathlib.Analysis.SpecificLimits.Basic Mathlib.Analysis.InnerProductSpace.Basic \
  Mathlib.Tactic.FieldSimp Mathlib.Tactic.Linarith Mathlib.Tactic.NormNum Mathlib.Tactic.Ring
lake build
```

The first command downloads the Mathlib artifacts needed by the proof imports.
The second checks the default `Huo` library, which imports every project proof.
Project warnings are errors, so incomplete proofs fail the build. Keep
`lean-toolchain`, `lakefile.toml`, and `lake-manifest.json` together when updating
versions. Dependencies and compiled output in `lean/.lake/` are ignored by Git.
See [Lean project setup](https://leanprover-community.github.io/install/project.html)
for installation details.

The [proof article](backward_propagation_proof.md#lean-proofs)
lists the formalized results and the limits of their coverage. Its Lean excerpt
uses `literalinclude` to read the actual source. Update the coverage table when
adding or changing theorems; a successful documentation build alone does not
check a Lean proof.

## GitHub Pages deployment

For initial repository setup, select **GitHub Actions** under **Build and
deployment → Source** in
[Settings → Pages](https://github.com/zhiqwang/huo/settings/pages). See
[GitHub's custom workflow instructions](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages).

The [Deploy documentation and demo workflow](https://github.com/zhiqwang/huo/actions/workflows/docs.yml)
runs for changes to documentation, the TypeScript package, Lean sources,
documentation dependencies, or the workflow itself:

- Pull requests build the TypeScript demo, run a strict Sphinx build, and
  check the Lean proofs.
- Pushes to `main` build the documentation and demo into one Pages artifact.
  Deployment waits for both the build and proof jobs to succeed. Documentation
  is served at `/huo/`, and the demo is served at `/huo/demo/`.
- **Run workflow** on `main` republishes manually; selecting another branch
  runs the checks without deploying.

Deployment uses GitHub's `GITHUB_TOKEN` and OIDC permissions without a personal
access token. Its status and site URL appear in the `github-pages` environment.
