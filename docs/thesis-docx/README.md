# DOCX export

Converts the LaTeX thesis into the DOCX format the faculty asks for, instead of
maintaining a second copy of the text.

```bash
# once, to populate the pandoc binary cache
uvx --from pypandoc-binary python -c "import pypandoc"

docs/thesis-docx/build.sh
```

The script renders the TikZ figures to PNG, flattens the LaTeX sources, converts
through pandoc against a generated reference document, then fixes numbering,
captions and cross-references. Everything it writes lands in `build/` and is
gitignored; the committed `.docx` is the reviewed output.

## Why `python-docx` is not a project dependency

The tooling needs `python-docx`, but it is deliberately absent from
`pyproject.toml`. `build.sh` pulls it in per-invocation instead:

```bash
uv run --with python-docx tools/postprocess.py ...
```

The reason is not style. `vnstock`, a runtime dependency of the forecasting
package, is currently **quarantined on PyPI**: its simple-index page carries
`pypi:project-status: quarantined` and serves no download links. Any edit to
`pyproject.toml` invalidates `uv.lock`, and the re-resolution that follows cannot
find `vnstock` at all, which breaks every `uv run` in the repository. The
committed lock file and the existing virtualenv are the only working resolution,
so a documentation-only tool must not be allowed to invalidate them.

`--with` installs into an overlay for that one command and leaves the lock
untouched. Declare the dependency normally once `vnstock` is restored or
replaced.
