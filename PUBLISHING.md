# Publishing `cubespec` to PyPI

This is the operator runbook. It covers the **first** release and every
**subsequent** release. There are two supported paths:

1. **Manual** — local `twine upload` with a PyPI API token.
2. **Trusted Publishing (recommended)** — GitHub Actions OIDC, no token
   stored anywhere.

---

## 0. Pre-flight checklist

- [ ] All tests green: `pytest python/tests/ -v`.
- [ ] Parity tests green: `pytest python/tests/test_parity.py -v`.
- [ ] Lint clean: `ruff check python/`.
- [ ] Type-check clean: `mypy python/src/`.
- [ ] Bumped `version` in `python/pyproject.toml`.
- [ ] Added a release entry to `python/CHANGELOG.md`.
- [ ] Updated `python/README.md` if the public API changed.
- [ ] README renders on GitHub: `grip python/README.md` (optional).

## 1. Build the artefacts

From the repo root:

```bash
cd python
rm -rf dist build src/*.egg-info
python -m build
ls dist/
# cubespec-X.Y.Z-py3-none-any.whl
# cubespec-X.Y.Z.tar.gz
twine check dist/*
```

`twine check` validates README rendering, classifiers, and metadata.

## 2. Local install smoke test

```bash
python -m venv /tmp/cubespec-smoke
source /tmp/cubespec-smoke/bin/activate
pip install dist/*.whl
cubespec --help
cubespec run --n 1000 --output /tmp/smoke.json
jq '.means.P9_compressive_strength' /tmp/smoke.json
deactivate
```

Expected: P9 mean ≈ 44.2 MPa.

## 3. TestPyPI dry-run

```bash
twine upload --repository testpypi dist/*
# then verify it installs cleanly from TestPyPI:
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ \
            cubespec
```

You'll need a TestPyPI account and a token saved in `~/.pypirc` under
`[testpypi]`.

## 4. Production upload (manual path)

```bash
twine upload dist/*
```

Saves your token in `~/.pypirc`:

```ini
[pypi]
  username = __token__
  password = pypi-AgEIcHlw…
```

## 5. Production upload (Trusted Publishing — recommended)

One-time setup:

1. Create the project on PyPI (first manual upload, OR use the
   "pending" trusted publisher feature).
2. On PyPI → *Manage project → Publishing → Add a new publisher* with:
   - Owner: `ai-systems-today`
   - Repository: `cubespec`
   - Workflow filename: `publish.yml`
   - Environment name: `pypi`
3. Push a tag:

```bash
git tag v0.1.0
git push origin v0.1.0
```

The `.github/workflows/publish.yml` workflow builds the wheel and
publishes via OIDC. No secret tokens leave PyPI.

## 6. Post-release

- [ ] GitHub Release: draft from the new tag, paste the changelog entry.
- [ ] Verify the project page on https://pypi.org/project/cubespec/.
- [ ] Verify a clean install: `pip install cubespec` in a fresh venv.
- [ ] Tweet / announce / update the dashboard footer if you publicise it.

## 7. Yanking and hotfixes

- **Yank** (hide from new installs without deleting): `twine yank cubespec==X.Y.Z` or via the PyPI UI. Existing pinned installs continue to work.
- **Hotfix**: bump the patch version (e.g. `0.1.0 → 0.1.1`), repeat steps 1–6.
- **Never** re-upload the same version — PyPI rejects it.

## 8. Versioning policy

Semver:

- `MAJOR` — breaking API change (function signatures, output schema).
- `MINOR` — new feature, backward-compatible.
- `PATCH` — bug fix, no API change.

Pre-1.0 the bar is relaxed: minor bumps may include breaking changes
provided the `CHANGELOG.md` flags them.
