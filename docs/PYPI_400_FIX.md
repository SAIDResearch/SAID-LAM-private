# Fixing PyPI upload "400 Bad Request" — step by step

The workflow now runs with **verbose uploads** so the next run will show PyPI’s real error in the Actions log. Use that message to choose the right fix below.

---

## Step 1: See the real error (after next run)

1. Push the workflow change (with `verbose: true`) and run the release again (re-run the failed job or push a new tag).
2. Open the **Publish to PyPI** step in the Actions log.
3. Scroll to the **verbose** twine output and find the line after `HTTPError: 400 Bad Request`. PyPI often returns a body like:
   - `"File already exists"` or `"This version already exists"` → go to **Step 2**.
   - `"Invalid value for ..."` or `"The use of local versions is not allowed"` → go to **Step 3**.
   - Anything about permissions or token → go to **Step 4**.

---

## Step 2: Version already on PyPI

**Symptom:** Log says the version (e.g. 1.0.10) already exists or file already exists.

**Fix:**

1. Open [pyproject.toml](../pyproject.toml) and set a **new** version, e.g. `version = "1.0.11"`.
2. Commit, push, then create and push a new tag:
   ```bash
   git add pyproject.toml && git commit -m "Bump version to 1.0.11" && git push origin main
   git tag v1.0.11 && git push origin v1.0.11
   ```
3. Wait for the workflow to run; it will upload the new version.

---

## Step 3: Invalid metadata or “local version” not allowed

**Symptom:** Error mentions metadata, version format, or “local versions”.

**Fix:**

- Ensure the version in `pyproject.toml` has **no local part** (e.g. use `1.0.11`, not `1.0.11+dev`).
- Ensure `name`, `version`, and `readme` are valid and the README file exists.

---

## Step 4: API token / permissions

**Symptom:** 400 or 403, or error about permissions/token.

**Fix:**

1. Log in to [pypi.org](https://pypi.org) → **Account** → **API tokens**.
2. Create a new token:
   - **Scope:** either **Entire account** or **Project: said-lam**.
   - Copy the token (starts with `pypi-`).
3. In GitHub: **Repo** → **Settings** → **Secrets and variables** → **Actions** (or **Environments** → `pypi-publish`).
4. Add or update secret **`PYPI_API_TOKEN`** with the new token.
5. Re-run the **Publish to PyPI** job (or push a new tag).

---

## Step 5: Test upload locally (optional)

To see the exact PyPI response on your machine:

```bash
# Build wheels (or use artifacts from CI)
maturin build --release

# Upload with twine (use your real token)
pip install twine
twine upload dist/* -u __token__ -p pypi-YOUR_TOKEN_HERE --verbose
```

The last lines will show PyPI’s response body and the real reason for 400.

---

## Summary

| What you see in verbose log | Do this |
|-----------------------------|--------|
| Version already exists / File already exists | Bump version in pyproject.toml, new tag, re-run. |
| Invalid metadata / local version | Fix version format and metadata in pyproject.toml. |
| Token / permission error | New PyPI API token, update `PYPI_API_TOKEN` in GitHub. |
| Something else | Copy the full error from the verbose log and use it to search or fix. |

After fixing, run the workflow again (same tag re-run or new version + new tag).
