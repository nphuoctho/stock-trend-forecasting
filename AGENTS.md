# Project Agent Instructions

## Environment

- This project uses `uv` and requires Python `>=3.12` from `pyproject.toml`.
- Run commands from the directory containing `pyproject.toml`.
- Use `uv sync` to create or update `.venv`, and use `uv run` for Python commands.
- Do not use a global virtual environment or invoke `pip` directly. For an ad hoc package, use `uv pip install --python .venv <package>`.

## GitHub Workflow

- Repository: `https://github.com/nphuoctho/stock-trend-forecasting`.
- `main` is the primary branch. `develop` is the integration branch.
- Before creating a branch, commit, push, pull request, or merge, run `gh api user --jq .login`. It must return exactly `nphuoctho`; otherwise STOP.
- The effective Git identity must be `user.name=nphuoctho` and must have a non-empty `user.email` before committing.
- Never commit, push feature changes, or merge directly into `main` or `develop`. Always create a new branch from `develop`, push only that branch, and create the pull request with `gh pr create --base develop --head <branch>`.
- The only pull request allowed to target `main` is a release pull request from `develop` to `main`. Never create a feature, fix, or maintenance pull request directly into `main`.
- During the first repository bootstrap, `develop` may be created as an unchanged pointer from `main`; all project changes must still enter `develop` through a pull request.
- Before merging, verify the actual pull request base and head with `gh pr view <number> --json baseRefName,headRefName`. Only `feature-or-fix -> develop` and `develop -> main` are valid.
- Keep branch names, commit messages, pull request titles, and pull request bodies clear and human-readable. Use plain ASCII wording without AI references, model names, bot markers, emojis, or generated-by metadata.
- Use the installed `ak:git` workflow for commit, push, pull request, and merge operations. Do not use another GitHub account or a virtual identity.
