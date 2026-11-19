# Rules for unattended overnight work

You are running unattended. Nobody can answer questions. Follow these rules exactly.

## Git
- Stay on the current branch (`overnight`). Never switch branches, never push,
  never rebase, never force anything, never reset to an earlier commit.
- One commit per finished task, with a clear message.

## Scope
- Only touch files inside this repo.
- Do not add dependencies or edit the dependency lists in pyproject.toml.
- Do not edit static/plotly.min.js.
- Keep existing API routes and request/response fields working. Adding new
  fields is fine; renaming or removing them is not.

## Tests
- Run tests with: .venv\Scripts\python -m pytest -q
- Tests must stay offline. Never call Yahoo Finance or any network service.
  Use synthetic prices like the fixtures in tests/conftest.py.
- Never change an existing test's expected values to make it pass. If an
  existing test breaks, fix your code or revert your change.
- Every task ends with the full test suite passing, or it gets reverted.

## When stuck
- After 3 failed attempts at a task, revert your code changes, mark the item
  BLOCKED in TODO.md with a one-line reason, note it in PROGRESS.md, commit,
  and stop.
