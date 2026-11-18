# Progress log

(Claude adds a short entry here after each task.)

- 2026-11-17 — README: added Windows (PowerShell) setup/run/test commands beside the Mac/Linux ones, and a separate "Test it" section. Docs only, no tests needed; suite passes (58). Windows run uses `python -m uvicorn` (works without relying on the uvicorn.exe shim); I verified the module imports but did not launch the server.
