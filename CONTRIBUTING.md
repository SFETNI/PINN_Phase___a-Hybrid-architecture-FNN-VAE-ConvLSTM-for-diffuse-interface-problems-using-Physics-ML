# Contributing

Contributions are welcome through focused issues and pull requests.

1. Create a Python 3.11 environment.
2. Install the project with `python -m pip install -e ".[dev]"`.
3. Run `python -m pytest` before opening a pull request.
4. Add tests for behavioral changes and document any public API change.

Scientific changes must state which architecture family they affect and must
not change a published benchmark configuration silently. New artifacts need a
SHA-256 manifest, provenance, and a clear redistribution license.

