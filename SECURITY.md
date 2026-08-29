# Security policy

Please use the repository host's private vulnerability-reporting facility for
suspected security issues. If that facility is unavailable, open a public issue
requesting a private contact route without including vulnerability details.
Never include credentials, confidential data, or exploit details in a public
issue.

PINN-Phase treats model checkpoints and NumPy archives as untrusted inputs.
Public loaders disable pickle-based NumPy loading, use restricted PyTorch
checkpoint loading, verify expected SHA-256 digests, and validate payload
structure before applying model state.

## Supported versions

Security updates are provided for the latest tagged release. Development
snapshots receive fixes on the default branch but are not stable releases.

PyTorch 2.6 or newer is mandatory because earlier releases are affected by
CVE-2025-32434. Restricted loading reduces code-execution risk but does not
make arbitrary artifacts harmless; the loaders therefore also enforce file,
archive, and payload limits.

