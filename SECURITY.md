# Security policy

## Supported versions

toolproof is pre-1.0. Only the latest released version receives security
fixes. There are no backports to earlier 0.x releases.

| Version | Supported |
| --- | --- |
| latest 0.x | yes |
| anything older | no |

## Reporting a vulnerability

Report privately through GitHub's
[private vulnerability reporting](https://github.com/priyank-agrawal/toolproof/security/advisories/new)
on this repository. Do not open a public issue for a vulnerability.

Please include the version, Python version, a minimal reproduction, and what
an attacker gains. You will get an acknowledgement within 7 days and a status
update within 30 days. If a fix ships, you will be credited in the advisory
and in `CHANGELOG.md` unless you ask otherwise.

## What is in scope

This library captures traces of agent runs and writes them to disk. The
security properties it claims, and therefore the things worth reporting:

- **Redaction runs before the sink.** Values matching the configured redaction
  keys must never reach a sink, a report, or a log line. A path where a
  redacted value is written out is a vulnerability, not a bug.
- **Secrets are never recorded.** Provider API keys, authorization headers and
  credentials must not appear in any span, trace, report or error message.
- **Tracebacks are never recorded.** The tool wrapper records an exception's
  class name only, because traceback text carries file paths and sometimes
  argument values.
- **Judge input is data, not instructions.** Claim and evidence text reaching a
  judge model is delimited and the prompt instructs the model to ignore
  instructions inside it. A reproducible prompt injection that flips a verdict
  is in scope; the calibration suite carries fixtures for this.
- **Eval runs touch nothing real.** Tools tagged `side_effect=True` must be
  stubbed or the runner refuses to start, and an injected fault must never
  reach the real dependency. A path that sends a real email or writes to a
  real database during an eval run is in scope.
- **Replay mode makes no network calls.** A cache miss in replay mode is a hard
  error, never a silent live call.
- **No code execution from data.** Loading a dataset, a trace, a report or a
  JSON Schema must not execute code from that file.

## What is out of scope

- The security of models, providers or third-party services you point this
  library at.
- Anything requiring an attacker who already controls the machine, the Python
  environment, or the config file, since the config file names an import path
  to execute by design.
- Committing your own traces or reports containing real user data. Redaction
  is configurable and defaults are not a substitute for reviewing what you
  commit; dogfood datasets use synthetic personas only.
- Denial of service through deliberately enormous input files.

## Supply chain

Releases are published to PyPI through GitHub Actions using
[trusted publishing](https://docs.pypi.org/trusted-publishers/) over OIDC. No
PyPI API token exists in this repository's secrets. Every release is built from
a tagged commit and its artifacts are attached to the corresponding GitHub
release, so a wheel on PyPI can be checked against one built from the tag.
