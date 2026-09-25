# Contributing

Use Windows 11 and Python 3.11 for supported development. Create an isolated environment with `setup.ps1`, make focused changes, and run `test.ps1` before opening a pull request.

Every geometry change must preserve the distinction between observed, georeferenced, and inferred products. Tests and reports may contain only computed metrics. Large videos, model weights, mission outputs, credentials, and sensitive location data must not be committed.

Pull requests should describe the input conditions tested, commands executed, measured results, and remaining limitations. New reconstruction backends need a documented license, checkpoint source, hardware envelope, deterministic configuration, and an integration test that can be skipped when the backend is unavailable.

