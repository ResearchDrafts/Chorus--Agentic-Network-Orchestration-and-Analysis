"""Command-line entry points for the RQ1 pipeline.

A package rather than loose files so the tests can import the entry points
and drive them against a mocked gateway. Every module here is also runnable
directly (`python scripts/run_e1.py`), which is how the notebook calls them;
the shared machinery lives in `sandbox/single_exposure.py` so neither form
carries logic the other does not.

Deliberately excluded from `[tool.setuptools.packages.find]`: these are
scripts run from a checkout, not library code something imports after a
`pip install`.
"""
