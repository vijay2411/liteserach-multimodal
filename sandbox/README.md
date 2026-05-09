# SemanticsD Sandbox

This directory holds seed files for SemanticsD development. Run `make dev-sandbox`
from the repo root and the daemon will index this folder (and only this folder).

State for the sandboxed daemon (config, db, logs) lives in `./sandbox/.semanticsd/`,
not in `~/Library/Application Support/semanticsd/`. Untracked.
