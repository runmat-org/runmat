# Simulation Docs Map

This folder is intentionally small and high-signal.

Canonical docs:

1. `GOAL.md` - end-state definition and binary completion criteria.
2. `STATUS.md` - current state with checkmarks.
3. `ROADMAP.md` - forward execution plan and slices.
4. `ARCHITECTURE.md` - implementation boundaries and contract rules.
5. `GOVERNANCE.md` - benchmark, readiness, CI gate policies.
6. `EM_TRACK.md` - Maxwell EM phase track and next slices.
7. `WORKLOG.md` - concise append-only progress log.

Archive policy:

- Legacy planning and long-form historical docs are moved to `ARCHIVE/` and treated as read-only context.
- New work should update canonical docs, not archived docs.

Editing rules:

- One concept per file.
- No repeated phase histories across files.
- Any simulation slice should update:
  - `STATUS.md` (if checkmarks changed),
  - `WORKLOG.md` (one concise bullet).
- `GOAL.md` changes must be rare and intentional.
