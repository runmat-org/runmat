# Agent Instructions

## No Allow

- Avoid adding `#[allow(...)]` attributes or lint suppressions.
- If an allow is absolutely unavoidable, exhaust alternatives first and add a brief inline comment explaining why the allow is necessary at that location.

## Documentation Co-Updates

- When making changes to an area of the codebase, update `docs/*.md` files if any relevant documentation has changed.