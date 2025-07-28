# RustMat

RustMat is an experimental next‑generation runtime for MATLAB code. The goal is a modern, high performance engine that feels like Octave but starts quickly and integrates with Jupyter. Every crate uses kebab‑case naming.

Key ideas:

- Tiered execution similar to V8 (baseline interpreter feeding an optimising JIT).
- Snapshotting of the standard library for fast start‑up.
- Compatibility with the Jupyter kernel protocol.

See `docs/architecture.md` for a full description of the design and
`docs/DEVELOPING.md` for contributing guidelines.

Ongoing plans and design notes are tracked in `PLAN.md`. Read it before
starting work to understand the current roadmap and past decisions.
