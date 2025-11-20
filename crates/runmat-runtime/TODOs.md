- [ ] When revisiting builtin constructors (zeros/ones/logspace variants, range helpers, etc.), wire them through `builtins::common::residency::sequence_gpu_preference` so they share the auto-offload-aware heuristics.
- [ ] `fprintf`/`sprintf`: accept tensor-backed formatSpec/char inputs reliably (current runtime rejects 1-D char tensors and breaks benchmark logging). Map to `String` before calling formatter and add tests.

