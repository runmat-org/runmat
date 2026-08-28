# runmat-meshing-execution

This crate is the narrow dependency-direction bridge from meshing-owned deterministic artifacts
to RunMat's portable execution and artifact contracts.

`runmat-meshing-core` remains authoritative for geometric stages, logical identities, chunks,
manifests, and closure validation. The execution and execution-artifact crates remain authoritative
for values, authorization, encryption context, storage, transfer, retries, fencing, and result
commit. This bridge translates between those domains without teaching the scheduler meshing
semantics. Its serial reference adapter validates host envelopes and input closures, supplies
algorithm-local cancellation/budget/progress control to meshing-owned stage kernels, and converts
validated logical records into fenced shared-artifact publications. It does not implement geometry
algorithms or select stage behavior.
