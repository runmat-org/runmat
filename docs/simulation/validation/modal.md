## Spectrum and modal basis

Modal verification checks both natural frequencies and the basis that represents them. Independent structural references and mesh refinement test the spectrum. Eigenproblem residuals, mass normalization and orthogonality check the computed basis against the original stiffness and mass operators. Close or repeated modes require subspace comparisons: signed vectors can change while representing the same physical motions.

Support and mass assumptions belong to the verified problem. A free body's rigid motions and a constrained structure's elastic modes require different interpretation. Fixing an arbitrary degree of freedom changes that problem. A requested mode count also does not, by itself, establish adequate coverage for a later response analysis.

## Execution and retained modes

The governed workflow must preserve frequencies, original mode vectors and their mesh/constraint mapping through admission, native execution, durable publication and cold reopening. Relative-amplitude figures are presentation views of those vectors. They are not loaded vibration amplitudes, and their displayed deformation scale is not a physical excitation.

## Bounded eigenspace search

The convergence policy declares both an iteration limit and a maximum Ritz-subspace dimension, including boundary probes. Planning reserves vectors, projected matrices, recovered fields and durable outputs against that capacity. Native execution rederives the same controls and stops expansion at the admitted bound. The standard defaults allow 256 vectors; the exploratory preset allows 64. These capacities are configurable numerical controls, not accuracy tolerances.

A mode-count cut inside a repeated eigenspace requires the complete cluster and a separated boundary probe. Reaching capacity without that evidence produces an unqualified result. The solver does not silently split a cluster or interpret the resource bound as the full physical spectrum. Increasing capacity requires a newly admitted workload under the same resource ceilings.

## Criteria and assessment records

<!-- fea:verification -->

Read assessment records with their operator identity, support conditions, mesh, requested spectrum and execution boundary. Basis integrity and a successful eigensolve support specific obligations; independent physical-reference evidence remains a separate requirement.

Explore the [modal-analysis guide](/docs/simulation/guides/modal-analysis), or read [validation methodology](/docs/simulation/validation) for the shared evidence approach.
