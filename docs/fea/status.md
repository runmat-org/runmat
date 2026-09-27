# Current FEA Status

This page separates language support, executable native whole-study support, and production qualification. A declaration being type-checkable does not mean its solver is admitted for a production run.

## Status Terms

| Term | Meaning |
| --- | --- |
| Checkable | The `.fea` compiler has typed syntax, semantic lowering, diagnostics, and tests for the contract. |
| Executable | The native whole-study runner can materialize and execute that exact contract locally and through its portable cluster root. |
| Experimental | Executable only when the study explicitly accepts an experimental formulation. |
| Production | Every required formulation, backend, scale, accuracy, invalid-case, independent-reference, CI, and release evidence gate is satisfied. |
| Unavailable | Admission fails closed; no proxy or fallback is presented as the requested result. |

## Product Workflow

| Capability | Current state |
| --- | --- |
| Typed `.fea` parsing, formatting, checking, modules, imported `.m` facts, and native LSP | Available |
| CLI `check`, `plan`, local `run`, cluster `run`, study/sweep selection, and JSON output | Available |
| Imported, constructed, derived, and pure `.m`-generated exact geometry | Available on an exact-geometry-capable host |
| STEP AP242 parameter/property inspection and explicit material adoption | Available within the documented read/evidence capabilities; unsupported regeneration fails closed |
| Deterministic Tet4/Tet10 exact-CAD meshing, quality evidence, local process execution, and cluster execution | Available |
| Typed structural whole-study planning, local/remote execution, recovery, publication, field paging, and rendering context | Available within the explicit formulation requirement |
| Generated time histories, response spectra, and random spectra from frozen `.m` imports | Available as portable materialization tasks |
| Browser `.fea` checking | Available |
| Browser geometry/meshing/solver execution | Unavailable; browser capability reporting is check-only |

## Whole-Study Physics Boundary

The typed native `.fea` executor currently admits structural procedure graphs. Structural static, modal, direct/modal transient, direct/modal harmonic, response-spectrum, random-response, prestress-dependent dynamics, and buckling have typed planning/execution contracts, but each formulation retains its own qualification status and may require `experimental`.

The experimental compiled-study executor also supports a thermal prefix → same-mesh thermomechanical
preload → buckling, prestressed modal, direct harmonic or modal-reduced harmonic consumer.
Local and remote publication retain canonical thermal fields, authored observation catalogs,
selectively readable histories and finalized reductions. This narrow coupled envelope does not
admit standalone thermal, arbitrary thermal dependencies or bidirectional coupling. Explicit
authored thermal domains and `thermomechanical preload` lowering are implemented. The
[thermal-prestress modal fixture](./examples/thermal-prestress-experimental.fea) passes actual-CAD
execution after pre-execution driver/worker replacement and cold import from declared objects,
including foreign-source and missing-object rejection. Full outer-root consumer/order coverage
and genuine mid-phase restart remain open; this is not production qualification.

Acoustic, electromagnetic, fluid, electro-thermal, conjugate heat-transfer and fluid-structure
procedure contracts are represented and checked by the language but are not yet admitted by the
typed native whole-study executor. Runtime object-level operations and solver components are not
a substitute for that complete path.

`docs/fea/examples/acoustic-production-rejected.fea` is a fail-closed example: it checks as an authored contract and rejects execution because the requested production family is unavailable.

## Structural Qualification

The repository contains substantial structural operator, formulation, observation, history, restart, modal, frequency, spectrum, random-response, property, recovery, local/remote, and result-publication evidence. Production status remains per formulation and per execution envelope. Open gates include the items recorded by the release evidence and validation documentation, including broader independent references and convergence, exact restart state where required, distributed-value redistribution/cooperative solves, release-hardware accelerator evidence, and closure of recorded result/recovery defects.

Do not infer production suitability from a successful experimental run. The study's `formulation` setting and the resulting numerical and evidence receipts are authoritative.

## Meshing Boundary

Exact meshing is available for admitted STEP, IGES, and BREP solids through the qualified OCCT capability. Direct and in-study meshing share canonical request and evidence contracts. The `.fea` language currently exposes global tetrahedral order, target size, chordal deviation, grading ratio, and deterministic seed; it does not expose boundary layers, local refinement fields, hybrid element families, or user-selectable algorithms.

## Known Open Defects

The current native-runner audit records three failing cases that block relevant gates:

- an explicit spatial observation requested without the canonical solver mesh;
- duplicate recovered semantic samples;
- a missing buckling mode-shape field in recovery/publication.

These are treated as product defects, not ignored tests or acceptable limitations.

## Verification Commands

Representative documentation/source checks:

```sh
runmat fea check docs/fea/examples/structural-static-experimental.fea
runmat fea check docs/fea/examples/structural-transient-experimental.fea
runmat fea check docs/fea/examples/modal-response-spectrum-experimental.fea
runmat fea check docs/fea/examples/modal-random-response-experimental.fea
```

Repository qualification additionally runs focused language, core, solver, meshing, native-runner, encrypted remote-execution, result-storage, and release-evidence gates. See [Verification and validation](./validation) for the evidence standard rather than treating one workspace test command as proof of engineering validity.

## Update Rule

Change a status only in the same slice that adds the implementation, independent evidence, negative cases, platform coverage, CI/release gate, and documentation. State the exact formulation and execution envelope; never promote an entire family because one fixture passed.
