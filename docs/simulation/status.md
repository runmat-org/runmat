# Current FEA Status

This page separates language support, executable native whole-study support, and validation evidence. A declaration being type-checkable does not mean the native executor runs it.

## Status Terms

| Term | Meaning |
| --- | --- |
| Checkable | The `.fea` compiler has typed syntax, semantic lowering, diagnostics, and tests for the contract. |
| Executable | The native whole-study runner can materialize and execute that exact contract locally and through its portable cluster root. |
| Validated | The formulation, backend, scale, accuracy, invalid-case, independent-reference, CI, and release evidence recorded in [Verification and validation](./validation) covers the stated use. |
| Unavailable | Admission fails closed; no proxy or fallback is presented as the requested result. |

## Product Workflow

| Capability | Current state |
| --- | --- |
| Typed `.fea` parsing, formatting, checking, modules, imported `.m` facts, and native LSP | Available |
| CLI `check`, `plan`, local `run`, cluster `run`, study/sweep selection, and JSON output | Available |
| Imported, constructed, derived, and pure `.m`-generated exact geometry | Available on an exact-geometry-capable host |
| STEP AP242 parameter/property inspection and explicit material adoption | Available within the documented read/evidence capabilities; unsupported regeneration fails closed |
| Deterministic Tet4/Tet10 exact-CAD meshing, quality evidence, local process execution, and cluster execution | Available |
| Typed structural whole-study planning, local/remote execution, recovery, publication, field paging, and rendering context | Available |
| Generated time histories, response spectra, and random spectra from frozen `.m` imports | Available as portable materialization tasks |
| Browser `.fea` checking | Available |
| Browser geometry/meshing/solver execution | Unavailable; browser capability reporting is check-only |

## Whole-Study Physics Boundary

The typed native `.fea` executor currently admits structural procedure graphs. Structural static, modal, direct/modal transient, direct/modal harmonic, response-spectrum, random-response, prestress-dependent dynamics, and buckling have typed planning/execution contracts. Each formulation and execution envelope carries its own validation evidence.

The compiled-study executor also supports a thermal prefix → same-mesh thermomechanical
preload → buckling, prestressed modal, direct harmonic or modal-reduced harmonic consumer.
Local and remote publication retain canonical thermal fields, authored observation catalogs,
selectively readable histories and finalized reductions. This narrow coupled envelope does not
admit standalone thermal, arbitrary thermal dependencies or bidirectional coupling. Explicit
authored thermal domains and `thermomechanical preload` lowering are implemented. The
[thermal-prestress modal fixture](./examples/thermal-prestress.fea) passes actual-CAD
execution after pre-execution driver/worker replacement and cold import from declared objects,
including foreign-source and missing-object rejection. Full outer-root consumer/order coverage
and genuine mid-phase restart remain open.

Electromagnetic studies run in the typed native executor: steady current flow, electrostatics
and magnetostatics (`electromagnetic static {}`), time-harmonic fields at one or several
frequencies (`electromagnetic harmonic { frequencies = [...] }`, one case per frequency), a
harmonic loss heating a thermal analysis (`heat_source = analysis.<harmonic>.state.<name>`), and
two-way electrothermal analysis (`electrothermal steady {}` / `electrothermal transient {...}`,
published as a heat-transfer case and a conduction case). Authored outputs and reductions over
their native fields are published. See
[electromagnetic validation](./validation/electromagnetic) for the analytic checks.

Fluid flow (steady and transient), conjugate heat transfer (steady and transient with steady
flow) and fluid–structure interaction (steady and transient) run through the typed native
whole-study executor on conformal Tet4 meshes; see [fluid flow](./physics/fluid),
[conjugate heat transfer](./physics/conjugate-heat-transfer) and
[fluid–structure interaction](./physics/fluid-structure-interaction) for their envelopes and
[fluid validation](./validation/fluid) for the reference comparisons.

`docs/simulation/examples/acoustic-cavity-harmonic.fea` is an acoustic harmonic study of a 1 m air cavity with incident pressure on its walls at 100, 200 and 400 Hz. It checks and plans as an authored acoustic contract. The native whole-study executor does not yet run this multi-frequency acoustic study, and rejects it rather than running a substitute.

## Structural Validation

The repository contains substantial structural operator, formulation, observation, history, restart, modal, frequency, spectrum, random-response, property, recovery, local/remote, and result-publication evidence. Validation status is recorded per formulation and per execution envelope. Open gates include the items recorded by the release evidence and validation documentation, including broader independent references and convergence, exact restart state where required, distributed-value redistribution/cooperative solves, release-hardware accelerator evidence, and closure of recorded result/recovery defects.

A successful run shows that its numerical acceptance gates passed; it does not by itself validate the formulation for a use. The run's numerical and evidence receipts are authoritative.

### Alpha Scientific Evidence Limits

Three registered refinement checks are outside their acceptance thresholds for the 0.7.0 alpha release:

| Check | Unresolved observation |
| --- | --- |
| Local acoustic radiation, quadratic elements | The coarse-grid velocity refinement ratio is 2.954, below the registered requirement of greater than 3. The finer-grid witness does not replace this gate. |
| Nonlinear solid, Tet4 moving yield front | One coarse accepted-history stress error is 5.43%, above the registered 5% limit. |
| Nonlinear solid, Tet10 moving yield front | Sampled internal-variable errors are not monotonic over the registered refinement sequence. Independent integral/norm checks do not substitute for the original pointwise gate. |

These are measured fixture limitations, not general error bounds or completed formulation validation. Their assertions are unchanged and remain in the ordinary geometry/analysis test group. Refinement and independent references remain necessary when assessing an engineering use.

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
runmat fea check docs/simulation/examples/structural-static.fea
runmat fea check docs/simulation/examples/structural-transient.fea
runmat fea check docs/simulation/examples/modal-response-spectrum.fea
runmat fea check docs/simulation/examples/modal-random-response.fea
```

Repository qualification additionally runs focused language, core, solver, meshing, native-runner, encrypted remote-execution, result-storage, and release-evidence gates. See [Verification and validation](./validation) for the evidence standard rather than treating one workspace test command as proof of engineering validity. Per-obligation results come from the generated evidence catalog in `docs/simulation/qualification/generated/`, regenerated by `evidence_report`; no status in this page is edited by hand from a test result.

## Update Rule

Change a status only in the same slice that adds the implementation, independent evidence, negative cases, platform coverage, CI/release gate, and documentation. State the exact formulation and execution envelope; never mark an entire family validated because one fixture passed.
