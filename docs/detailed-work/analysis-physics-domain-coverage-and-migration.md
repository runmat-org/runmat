# Analysis Physics Domain Coverage and Target Shape

This document defines the target physics-domain shape for analysis study schemas and runtime ownership.

Companion docs:

- workflow contract: `docs/detailed-work/code-reviewed-fea-workflow.md`
- program context: `docs/detailed-work/geo-and-analysis.md`

## Product Position

RunMat today is strongest in structural mechanics and partially implemented in multiphysics branches.

| Area | Status today | Notes |
| --- | --- | --- |
| Linear static structural | implemented | Core structural path. |
| Modal structural | implemented | Contracted and tested. |
| Transient structural | implemented | Contracted and tested. |
| Nonlinear structural | implemented | Contracted with diagnostics/governance. |
| Thermo-mechanical coupling | partial | Not fully material-owned yet. |
| Electro-thermal coupling | partial | Not a full EM domain. |
| Plasticity | partial/proxy | Transitioning to constitutive material ownership. |
| Contact | partial/proxy | Transitioning to model-owned interface/contact ownership. |
| Full standalone thermal domain | not complete | Missing full thermal constitutive ownership + dedicated thermal pathway. |
| Full EM domain (Maxwell-class) | not implemented | Out of scope for near-term "most engineers" target. |

## Hard-Cutover Rule

This design is a hard cutover target, not a backward-compat migration path.

- treat the model-owned physics shape as the canonical `v1`,
- do not preserve legacy proxy-first schema variants as long-term contracts,
- run options remain numeric/execution controls, not the primary home of constitutive physics.

## Canonical Ownership by Domain

| Domain | Canonical owner |
| --- | --- |
| Mechanical elastic | `model.materials[]` |
| Thermal constitutive properties | `model.materials[].thermal` |
| Electrical constitutive properties | `model.materials[].electrical` |
| Plastic constitutive model | `model.materials[].plastic` |
| Contact interface model | `model.interfaces[].contact` |
| Loads/BC/steps | `model.loads`, `model.boundary_conditions`, `model.steps` |

## Target Capability Set for Most Engineers

1. Structural baseline: linear static, modal, transient, nonlinear.
2. Thermal-structural: material-owned thermal properties with thermo-mechanical coupling.
3. Electro-thermal (engineering level): material-owned electrical + thermal properties with Joule-heating style coupling.
4. Practical nonlinear: constitutive plasticity and interface-owned contact (frictionless + frictional baseline).
5. Stable inspection + governance: `run.data` arrays, diagnostics, readiness/trend policy by domain.

Not required for this milestone:

- full Maxwell-class EM multiphysics,
- niche constitutive families beyond common industrial baselines.

## What It Takes to Complete the Set

### Phase 1: Schema Ownership Completion

- finalize study/model schema blocks for `thermal`, `electrical`, `plastic`, and `interfaces.contact`.
- remove physics-defining meaning from run-option proxy fields.
- enforce validation that coupled studies provide required model-owned physics fields.

Exit criteria:

- canonical studies express thermo/electro/plastic/contact in model-owned blocks,
- validation rejects missing required physics blocks for enabled domains.

### Phase 2: Runtime Consumption Completion

- source solver multiphysics contexts from model-owned fields by default,
- keep run options as numeric/execution settings only,
- verify diagnostics show model-owned context fingerprints.

Exit criteria:

- solver runs consume model-owned physics definitions end-to-end,
- domain fixtures pass with target-shape studies.

### Phase 3: Inspection/Data Completion

- require run-kind/domain-specific arrays in `run.data`,
- keep JSON envelopes governance-focused and `run.data` inspection-focused.

Exit criteria:

- desktop/script inspection flows work consistently across structural + target multiphysics domains,
- array names/attrs remain stable for automation.

### Phase 4: Governance Completion

- add strict branch checks for target domain readiness,
- require model-owned domain evidence for protected branches.

Exit criteria:

- protected branch policy enforces target-shape physics ownership and domain quality gates.
