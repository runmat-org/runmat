//! Standing heat-equation requirements: standalone thermal domains, source realization,
//! physical recovery, independent accuracy and governed execution. Thermoelastic,
//! electrothermal and conjugate interfaces retain separate coupling acceptance.
use crate::requirement::*;
const fn scopes(execution: ExecutionEnvelope) -> [Envelope; 4] {
    [
        envelope(Formulation::ThermalSteady, MeshOrder::Tet4, execution),
        envelope(Formulation::ThermalSteady, MeshOrder::Tet10, execution),
        envelope(Formulation::ThermalTransient, MeshOrder::Tet4, execution),
        envelope(Formulation::ThermalTransient, MeshOrder::Tet10, execution),
    ]
}
const KERNEL: &[Envelope] = &scopes(ExecutionEnvelope::NumericalKernel);
const PUBLIC: &[Envelope] = &scopes(ExecutionEnvelope::PublicStudy);
const NATIVE: &[Envelope] = &scopes(ExecutionEnvelope::NativeProcess);
const DURABLE: [Envelope; 4] = scopes(ExecutionEnvelope::Durable);
const REMOTE: [Envelope; 4] = scopes(ExecutionEnvelope::RemoteExecution);
const PARTITIONED: [Envelope; 4] = scopes(ExecutionEnvelope::DomainDecomposed);
const ACCELERATOR: [Envelope; 4] = scopes(ExecutionEnvelope::ClaimedAccelerator);
const EXECUTION: &[Envelope] = &[
    PUBLIC[0],
    PUBLIC[1],
    PUBLIC[2],
    PUBLIC[3],
    NATIVE[0],
    NATIVE[1],
    NATIVE[2],
    NATIVE[3],
    DURABLE[0],
    DURABLE[1],
    DURABLE[2],
    DURABLE[3],
    REMOTE[0],
    REMOTE[1],
    REMOTE[2],
    REMOTE[3],
    PARTITIONED[0],
    PARTITIONED[1],
    PARTITIONED[2],
    PARTITIONED[3],
    ACCELERATOR[0],
    ACCELERATOR[1],
    ACCELERATOR[2],
    ACCELERATOR[3],
];
const STEADY_PUBLIC: &[Envelope] = &[PUBLIC[0], PUBLIC[1]];
const TRANSIENT_PUBLIC: &[Envelope] = &[PUBLIC[2], PUBLIC[3]];
const FEA_THERMAL_DOMAIN_001: Requirement = Requirement {
    id: "FEA-THERMAL-DOMAIN-001",
    title: "Independent thermal domain and regional heat-equation assembly",
    stages: &[Stage::Thermal],
    acceptance: concat!(
        "Heat transfer owns an independent typed domain and initial-temperature ",
        "contract. Assemble physical element conductivity, capacity and heat sources ",
        "from resolved material regions on the canonical topology; steady and ",
        "transient operators remain distinct.",
    ),
    owners: &["crates/runmat-analysis/fea/src/physics/thermal/assembly.rs"],
    dependencies: &[],
    obligations: &[
        obligation(
            "domain",
            concat!(
                "Declare thermal reference temperature, initial temperature and compatible ",
                "material capability explicitly; reject missing, ambiguous or incompatible ",
                "domains, targets and initial-field lineage.",
            ),
            EvidenceKind::SemanticRealization,
            PUBLIC,
        ),
        obligation(
            "assembly",
            concat!(
                "Integrate conductivity, density times heat capacity, and volumetric sources ",
                "per element and material region using shared topology, properties, algebra ",
                "and constraints; preserve conductivity nullspace and integrated capacity.",
            ),
            EvidenceKind::IndependentReference,
            KERNEL,
        ),
        obligation(
            "constraints",
            concat!(
                "Apply prescribed, tied and periodic temperature constraints and supported ",
                "affine temperature jumps through the shared exact constraint authority. ",
                "Steady components require physical anchoring or an explicitly supported ",
                "nullspace treatment.",
            ),
            EvidenceKind::SemanticRealization,
            KERNEL,
        ),
    ],
};
const FEA_THERMAL_INPUT_001: Requirement = Requirement {
    id: "FEA-THERMAL-INPUT-001",
    title: "Exact thermal boundaries and shared excitation realization",
    stages: &[Stage::Thermal],
    acceptance: concat!(
        "Prescribed temperature, heat flux, convection and documented radiation act ",
        "on exact authored faces. Explicit, generated and field-artifact thermal ",
        "histories share typed signal and realization authority and equivalent ",
        "governed source and boundary energy.",
    ),
    owners: &["crates/runmat-analysis/fea/src/physics/thermal/assembly.rs"],
    dependencies: &[],
    obligations: &[
        obligation(
            "boundaries",
            concat!(
                "Resolve exact face targets and integrate prescribed temperature, ",
                "inward/outward heat-flux convention, convection and radiation laws with ",
                "documented units, normals and nonlinear domain restrictions.",
            ),
            EvidenceKind::IndependentReference,
            KERNEL,
        ),
        obligation(
            "histories",
            concat!(
                "Realize time-, parameter- and field-dependent heat generation, heat flux, ",
                "ambient temperature and prescribed temperature through shared signals and ",
                "authenticated realized field artifacts; reject unsupported samples and ",
                "incompatible fields.",
            ),
            EvidenceKind::SemanticRealization,
            PUBLIC,
        ),
        obligation(
            "equivalence",
            concat!(
                "Explicit, generated and field-artifact versions of equivalent inputs yield ",
                "equivalent governed source and boundary energy under the same physical and ",
                "numerical controls.",
            ),
            EvidenceKind::ExecutionEquivalence,
            PUBLIC,
        ),
    ],
};
const FEA_THERMAL_SOLVE_001: Requirement = Requirement {
    id: "FEA-THERMAL-SOLVE-001",
    title: "Distinct steady and transient solves with nonlinear closure",
    stages: &[
        Stage::Thermal
    ],
    acceptance: concat!(
"Use real heat-equation solves rather than graph diffusion or relaxation ",
"proxies. Document transient integration and nonlinear treatment of ",
"temperature-dependent coefficients and radiation. Retain actual residual, ",
"increment and energy-closure evidence before acceptance.",
),
    owners: &[
        "crates/runmat-analysis/fea/src/pipeline/thermal/physical.rs"
    ],
    dependencies: &[
    ],
    obligations: &[
        obligation(
            "steady",
            concat!(
"Solve stationary conductivity/source/boundary equilibrium and verify the ",
"physical residual on the original operator.",
),
            EvidenceKind::IndependentReference,
            STEADY_PUBLIC
        ),
        obligation(
            "transient",
            concat!(
"Document the transient integration method, initial-state semantics and ",
"sampling; demonstrate its temporal order independently of spatial error.",
),
            EvidenceKind::TemporalConvergence,
            TRANSIENT_PUBLIC
        ),
        obligation(
            "nonlinear",
            concat!(
"Temperature-dependent property and radiation iterations meet documented ",
"physical residual, field-increment and energy-closure criteria. Failed or ",
"exhausted iterations cannot produce an accepted state.",
),
            EvidenceKind::AccuracyAcceptance,
            PUBLIC
        ),
    ],
};
const FEA_THERMAL_FIELDS_001: Requirement = Requirement {
    id: "FEA-THERMAL-FIELDS-001",
    title: "Physical thermal recovery and energy accounting",
    stages: &[
        Stage::Thermal
    ],
    acceptance: concat!(
"Publish original nodal and integration-point temperature, temperature ",
"gradient, heat flux, actual authored/generated heat source and ",
"boundary-face heat transfer with correct units and topology. Compute global ",
"and per-region energy balances from the same physical transfers used in ",
"assembly.",
),
    owners: &[
        "crates/runmat-analysis/fea/src/physics/thermal/assembly/recovery.rs"
    ],
    dependencies: &[
    ],
    obligations: &[
        obligation(
            "fields",
            concat!(
"Retain recoverable original thermal fields, integration inventories and ",
"boundary-face transfers with explicit locations, components, units and ",
"canonical topology; recover physical conductivity times temperature ",
"gradient with the documented sign.",
),
            EvidenceKind::OriginalStateComparison,
            PUBLIC
        ),
        obligation(
            "global_energy",
            concat!(
"Account for supplied heat, boundary transfer, prescribed reactions and ",
"transient stored energy without counting internal conduction as external ",
"power or hiding opposing flows through cancellation.",
),
            EvidenceKind::PhysicalBalance,
            KERNEL
        ),
        obligation(
            "regional_energy",
            concat!(
"Compute per-region energy balances and consistent interface exchange so ",
"regional totals reconcile with the global balance.",
),
            EvidenceKind::PhysicalBalance,
            PUBLIC
        ),
    ],
};
const FEA_THERMAL_ACCURACY_001: Requirement = Requirement {
    id: "FEA-THERMAL-ACCURACY-001",
    title: "Independent thermal references and refinement qualification",
    stages: &[Stage::Thermal],
    acceptance: concat!(
        "Thermal qualification requires independent linear-temperature patches, ",
        "internally heated slabs, plates and cylinders, convection cases and ",
        "lumped-capacity limits, with spatial convergence, temporal convergence and ",
        "energy closure. Retain references, meshes, controls and original acceptance ",
        "limits; operator residual alone is insufficient.",
    ),
    owners: &["crates/runmat-analysis/fea/src/execution/accuracy_source.rs"],
    dependencies: &[],
    obligations: &[
        obligation(
            "references",
            concat!(
                "Pass independent patch, internally heated slab, plate and cylinder, ",
                "convection and lumped-capacity references across the supported orders and ",
                "physical boundary/source regimes.",
            ),
            EvidenceKind::IndependentReference,
            PUBLIC,
        ),
        obligation(
            "space",
            concat!(
                "Demonstrate spatial convergence of temperature, gradient, flux, stored heat ",
                "and boundary transfer against independent continuum references while ",
                "bounding temporal contamination.",
            ),
            EvidenceKind::SpatialConvergence,
            PUBLIC,
        ),
        obligation(
            "time",
            concat!(
                "Demonstrate temporal convergence against independent continuum references ",
                "on spatially controlled meshes, including heating and cooling and nonlinear ",
                "boundary regimes.",
            ),
            EvidenceKind::TemporalConvergence,
            TRANSIENT_PUBLIC,
        ),
        obligation(
            "gate",
            concat!(
                "Mandatory thermal accuracy assessments bind original fields, energy ",
                "diagnostics and realization evidence to the result and block production ",
                "acceptance when missing, failed or indeterminate.",
            ),
            EvidenceKind::AccuracyAcceptance,
            PUBLIC,
        ),
    ],
};
const FEA_THERMAL_EXECUTION_001: Requirement = Requirement {
    id: "FEA-THERMAL-EXECUTION-001",
    title: "Governed thermal studies, persistence and execution evidence",
    stages: &[Stage::Thermal],
    acceptance: concat!(
        "Thermal studies support parallel assembly and parameter studies, bounded ",
        "resource admission, standalone operation classification, original field ",
        "persistence and supported restart. Every claimed execution envelope records ",
        "actual CPU, tensor/device and fallback behavior; task distribution is ",
        "distinguished from partitioned solve execution.",
    ),
    owners: &["crates/runmat-analysis/fea/src/execution/thermal_workload.rs"],
    dependencies: &[
        concat!(
            "Thermoelastic transitions separately validate temperature reference, ",
            "transfer/projection, integration-point thermal strain, mechanical ",
            "equilibrium and sequence lineage, including supported prestressed modal, ",
            "harmonic and buckling branches.",
        ),
        concat!(
            "Two-way thermoelastic policy is required only for its supported envelope ",
            "and must establish field and energy closure; electrothermal and conjugate ",
            "heat-transfer interfaces require independent coupling qualification.",
        ),
    ],
    obligations: &[
        obligation(
            "admission",
            concat!(
                "Admit source/initial-field realization, physical assembly, solve, retained ",
                "fields, receipts and publication before the respective work; reject ",
                "shortages and substituted budgets and record requested resources separately ",
                "from measured usage.",
            ),
            EvidenceKind::ResourceAdmission,
            EXECUTION,
        ),
        obligation(
            "execution",
            concat!(
                "Demonstrate parallel assembly and parameter-study execution equivalence; ",
                "qualify actual native, durable, encrypted remote, partitioned and claimed ",
                "accelerator behavior. Host fallback cannot satisfy device execution and ",
                "fanout cannot satisfy a distributed solve.",
            ),
            EvidenceKind::ExecutionEquivalence,
            EXECUTION,
        ),
        obligation(
            "publication",
            concat!(
                "Classify standalone thermal operation versions correctly and preserve ",
                "original thermal result, reference, topology, source and accuracy lineage ",
                "through cold receipt/state publication; reject corruption, incompatible ",
                "restart and partial publication.",
            ),
            EvidenceKind::PublicationIntegrity,
            EXECUTION,
        ),
        obligation(
            "delivery",
            concat!(
                "Provide release evidence and reproducible examples documenting ",
                "formulations, boundaries, integration method, nonlinear closure, units, ",
                "accuracy limits and supported execution envelopes.",
            ),
            EvidenceKind::AccuracyAcceptance,
            PUBLIC,
        ),
    ],
};
pub(crate) const REQUIREMENTS: &[Requirement] = &[
    FEA_THERMAL_DOMAIN_001,
    FEA_THERMAL_INPUT_001,
    FEA_THERMAL_SOLVE_001,
    FEA_THERMAL_FIELDS_001,
    FEA_THERMAL_ACCURACY_001,
    FEA_THERMAL_EXECUTION_001,
];
