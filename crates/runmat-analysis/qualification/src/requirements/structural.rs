//! Standing acceptance for linear solids, beams, shells and mixed structures.
use crate::requirement::*;
const DOMAIN: &[Envelope] = &[
    envelope(
        Formulation::LinearStructural,
        MeshOrder::Tet4,
        ExecutionEnvelope::NumericalKernel,
    ),
    envelope(
        Formulation::LinearStructural,
        MeshOrder::Tet10,
        ExecutionEnvelope::NumericalKernel,
    ),
    envelope(
        Formulation::LinearStructural,
        MeshOrder::Beam,
        ExecutionEnvelope::NumericalKernel,
    ),
    envelope(
        Formulation::LinearStructural,
        MeshOrder::Shell,
        ExecutionEnvelope::NumericalKernel,
    ),
    envelope(
        Formulation::LinearStructural,
        MeshOrder::Tet4,
        ExecutionEnvelope::PublicStudy,
    ),
    envelope(
        Formulation::LinearStructural,
        MeshOrder::Tet10,
        ExecutionEnvelope::PublicStudy,
    ),
    envelope(
        Formulation::LinearStructural,
        MeshOrder::Beam,
        ExecutionEnvelope::PublicStudy,
    ),
    envelope(
        Formulation::LinearStructural,
        MeshOrder::Shell,
        ExecutionEnvelope::PublicStudy,
    ),
    envelope(
        Formulation::LinearStructural,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::NativeProcess,
    ),
    envelope(
        Formulation::LinearStructural,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::Durable,
    ),
    envelope(
        Formulation::LinearStructural,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::RemoteExecution,
    ),
    envelope(
        Formulation::LinearStructural,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::ParallelCpu,
    ),
    envelope(
        Formulation::LinearStructural,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::DomainDecomposed,
    ),
    envelope(
        Formulation::LinearStructural,
        MeshOrder::CanonicalMixedTopology,
        ExecutionEnvelope::ClaimedAccelerator,
    ),
];
const FEA_STRUCTURAL_SOLID_001: Requirement = Requirement {
    id: "FEA-STRUCTURAL-SOLID-001",
    title: "Physical linear solid operators, loads and assigned-material recovery",
    stages: &[Stage::LinearStructures],
    acceptance: concat!(
        "Physical linear solid operators, loads and assigned-material recovery must ",
        "satisfy all applicable supported-envelope criteria. Independent physical, ",
        "semantic, accuracy, resource and publication evidence are separate; ",
        "missing, failed or indeterminate mandatory evidence prevents production ",
        "acceptance.",
    ),
    owners: &["crates/runmat-analysis/fea/src/assembly/solver_solid.rs"],
    obligations: &[
        obligation(
            "quadrature_mass",
            concat!(
                "Complete Tet4 verification and retain it as the initial serial reference. ",
                "Complete Tet10 volume and face quadrature, recovery, stress ",
                "extrapolation/projection, pressure loading, and estimator support. Assemble ",
                "consistent mass and a documented optional lumped mass from density and ",
                "geometry.",
            ),
            EvidenceKind::IndependentReference,
            DOMAIN,
        ),
        obligation(
            "damping_body_loads",
            concat!(
                "Implement documented damping inputs, initially Rayleigh damping where ",
                "applicable. Integrate body forces by region, material density, shape ",
                "function, and volume. Implement nonzero prescribed displacement with ",
                "correct elimination or lifting and reaction recovery.",
            ),
            EvidenceKind::IndependentReference,
            DOMAIN,
        ),
        obligation(
            "material_recovery",
            concat!(
                "Make stress, strain, energy, and derived-field recovery use each element's ",
                "assigned material. Define treatment of discontinuous stress at material ",
                "interfaces. Preserve exact force and moment resultants for distributed ",
                "surface and wrench loads.",
            ),
            EvidenceKind::IndependentReference,
            DOMAIN,
        ),
        obligation(
            "load_excitation_diagnostics",
            concat!(
                "Realize constant and parameterized static excitations through the shared ",
                "signal/excitation path rather than a formulation-private load vector. ",
                "Detect rigid modes, inadequate constraint, inversion, poor conditioning, ",
                "and unsupported element types with typed diagnostics.",
            ),
            EvidenceKind::IndependentReference,
            DOMAIN,
        ),
    ],
    dependencies: &[concat!(
        "Shared typed authoring, canonical topology/material/constraint/excitation ",
        "authorities; governed resources and execution; independently qualified ",
        "accuracy and semantic publication.",
    )],
};
const FEA_STRUCTURAL_BEAM_001: Requirement = Requirement {
    id: "FEA-STRUCTURAL-BEAM-001",
    title: "Physical three-dimensional beams and explicit mixed-model coupling",
    stages: &[Stage::LinearStructures],
    acceptance: concat!(
        "Physical three-dimensional beams and explicit mixed-model coupling must ",
        "satisfy all applicable supported-envelope criteria. Independent physical, ",
        "semantic, accuracy, resource and publication evidence are separate; ",
        "missing, failed or indeterminate mandatory evidence prevents production ",
        "acceptance.",
    ),
    owners: &["crates/runmat-analysis/fea/src/assembly/elements/beam.rs"],
    obligations: &[
        obligation(
            "physical_beam_mixed_dofs",
            concat!(
                "Retain the real three-dimensional beam formulation and verify ",
                "transformation, shear/bending assumptions, mass, and section recovery. ",
                "Define explicit node/constraint coupling between beams, shells, and solids. ",
                "Assemble mixed models into one global degree-of-freedom system.",
            ),
            EvidenceKind::IndependentReference,
            DOMAIN,
        ),
        obligation(
            "coupling_references",
            concat!(
                "Reject incompatible or disconnected couplings. Add frame, cantilever, ",
                "torsion, and mixed solid-beam benchmarks.",
            ),
            EvidenceKind::IndependentReference,
            DOMAIN,
        ),
    ],
    dependencies: &[concat!(
        "Shared typed authoring, canonical topology/material/constraint/excitation ",
        "authorities; governed resources and execution; independently qualified ",
        "accuracy and semantic publication.",
    )],
};
const FEA_STRUCTURAL_SHELL_001: Requirement = Requirement {
    id: "FEA-STRUCTURAL-SHELL-001",
    title: "Physical membrane and bending shells",
    stages: &[Stage::LinearStructures],
    acceptance: concat!(
        "Physical membrane and bending shells must satisfy all applicable ",
        "supported-envelope criteria. Independent physical, semantic, accuracy, ",
        "resource and publication evidence are separate; missing, failed or ",
        "indeterminate mandatory evidence prevents production acceptance.",
    ),
    owners: &["crates/runmat-analysis/fea/src/assembly/elements/shell.rs"],
    obligations: &[
        obligation(
            "physical_shell_operator",
            concat!(
                "Select and document the supported shell formulation and assumptions. ",
                "Implement membrane and bending behavior, local frames, thickness ",
                "integration, drilling stabilization, mass, and recovery. Validate patch ",
                "behavior, locking sensitivity, rigid-body modes, orientation, and mesh ",
                "convergence.",
            ),
            EvidenceKind::IndependentReference,
            DOMAIN,
        ),
        obligation(
            "proxy_removal",
            "Remove the pair-spring proxy when the production formulation lands.",
            EvidenceKind::IndependentReference,
            DOMAIN,
        ),
    ],
    dependencies: &[concat!(
        "Shared typed authoring, canonical topology/material/constraint/excitation ",
        "authorities; governed resources and execution; independently qualified ",
        "accuracy and semantic publication.",
    )],
};
const FEA_STRUCTURAL_REFERENCE_001: Requirement = Requirement {
    id: "FEA-STRUCTURAL-REFERENCE-001",
    title: "Linear structural independent references and physical balances",
    stages: &[Stage::LinearStructures],
    acceptance: concat!(
        "Linear structural independent references and physical balances must satisfy ",
        "all applicable supported-envelope criteria. Independent physical, semantic, ",
        "accuracy, resource and publication evidence are separate; missing, failed ",
        "or indeterminate mandatory evidence prevents production acceptance.",
    ),
    owners: &["crates/runmat-analysis/fea/src/pipeline/linear_static.rs"],
    obligations: &[
        obligation(
            "patch_reference_convergence",
            concat!(
                "Constant-strain and rigid-body patch tests. Cantilever, thick-cylinder, ",
                "torsion, and multi-material interface cases. Tet4/Tet10 h- and ",
                "p-convergence with expected rates.",
            ),
            EvidenceKind::IndependentReference,
            DOMAIN,
        ),
        obligation(
            "balance_independent_reference",
            concat!(
                "Load resultant, reaction balance, strain-energy, and reciprocal-work ",
                "checks. Independent reference comparisons.",
            ),
            EvidenceKind::IndependentReference,
            DOMAIN,
        ),
    ],
    dependencies: &[concat!(
        "Shared typed authoring, canonical topology/material/constraint/excitation ",
        "authorities; governed resources and execution; independently qualified ",
        "accuracy and semantic publication.",
    )],
};
const FEA_STRUCTURAL_CONSTRAINT_001: Requirement = Requirement {
    id: "FEA-STRUCTURAL-CONSTRAINT-001",
    title: "Complete typed structural constraints and constitutive connectors",
    stages: &[Stage::LinearStructures],
    acceptance: concat!(
        "Complete typed structural constraints and constitutive connectors must ",
        "satisfy all applicable supported-envelope criteria. Independent physical, ",
        "semantic, accuracy, resource and publication evidence are separate; ",
        "missing, failed or indeterminate mandatory evidence prevents production ",
        "acceptance.",
    ),
    owners: &["crates/runmat-analysis/fea/src/constraints/mod.rs"],
    obligations: &[
        obligation(
            "fixed_component_motion",
            concat!(
                "The supported structural surface provides fixed support, component-wise ",
                "prescribed translation, and component-wise prescribed rotation where ",
                "rotational DOFs exist.",
            ),
            EvidenceKind::IndependentReference,
            DOMAIN,
        ),
        obligation(
            "normal_symmetry_rigid_coupling",
            concat!(
                "The supported structural surface provides normal displacement/roller ",
                "support, symmetry/antisymmetry, and rigid coupling to a reference point.",
            ),
            EvidenceKind::IndependentReference,
            DOMAIN,
        ),
        obligation(
            "distributing_ties_equations",
            concat!(
                "The supported structural surface provides distributing reference-point ",
                "coupling, ties over matching/supported nonmatching surfaces, and general ",
                "linear multipoint equations.",
            ),
            EvidenceKind::IndependentReference,
            DOMAIN,
        ),
        obligation(
            "periodicity_releases",
            concat!(
                "The supported structural surface provides translational periodicity, cyclic ",
                "symmetry, and beam/shell releases and supported joint relations.",
            ),
            EvidenceKind::IndependentReference,
            DOMAIN,
        ),
        obligation(
            "typed_connector_inventory",
            concat!(
                "The supported structural surface provides spring, dashpot, and elastic-found",
                "ation connectors.",
            ),
            EvidenceKind::IndependentReference,
            DOMAIN,
        ),
        obligation(
            "connectors",
            concat!(
                "Connectors are typed constitutive objects and contribute to stiffness, ",
                "damping, or residual assembly. They are not represented as exact equation ",
                "rows.",
            ),
            EvidenceKind::SemanticRealization,
            DOMAIN,
        ),
    ],
    dependencies: &[concat!(
        "Shared typed authoring, canonical topology/material/constraint/excitation ",
        "authorities; governed resources and execution; independently qualified ",
        "accuracy and semantic publication.",
    )],
};
const FEA_STRUCTURAL_EXECUTION_001: Requirement = Requirement {
    id: "FEA-STRUCTURAL-EXECUTION-001",
    title: "Governed production linear structures and execution equivalence",
    stages: &[Stage::LinearStructures],
    acceptance: concat!(
        "Governed production linear structures and execution equivalence must ",
        "satisfy all applicable supported-envelope criteria. Independent physical, ",
        "semantic, accuracy, resource and publication evidence are separate; ",
        "missing, failed or indeterminate mandatory evidence prevents production ",
        "acceptance.",
    ),
    owners: &["crates/runmat-analysis/fea/src/pipeline/linear_static.rs"],
    obligations: &[
        obligation(
            "solid_execution",
            concat!(
                "Complete Tet4 correctness and evidence. Complete Tet10 assembly, loading, ",
                "recovery, and estimation. Implement physical mass, damping, body forces, ",
                "and prescribed displacement.",
            ),
            EvidenceKind::ExecutionEquivalence,
            DOMAIN,
        ),
        obligation(
            "shared_excitation_recovery_mixed",
            concat!(
                "Route static loads and parameterized excitation cases through the shared ",
                "excitation realization and prove resultant/moment preservation. Complete ",
                "per-material recovery. Complete mixed beam/solid behavior.",
            ),
            EvidenceKind::ExecutionEquivalence,
            DOMAIN,
        ),
        obligation(
            "constraints_reactions_shell",
            concat!(
                "Implement component-wise prescribed translation/rotation, normal support, ",
                "symmetry, rigid/distributing coupling, ties, MPC equations, periodicity, ",
                "releases, and supported connectors. Implement generalized reaction, ",
                "constraint work, violation, and rank evidence. Replace shell proxy and ",
                "complete mixed shell behavior.",
            ),
            EvidenceKind::ExecutionEquivalence,
            DOMAIN,
        ),
        obligation(
            "bounded_parallel_accelerator_remote",
            concat!(
                "Add resource-aware scoped shared-memory assembly, solve, and recovery ",
                "parallelism without nested oversubscription. Promote a native sparse ",
                "accelerator path only after native SpMV, full Krylov iteration, useful ",
                "preconditioning, and operation-level evidence pass. Add whole-study remote ",
                "and independent load-case execution through the shared analysis workload ",
                "ABI.",
            ),
            EvidenceKind::ExecutionEquivalence,
            DOMAIN,
        ),
        obligation(
            "release_supported_formulations",
            concat!(
                "Add family release evidence and documentation. Each promoted structural ",
                "formulation satisfies its evidence inventory and production definition. ",
                "Unsupported mixed combinations fail explicitly.",
            ),
            EvidenceKind::ExecutionEquivalence,
            DOMAIN,
        ),
        obligation(
            "connector_excitation_equivalence",
            concat!(
                "Structural constraint and connector verification inventory passes, ",
                "including reference-point resultants and rotated frames. Explicit and ",
                "generated static excitations have equivalent realized resultants, moments, ",
                "reactions, and energy. Serial and parallel results satisfy the declared ",
                "equivalence policy.",
            ),
            EvidenceKind::ExecutionEquivalence,
            DOMAIN,
        ),
        obligation(
            "parallel_gpu_equivalence",
            concat!(
                "Claimed GPU runs satisfy the promoted sparse-solver envelope without hidden ",
                "host SpMV or preconditioner execution.",
            ),
            EvidenceKind::ExecutionEquivalence,
            DOMAIN,
        ),
    ],
    dependencies: &[concat!(
        "Shared typed authoring, canonical topology/material/constraint/excitation ",
        "authorities; governed resources and execution; independently qualified ",
        "accuracy and semantic publication.",
    )],
};
pub(crate) const REQUIREMENTS: &[Requirement] = &[
    FEA_STRUCTURAL_SOLID_001,
    FEA_STRUCTURAL_BEAM_001,
    FEA_STRUCTURAL_SHELL_001,
    FEA_STRUCTURAL_REFERENCE_001,
    FEA_STRUCTURAL_CONSTRAINT_001,
    FEA_STRUCTURAL_EXECUTION_001,
];
