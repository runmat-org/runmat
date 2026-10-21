//! Chained studies and public adaptive analysis.
use crate::requirement::*;
const CHAIN: &[Envelope] = cells!([Formulation::Thermoelastic, Formulation::StructuralStability, Formulation::ModalStructural], [MeshOrder::Tet4, MeshOrder::Tet10], [ExecutionEnvelope::NumericalKernel, ExecutionEnvelope::PublicStudy]);
const ADAPT: &[Envelope] = cells!([Formulation::AdaptiveAnalysis], [MeshOrder::Tet4, MeshOrder::Tet10], [ExecutionEnvelope::PublicStudy, ExecutionEnvelope::NumericalKernel]);
const FEA_COUPLING_CHAIN_001: Requirement = Requirement {
    id: "FEA-COUPLING-CHAIN-001",
    title: "Chained thermal-structural studies",
    stages: &[Stage::Thermal, Stage::Dynamics, Stage::PublicSurface],
    acceptance: "Thermal studies chain into preload, prestressed modal, harmonic and buckling studies from one `.fea` file, with the thermomechanical references the page quotes.",
    owners: &["crates/runmat-analysis/fea/src/physics/coupling/thermo_mechanical.rs"],
    dependencies: &[],
    obligations: &[
        obligation(
            "thermal_to_prestress_to_modal",
            "A study runs thermal, then preload, then buckling or prestressed modal in sequence on one mesh without the mixed-sequence rejection, with the retained states typed and checked.",
            EvidenceKind::PublicRun,
            CHAIN,
        ),
        obligation(
            "thermomechanical_references",
            "A uniformly heated constrained bar (80 K rise) gives axial stress -E alpha deltaT (-192 MPa for the page's bar); the first bending frequency of a bar heated 200 K over unheated is 0.963 within the declared tolerance.",
            EvidenceKind::IndependentReference,
            CHAIN,
        ),
    ],
};
const FEA_ADAPTATION_PUBLIC_001: Requirement = Requirement {
    id: "FEA-ADAPTATION-PUBLIC-001",
    title: "Adaptive analysis from the language and CLI",
    stages: &[Stage::Adaptation, Stage::PublicSurface],
    acceptance: "Estimator-driven adaptive analysis is authored in the study language and runs from the public command, with the accuracy target and refinement history published.",
    owners: &["crates/runmat-analysis/fea/src/results/accuracy/tests.rs"],
    dependencies: &[],
    obligations: &[
        obligation(
            "adaptive_study_authoring",
            "An adaptive analysis is declared in a `.fea` study with its target quantity and tolerance, admitted by check, run from the public command, and publishes the estimator, refinement history and final verdict.",
            EvidenceKind::PublicRun,
            ADAPT,
        ),
    ],
};
pub(crate) const REQUIREMENTS: &[Requirement] = &[
    FEA_COUPLING_CHAIN_001,
    FEA_ADAPTATION_PUBLIC_001,
];
