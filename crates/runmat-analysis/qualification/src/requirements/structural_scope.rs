//! Structural end state: capabilities the page claims.
use crate::requirement::*;
const STRUCT: &[Envelope] = cells!([Formulation::LinearStructural, Formulation::ModalStructural, Formulation::LinearStructuralDynamics, Formulation::StructuralStability], [MeshOrder::Tet4, MeshOrder::Tet10], [ExecutionEnvelope::NumericalKernel, ExecutionEnvelope::PublicStudy]);
const SHELLBEAM: &[Envelope] = cells!([Formulation::LinearStructural], [MeshOrder::Beam, MeshOrder::Shell], [ExecutionEnvelope::NumericalKernel, ExecutionEnvelope::PublicStudy]);
const NONLIN: &[Envelope] = cells!([Formulation::NonlinearStructural, Formulation::SmallStrainJ2], [MeshOrder::Tet4, MeshOrder::Tet10], [ExecutionEnvelope::NumericalKernel, ExecutionEnvelope::PublicStudy]);
const CONTACT: &[Envelope] = cells!([Formulation::PenaltyCoulombContact, Formulation::SmallStrainJ2PenaltyCoulombContact], [MeshOrder::Tet4, MeshOrder::Tet10], [ExecutionEnvelope::NumericalKernel, ExecutionEnvelope::PublicStudy]);
const LINEAR: &[Envelope] = cells!([Formulation::LinearStructural, Formulation::ModalStructural], [MeshOrder::Tet4, MeshOrder::Tet10], [ExecutionEnvelope::NumericalKernel, ExecutionEnvelope::PublicStudy]);
const DYN: &[Envelope] = cells!([Formulation::LinearStructuralDynamics, Formulation::ModalStructural], [MeshOrder::Tet4, MeshOrder::Tet10], [ExecutionEnvelope::NumericalKernel, ExecutionEnvelope::PublicStudy]);
const LOADS: &[Envelope] = cells!([Formulation::LinearStructural, Formulation::LinearStructuralDynamics, Formulation::NonlinearStructural], [MeshOrder::Tet4, MeshOrder::Tet10], [ExecutionEnvelope::NumericalKernel, ExecutionEnvelope::PublicStudy]);
const FEA_STRUCTURAL_PUBLIC_001: Requirement = Requirement {
    id: "FEA-STRUCTURAL-PUBLIC-001",
    title: "Structural capabilities the page claims",
    stages: &[Stage::LinearStructures, Stage::PublicSurface],
    acceptance: "Every structural capability the page states runs from a `.fea` study on imported geometry with its result checked against a reference: shells and beams, the full load and connection inventory, nonlinear analyses, and the references the page quotes.",
    owners: &["crates/runmat-analysis/fea/src/physics/structural/contact/mod.rs"],
    dependencies: &[],
    obligations: &[
        obligation(
            "shells_and_beams_run",
            "A shell panel (MITC3, homogeneous and laminate layups with orthotropic plies) and a beam frame (general, rectangular, circular and tube sections) run from a `.fea` study on imported geometry and publish results; a plate matches the plate-bending formula, a frame matches beam theory, and a frame with code limits written as assessments returns verdicts.",
            EvidenceKind::IndependentReference,
            SHELLBEAM,
        ),
        obligation(
            "load_inventory_runs",
            "Every structural load the docs list runs and conserves its resultant: total, distributed and remote-point forces, moments and torque, traction, pressure (wind, mapped from a fluid result, hydrostatic), gravity and body force, inertial and centrifugal acceleration, base acceleration and earthquake motion, bolt preload, bearing loads on holes, thermal expansion, line loads, harmonic amplitudes, load cases and combinations, and local coordinate systems.",
            EvidenceKind::PhysicalBalance,
            LOADS,
        ),
        obligation(
            "connection_inventory_runs",
            "Fixed supports, prescribed motion, couplings, ties, periodic constraints, equations, axial springs and dashpots, point springs to ground, elastic foundations, six-DOF bushings, rotational springs, nonlinear springs and gaps, frequency-dependent mounts and bonded interfaces run with reactions that balance the loads, and no weak spring, mass, damping, reference constraint or load case is added silently.",
            EvidenceKind::PhysicalBalance,
            LOADS,
        ),
        obligation(
            "large_deformation",
            "A `structural nonlinear` cantilever under a tip load to large rotation matches the elastica solution with Newton and load stepping.",
            EvidenceKind::IndependentReference,
            NONLIN,
        ),
        obligation(
            "plasticity_example",
            "A published J2 plasticity example (linear isotropic hardening, follower loads) runs and matches the manufactured or analytic reference including unloading.",
            EvidenceKind::IndependentReference,
            NONLIN,
        ),
        obligation(
            "contact_example",
            "A published frictionless and Coulomb contact example (bolted bracket) runs from the public command and matches a Hertz or sliding reference.",
            EvidenceKind::IndependentReference,
            CONTACT,
        ),
        obligation(
            "hand_calculation_references",
            "Cantilever tip deflection matches PL^3/(3EI) (0.199 against 0.200 mm), the first bending frequency matches Euler-Bernoulli (81.3 against 81.5 Hz), and a random-vibration study matches Miles' equation, with the coarse-mesh failure (0.1197 mm Tet4) flagged as cannot-tell rather than accepted.",
            EvidenceKind::IndependentReference,
            LINEAR,
        ),
        obligation(
            "result_publication_defects",
            "Explicit spatial observations without a canonical solver mesh, duplicate recovered semantic samples and missing buckling mode-shape fields are fixed: observations resolve, samples are unique, and a buckling run publishes its mode shape field.",
            EvidenceKind::PublicationIntegrity,
            STRUCT,
        ),
        obligation(
            "vibration_qualification",
            "A random-response study driven by a PSD (GEVS-type) from modal data reports RMS and 3-sigma stress, matching an independent modal combination.",
            EvidenceKind::IndependentReference,
            DYN,
        ),
    ],
};
pub(crate) const REQUIREMENTS: &[Requirement] = &[
    FEA_STRUCTURAL_PUBLIC_001,
];
