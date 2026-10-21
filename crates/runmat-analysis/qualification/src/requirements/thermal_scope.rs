//! Thermal end state: published thermal studies.
use crate::requirement::*;
const THERMAL: &[Envelope] = cells!([Formulation::ThermalSteady, Formulation::ThermalTransient], [MeshOrder::Tet4, MeshOrder::Tet10], [ExecutionEnvelope::NumericalKernel, ExecutionEnvelope::PublicStudy]);
const STEADY: &[Envelope] = cells!([Formulation::ThermalSteady], [MeshOrder::Tet4, MeshOrder::Tet10], [ExecutionEnvelope::NumericalKernel, ExecutionEnvelope::PublicStudy]);
const TRANSIENT: &[Envelope] = cells!([Formulation::ThermalTransient], [MeshOrder::Tet4, MeshOrder::Tet10], [ExecutionEnvelope::NumericalKernel, ExecutionEnvelope::PublicStudy]);
const FEA_THERMAL_PUBLIC_001: Requirement = Requirement {
    id: "FEA-THERMAL-PUBLIC-001",
    title: "Thermal studies the pages run",
    stages: &[Stage::Thermal, Stage::PublicSurface],
    acceptance: "The thermal studies the page runs work as published: a film coefficient input swept for cooling, radiation to the surroundings, the heat sink at its balance, and power-cycle transients, each checked against a closed form or a balance.",
    owners: &["crates/runmat-analysis/fea/src/physics/thermal/assembly.rs"],
    dependencies: &[],
    obligations: &[
        obligation(
            "film_coefficient_input",
            "A convection film coefficient is a typed input quantity (`heat_transfer_coefficient`), usable as a sweep parameter and on named faces with ambient temperature, and heat flux into and out of the part balances.",
            EvidenceKind::PhysicalBalance,
            STEADY,
        ),
        obligation(
            "radiation_to_ambient",
            "Gray surface-to-ambient radiation (emissivity) with convection on the same faces runs and closes its energy balance; a radiating plate matches sigma epsilon A (T^4 - T_inf^4).",
            EvidenceKind::IndependentReference,
            THERMAL,
        ),
        obligation(
            "heat_sink_balance",
            "The KiCad heat sink (aluminum, about 10 W on the mounting face, 35 W/m2/K convection to 25 C) solves and heater power in equals convection out within the accuracy target, with the peak temperature rise reported.",
            EvidenceKind::PhysicalBalance,
            STEADY,
        ),
        obligation(
            "cooling_requirement_sweep",
            "A sweep of film coefficient from still air to a fan evaluates peak temperature against 85 C per case and reports the minimum cooling that passes.",
            EvidenceKind::PublicRun,
            STEADY,
        ),
        obligation(
            "power_cycle_transient",
            "A transient power cycle runs and matches the lumped-capacitance reference for a high-conductivity body and the Fourier series for a slab.",
            EvidenceKind::IndependentReference,
            TRANSIENT,
        ),
    ],
};
pub(crate) const REQUIREMENTS: &[Requirement] = &[
    FEA_THERMAL_PUBLIC_001,
];
