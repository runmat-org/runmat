//! Electromagnetics end state (REG-EM-001 to 003, harmonic sweeps, outputs and electrothermal).
use crate::requirement::*;
const TRANS: &[Envelope] = cells!(
    [Formulation::ElectromagneticTransient],
    [MeshOrder::EdgeElements],
    [
        ExecutionEnvelope::NumericalKernel,
        ExecutionEnvelope::PublicStudy
    ]
);
const NONLIN: &[Envelope] = cells!(
    [Formulation::SaturatingMagnetics],
    [MeshOrder::EdgeElements],
    [
        ExecutionEnvelope::NumericalKernel,
        ExecutionEnvelope::PublicStudy
    ]
);
const INPUTS_SCALAR: &[Envelope] = cells!(
    [
        Formulation::Electrostatic,
        Formulation::ElectricalConduction
    ],
    [MeshOrder::Tet4, MeshOrder::Tet10],
    [
        ExecutionEnvelope::NumericalKernel,
        ExecutionEnvelope::PublicStudy
    ]
);
const INPUTS_MAG: &[Envelope] = cells!(
    [Formulation::Magnetostatic, Formulation::HarmonicMaxwell],
    [MeshOrder::EdgeElements],
    [
        ExecutionEnvelope::NumericalKernel,
        ExecutionEnvelope::PublicStudy
    ]
);
const FREQ: &[Envelope] = cells!(
    [Formulation::HarmonicMaxwell],
    [MeshOrder::EdgeElements],
    [
        ExecutionEnvelope::NumericalKernel,
        ExecutionEnvelope::PublicStudy
    ]
);
const OUT: &[Envelope] = cells!(
    [
        Formulation::Electrostatic,
        Formulation::ElectricalConduction,
        Formulation::Magnetostatic,
        Formulation::HarmonicMaxwell
    ],
    [MeshOrder::Tet4, MeshOrder::Tet10, MeshOrder::EdgeElements],
    [ExecutionEnvelope::PublicStudy]
);
const ETH: &[Envelope] = cells!(
    [Formulation::Electrothermal],
    [MeshOrder::Tet4, MeshOrder::Tet10, MeshOrder::EdgeElements],
    [
        ExecutionEnvelope::NumericalKernel,
        ExecutionEnvelope::PublicStudy
    ]
);
const FEA_EM_TRANSIENT_001: Requirement = Requirement {
    id: "FEA-EM-TRANSIENT-001",
    title: "Transient electromagnetics",
    stages: &[Stage::Electromagnetics],
    acceptance: "Time-domain eddy currents (A-V formulation) run from time-history drives, including switching and pulse loads, with the loss passed to heat, from an `electromagnetic transient` study through `runmat run`.",
    owners: &["crates/runmat-analysis/fea/src/physics/maxwell/system.rs"],
    dependencies: &[],
    obligations: &[
        obligation(
            "step_current_diffusion",
            "A step current into a conducting slab diffuses with the analytic error-function series (field and current density) within the declared tolerance, converging in time step and mesh.",
            EvidenceKind::IndependentReference,
            TRANS,
        ),
        obligation(
            "switching_and_pulse_loads",
            "Pulsed and switched voltage or current drives from authored time signals run and conserve energy: supplied equals stored magnetic energy plus Joule loss plus work.",
            EvidenceKind::PhysicalBalance,
            TRANS,
        ),
        obligation(
            "loss_to_heat",
            "The Joule loss history is available to a thermal transient as a conservative heat source without remeshing.",
            EvidenceKind::PhysicalBalance,
            TRANS,
        ),
    ],
};
const FEA_EM_NONLINEAR_001: Requirement = Requirement {
    id: "FEA-EM-NONLINEAR-001",
    title: "Saturating iron",
    stages: &[Stage::Electromagnetics],
    acceptance: "Nonlinear B-H curves (tabulated, monotone spline) run in magnetostatics, in harmonic analysis through an effective permeability, and in transient analysis, with Newton and line search.",
    owners: &["crates/runmat-analysis/fea/src/physics/maxwell/system.rs"],
    dependencies: &[],
    obligations: &[
        obligation(
            "bh_curve_inputs",
            "Tabulated B-H data is validated monotone and dimensioned, evaluated with a monotone spline, and extrapolated to the vacuum slope above the last point; non-monotone or non-physical data is rejected.",
            EvidenceKind::RejectionInvariant,
            NONLIN,
        ),
        obligation(
            "newton_with_line_search",
            "The nonlinear magnetic solve converges with Newton and line search, publishes iteration history, and fails typed when it stagnates.",
            EvidenceKind::IndependentReference,
            NONLIN,
        ),
        obligation(
            "saturating_toroid_reference",
            "A saturating toroid reproduces the analytic B(H) relation (H = N I / l) within the declared tolerance in magnetostatic, harmonic effective-permeability and transient forms, and a current-sensor core near saturation reports flux through iron and gap.",
            EvidenceKind::IndependentReference,
            NONLIN,
        ),
    ],
};
const FEA_EM_INPUTS_002: Requirement = Requirement {
    id: "FEA-EM-INPUTS-002",
    title: "Every declared electromagnetic input runs",
    stages: &[Stage::Electromagnetics],
    acceptance: "Every electromagnetic input the language declares runs and is documented: terminal_voltage, electric_current terminals, electric_charge, current_density, magnetic_vector_potential, magnetic.insulation, equipotential, floating, integral-current and periodic constraints.",
    owners: &["crates/runmat-analysis/fea/src/physics/maxwell/system.rs"],
    dependencies: &[],
    obligations: &[
        obligation(
            "scalar_potential_inputs",
            "terminal_voltage, electric current terminals, electric_charge, equipotential, floating and integral-current constraints run for electrostatics and steady conduction, with total current and charge balances closed.",
            EvidenceKind::PhysicalBalance,
            INPUTS_SCALAR,
        ),
        obligation(
            "magnetic_inputs",
            "current_density, magnetic_vector_potential, magnetic.insulation and periodic constraints run for magnetostatics and harmonic Maxwell with typed rank-correct constraints.",
            EvidenceKind::SemanticRealization,
            INPUTS_MAG,
        ),
    ],
};
const FEA_EM_FREQUENCY_001: Requirement = Requirement {
    id: "FEA-EM-FREQUENCY-001",
    title: "Multi-frequency harmonic electromagnetics and fields",
    stages: &[Stage::Electromagnetics],
    acceptance: "Harmonic Maxwell runs several frequencies in one study through the executor with skin effect resolved, AC loss computed, outer air boundaries that do not corrupt the interior field, and the fields and terminal quantities the pages quote.",
    owners: &["crates/runmat-analysis/fea/src/physics/maxwell/system.rs"],
    dependencies: &[],
    obligations: &[
        obligation(
            "multi_frequency_study",
            "A single study runs 3, 10 and 30 kHz (or any authored frequency set) as one analysis with per-frequency fields and status in authored order.",
            EvidenceKind::PublicRun,
            FREQ,
        ),
        obligation(
            "skin_effect_reference",
            "Conductor current density and AC resistance reproduce the skin-depth reference delta = sqrt(2 / (omega mu sigma)) for a round conductor or slab within the declared tolerance on a mesh that resolves delta.",
            EvidenceKind::IndependentReference,
            FREQ,
        ),
        obligation(
            "air_truncation",
            "Interior fields and coil loss change by less than the declared tolerance when the surrounding air domain is enlarged, so the outer boundary choice is documented as adequate.",
            EvidenceKind::SpatialConvergence,
            FREQ,
        ),
        obligation(
            "ac_loss_power",
            "Cycle-averaged loss in a body (the induction-hardening gear at 10 kHz) equals the integral of J dot E over the body and matches terminal power minus stored-energy exchange.",
            EvidenceKind::PhysicalBalance,
            FREQ,
        ),
    ],
};
const FEA_EM_OUTPUTS_001: Requirement = Requirement {
    id: "FEA-EM-OUTPUTS-001",
    title: "Electromagnetic outputs and fields",
    stages: &[Stage::Electromagnetics],
    acceptance: "Electromagnetic studies publish the derived values and fields the pages show: terminal voltage drop, resistance, current, loss power and field maxima, and current density, flux density and electric field with magnitudes (and phases for AC).",
    owners: &["crates/runmat-analysis/fea/src/physics/maxwell/system.rs"],
    dependencies: &[],
    obligations: &[
        obligation(
            "terminal_quantities",
            "Voltage drop between named faces, terminal current, resistance and loss power publish and agree with V = I R for the busbar and with energy balances; a high-voltage electrode reports peak field and an assessment against its limit, sharp versus rounded.",
            EvidenceKind::PhysicalBalance,
            OUT,
        ),
        obligation(
            "vector_field_publication",
            "Current density, magnetic flux density and electric field publish as element or node vector fields with magnitudes (and phasors for AC) in the form the viewer reads for cut planes, field lines, arrows and isosurfaces.",
            EvidenceKind::PublicationIntegrity,
            OUT,
        ),
    ],
};
const FEA_COUPLING_ELECTROTHERMAL_002: Requirement = Requirement {
    id: "FEA-COUPLING-ELECTROTHERMAL-002",
    title: "Two-way and transient Joule and AC-loss heating",
    stages: &[Stage::Electromagnetics, Stage::Thermal],
    acceptance: "Electrothermal studies run from a `.fea` study with temperature-dependent conductivity iterated to convergence, transient heating, and harmonic AC loss chained into a transient heat analysis.",
    owners: &["crates/runmat-analysis/fea/src/physics/coupling/electro_thermal.rs"],
    dependencies: &[],
    obligations: &[
        obligation(
            "two_way_conductivity_feedback",
            "`electrothermal steady` with sigma(T) = sigma0 / (1 + alpha (T - T0)) iterates to a converged fixed point; the busbar with alpha = 0.00393 per K reproduces power equal to V times I.",
            EvidenceKind::IndependentReference,
            ETH,
        ),
        obligation(
            "transient_electrothermal",
            "`electrothermal transient` runs with the same coupling and matches a lumped adiabatic heating reference over a short time and steady state over a long time.",
            EvidenceKind::TemporalConvergence,
            ETH,
        ),
        obligation(
            "ac_loss_into_transient_heat",
            "Cycle-averaged harmonic loss from the gear and inductor drives a 3 s transient heat step with the loss field mapped conservatively (total power preserved), reporting peak gear surface temperature.",
            EvidenceKind::PhysicalBalance,
            ETH,
        ),
        obligation(
            "public_route",
            "Electrothermal studies are admitted at `runmat run` and run remotely.",
            EvidenceKind::PublicRun,
            ETH,
        ),
    ],
};
pub(crate) const REQUIREMENTS: &[Requirement] = &[
    FEA_EM_TRANSIENT_001,
    FEA_EM_NONLINEAR_001,
    FEA_EM_INPUTS_002,
    FEA_EM_FREQUENCY_001,
    FEA_EM_OUTPUTS_001,
    FEA_COUPLING_ELECTROTHERMAL_002,
];
