//! Acoustics end state (REG-ACOUS-001 to 009 and the acoustics page studies).
use crate::requirement::*;
const MODAL: &[Envelope] = cells!(
    [Formulation::AcousticModal],
    [MeshOrder::Tet4, MeshOrder::Tet10],
    [
        ExecutionEnvelope::NumericalKernel,
        ExecutionEnvelope::PublicStudy
    ]
);
const TRANS: &[Envelope] = cells!(
    [Formulation::AcousticTransient],
    [MeshOrder::Tet4, MeshOrder::Tet10],
    [
        ExecutionEnvelope::NumericalKernel,
        ExecutionEnvelope::PublicStudy
    ]
);
const HARM: &[Envelope] = cells!(
    [Formulation::AcousticHarmonic],
    [MeshOrder::Tet4, MeshOrder::Tet10],
    [
        ExecutionEnvelope::NumericalKernel,
        ExecutionEnvelope::PublicStudy
    ]
);
const HARMTRANS: &[Envelope] = cells!(
    [
        Formulation::AcousticHarmonic,
        Formulation::AcousticTransient
    ],
    [MeshOrder::Tet4, MeshOrder::Tet10],
    [
        ExecutionEnvelope::NumericalKernel,
        ExecutionEnvelope::PublicStudy
    ]
);
const COUPLED: &[Envelope] = cells!(
    [Formulation::AcousticStructureCoupled],
    [MeshOrder::Tet4, MeshOrder::Tet10],
    [
        ExecutionEnvelope::NumericalKernel,
        ExecutionEnvelope::PublicStudy
    ]
);
const FEA_ACOUSTIC_MODAL_001: Requirement = Requirement {
    id: "FEA-ACOUSTIC-MODAL-001",
    title: "Acoustic cavity modal analysis",
    stages: &[Stage::Acoustics],
    acceptance: "Natural frequencies and pressure mode shapes of an enclosed air volume with rigid and impedance walls publish from an `acoustic modal` study through `runmat run` with eigen-residuals, mass normalization and resource admission that accepts real models.",
    owners: &["crates/runmat-analysis/fea/src/physics/acoustic/canonical.rs"],
    dependencies: &[],
    obligations: &[
        obligation(
            "rigid_box_modes",
            "Rigid box modes equal c/2 sqrt((l/Lx)^2 + (m/Ly)^2 + (n/Lz)^2); the first 10 modes agree within 0.5 percent on Tet10.",
            EvidenceKind::IndependentReference,
            MODAL,
        ),
        obligation(
            "impedance_walls_and_damping_in_modes",
            "Modal studies with impedance walls (headliner 1200 Pa s/m, rigid glazing) return complex or damped modes consistent with the harmonic response peak positions.",
            EvidenceKind::IndependentReference,
            MODAL,
        ),
        obligation(
            "modal_publication",
            "Modal runs publish frequency, pressure mode shape fields (one view per mode), residuals and normalization, pass resource admission on a cabin-sized model (3 cubic metre air volume, 12 modes), and the cabin study's first mode is reported.",
            EvidenceKind::PublicRun,
            MODAL,
        ),
    ],
};
const FEA_ACOUSTIC_TRANSIENT_001: Requirement = Requirement {
    id: "FEA-ACOUSTIC-TRANSIENT-001",
    title: "Acoustic transient analysis",
    stages: &[Stage::Acoustics],
    acceptance: "Time-domain pressure from a time-history source publishes from an `acoustic transient` study with an energy balance.",
    owners: &["crates/runmat-analysis/fea/src/physics/acoustic/canonical.rs"],
    dependencies: &[],
    obligations: &[
        obligation(
            "duct_pulse",
            "A one-dimensional pulse in a rigid duct matches d'Alembert travel time and amplitude to the declared tolerance.",
            EvidenceKind::IndependentReference,
            TRANS,
        ),
        obligation(
            "pressure_time_history_publication",
            "An `acoustic transient` run publishes the complete ordered time history (time axis and one pressure snapshot per step including the initial state) as retained fields that are finite, bounded by the authored end time and readable cold, with the prescribed boundary carrying the authored pulse exactly.",
            EvidenceKind::PublicationIntegrity,
            TRANS,
        ),
        obligation(
            "energy_balance",
            "The transient energy balance (stored plus radiated plus dissipated minus supplied) closes to the declared tolerance, and time-step convergence is demonstrated.",
            EvidenceKind::PhysicalBalance,
            TRANS,
        ),
    ],
};
const FEA_ACOUSTIC_OUTPUTS_001: Requirement = Requirement {
    id: "FEA-ACOUSTIC-OUTPUTS-001",
    title: "Sound pressure level, transmission loss, intensity and power",
    stages: &[Stage::Acoustics],
    acceptance: "Acoustic studies publish the engineering outputs the pages quote: sound pressure level as an output and field, transmission and insertion loss of a duct element over a frequency sweep, active intensity and sound power through named surfaces, per frequency and per one-third-octave band.",
    owners: &["crates/runmat-analysis/fea/src/physics/acoustic/canonical.rs"],
    dependencies: &[],
    obligations: &[
        obligation(
            "sound_pressure_level",
            "The `level` output and an SPL field (dB re 20 micropascal, rms, optional A-weighting) publish per frequency from a harmonic study; a plane wave of 1 Pa amplitude reads 91.0 dB; point outputs interpolate the field at an authored position.",
            EvidenceKind::IndependentReference,
            HARM,
        ),
        obligation(
            "transmission_and_insertion_loss",
            "`transmission_loss` from incident and transmitted power (two-port) over a one-third-octave sweep 50 Hz to 2 kHz reproduces the simple expansion chamber TL = 10 log10(1 + 1/4 (m - 1/m)^2 sin^2(kL)) up to the first cross mode; the muffler study reports peak TL, its band, the lowest TL above the first peak and the bands below 15 dB.",
            EvidenceKind::IndependentReference,
            HARM,
        ),
        obligation(
            "intensity_and_sound_power",
            "The active intensity field and `sound_power` through a named surface publish; power through a sphere around a monopole equals rho c k^2 Q^2 / (8 pi) within 1 percent.",
            EvidenceKind::IndependentReference,
            HARM,
        ),
        obligation(
            "frequency_set_generation",
            "A study generates one-third-octave band centers (and other authored frequency sets), runs each frequency, and publishes outputs in authored order with per-frequency status.",
            EvidenceKind::PublicRun,
            HARM,
        ),
        obligation(
            "output_arithmetic_across_analyses",
            "Outputs of two analyses in one study (treated and bare room) combine into a difference output such as the SPL reduction at the operator position.",
            EvidenceKind::SemanticRealization,
            HARM,
        ),
    ],
};
const FEA_ACOUSTIC_BOUNDARY_002: Requirement = Requirement {
    id: "FEA-ACOUSTIC-BOUNDARY-002",
    title: "Radiation, frequency-dependent impedance and air damping",
    stages: &[Stage::Acoustics],
    acceptance: "Acoustic boundaries and media beyond constant real impedance run: spherical radiation on an outer sphere or hemisphere, complex and frequency-dependent impedance from a table or a porous-layer model, region damping, and periodic cells for band gaps.",
    owners: &["crates/runmat-analysis/fea/src/physics/acoustic/canonical.rs"],
    dependencies: &[],
    obligations: &[
        obligation(
            "spherical_radiation",
            "A spherical radiation boundary on an outer sphere or hemisphere converges toward the exact impedance: a pulsating sphere's surface pressure is within 2 percent of rho c u i k a / (1 + i k a) at ka = 1.",
            EvidenceKind::IndependentReference,
            HARM,
        ),
        obligation(
            "complex_frequency_dependent_impedance",
            "An impedance boundary takes a complex value and a frequency table, and a `delany_bazley` porous layer (flow resistivity, thickness); normal-incidence absorption of a 50 mm layer matches the Delany-Bazley closed form; the room study's treated and bare ceilings run.",
            EvidenceKind::IndependentReference,
            HARM,
        ),
        obligation(
            "region_damping",
            "Loss factor or complex sound speed per region is accepted at assembly (now rejected) and a damped duct resonance peak width matches the closed form.",
            EvidenceKind::IndependentReference,
            HARMTRANS,
        ),
        obligation(
            "bloch_band_gap",
            "A periodic unit cell with Bloch boundaries yields the band gap of a rod array, compared against an independent plane-wave expansion.",
            EvidenceKind::IndependentReference,
            HARM,
        ),
    ],
};
const FEA_ACOUSTIC_COUPLING_001: Requirement = Requirement {
    id: "FEA-ACOUSTIC-COUPLING-001",
    title: "One-way structure-borne sound",
    stages: &[Stage::Acoustics, Stage::FluidStructureInteraction],
    acceptance: "The normal velocity of a structural harmonic or modal result drives an acoustic study through a retained state, with conservative surface transfer.",
    owners: &["crates/runmat-analysis/fea/src/physics/acoustic/canonical.rs"],
    dependencies: &[],
    obligations: &[
        obligation(
            "structural_velocity_drives_air",
            "An acoustic step reads a retained structural velocity state on a wetted surface (`normal_velocity on face = analysis.s.state.velocity` with `retain velocity`); a baffled piston matches the Rayleigh integral on axis, and the speaker study (cone harmonic, then 200 Hz, 1 kHz and 4 kHz radiation) reports on-axis SPL and directivity.",
            EvidenceKind::IndependentReference,
            COUPLED,
        ),
        obligation(
            "surface_transfer_conservation",
            "Volume velocity is conserved through the transfer between non-matching structural and acoustic surface meshes.",
            EvidenceKind::PhysicalBalance,
            COUPLED,
        ),
    ],
};
pub(crate) const REQUIREMENTS: &[Requirement] = &[
    FEA_ACOUSTIC_MODAL_001,
    FEA_ACOUSTIC_TRANSIENT_001,
    FEA_ACOUSTIC_OUTPUTS_001,
    FEA_ACOUSTIC_BOUNDARY_002,
    FEA_ACOUSTIC_COUPLING_001,
];
