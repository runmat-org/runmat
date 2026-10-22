//! Retained harmonic response at one frequency (peak phasors, exp(+i omega t)).
use super::{complex::ComplexVector3, ComplexPhasor, MaxwellDriveResponse};
use runmat_execution::Digest;
use runmat_meshing_core::StableDigest;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaxwellFieldSample {
    pub sample_identity: StableDigest,
    pub element_identity: StableDigest,
    pub ordinal: usize,
    pub position_m: [f64; 3],
    pub volume_weight_m3: f64,
    pub electric_field_v_per_m: ComplexVector3,
    /// Faraday exp(+iωt): B=i curl E/ω.
    pub flux_density_t: ComplexVector3,
    pub magnetic_field_a_per_m: ComplexVector3,
    pub electric_displacement_c_per_m2: ComplexVector3,
    pub conduction_current_density_a_per_m2: ComplexVector3,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaxwellResponse {
    pub input_identity: Digest,
    pub system_plan_identity: Digest,
    pub mesh_digest: StableDigest,
    pub projection_digest: Digest,
    pub sampling_digest: Digest,
    pub angular_frequency_rad_per_s: f64,
    pub edge_node_identities: Vec<[StableDigest; 2]>,
    /// Complex oriented line integrals of E, in V; PEC edges are exactly zero.
    pub edge_circulations_v: Vec<ComplexPhasor>,
    pub samples: Vec<MaxwellFieldSample>,
    /// Terminal voltage, current and power of every driven conductor.
    pub drives: Vec<MaxwellDriveResponse>,
    /// Peak-phasor average energies: (1/4)∫ε|E|² and (1/4)∫μ|H|².
    pub electric_energy_j: f64,
    pub magnetic_energy_j: f64,
    /// Complex power delivered by the sources: terminal drives (1/2) U conj(I) plus
    /// -(1/2)∫E·conj(J) of impressed volume and gap-feed currents.
    pub source_power_va: ComplexPhasor,
    /// Cycle-averaged Joule loss (1/2)∫sigma|E|^2 with the total field E.
    pub ohmic_power_w: f64,
    /// 2ω(Wmagnetic−Welectric), positive inductive for exp(+iωt).
    pub reactive_power_var: f64,
    /// Original unpreconditioned free-edge complex physical weak equation.
    pub relative_physical_residual: f64,
    pub relative_complex_power_error: f64,
    /// Interior corner-gradient tests of div[(σ+iωε)E+J]=0; no solenoidal projection.
    pub relative_charge_continuity_residual: f64,
}

