//! Sparse direct LU over the complete real complex block; physical acceptance.
use super::{
    assembly::MaxwellAssembly, complex::ComplexVector3, operator, recovery, ComplexPhasor,
    MaxwellError,
};
use crate::algebra::{dot_canonical, spmv};
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
    /// Peak-phasor average energies: (1/4)∫ε|E|² and (1/4)∫μ|H|².
    pub electric_energy_j: f64,
    pub magnetic_energy_j: f64,
    /// Delivered source power: -(1/2)∫E·conj(J_impressed), not the opposite sign.
    pub source_power_va: ComplexPhasor,
    pub ohmic_power_w: f64,
    /// 2ω(Wmagnetic−Welectric), positive inductive for exp(+iωt).
    pub reactive_power_var: f64,
    /// Original unpreconditioned free-edge complex physical weak equation.
    pub relative_physical_residual: f64,
    pub relative_complex_power_error: f64,
    /// Interior corner-gradient tests of div[(σ+iωε)E+J]=0; no solenoidal projection.
    pub relative_charge_continuity_residual: f64,
    pub iterations: u32,
}

pub(super) fn solve(
    assembled: MaxwellAssembly<'_>,
    tolerance: f64,
) -> Result<MaxwellResponse, MaxwellError> {
    let n = assembled.block.rows();
    // The shared Krylov recurrence uses dot-based norms. A finite nonzero
    // forcing must not be mistaken for zero when its squared norm underflows.
    // No dimensional floor, rescaled source, or altered physical equation.
    let krylov_rhs_squared_norm = dot_canonical(&assembled.rhs, &assembled.rhs);
    if !krylov_rhs_squared_norm.is_finite()
        || (krylov_rhs_squared_norm == 0.0 && assembled.rhs.iter().any(|value| *value != 0.0))
    {
        return Err(MaxwellError::Unsupported(
            "physical forcing norm is outside the representable shared Krylov envelope".into(),
        ));
    }
    let (solution, iterations) = if n == 0 {
        (Vec::new(), 0)
    } else {
        // Complete real block of the complex electric-field system, factored by sparse LU.
        let (solution, _) = crate::solve::shared_sparse::solve_direct_real(
            &assembled.block,
            &assembled.rhs,
            crate::solve::shared_sparse::DirectStructure::General,
        )
        .map_err(|error| MaxwellError::NotConverged(format!("sparse direct LU: {error}")))?;
        (solution, 1)
    };
    let action = spmv(&assembled.block, &solution).map_err(operator)?;
    let defect: Vec<_> = action
        .iter()
        .zip(&assembled.rhs)
        .map(|(a, b)| a - b)
        .collect();
    // Independently measure the ORIGINAL equation with a stable norm: residual
    // squaring must not underflow and hide incomplete Krylov convergence.
    let norm = |x: &[f64]| x.iter().fold(0.0_f64, |sum, value| sum.hypot(*value));
    let load_norm = norm(&assembled.rhs);
    let residual = if load_norm > 0.0 {
        norm(&defect) / load_norm
    } else {
        norm(&defect)
    };
    if !residual.is_finite() || residual > tolerance * 8.0 {
        return Err(MaxwellError::NotConverged(format!(
            "physical complex residual={residual:e}"
        )));
    }
    let free = assembled.plan.free_edge_indices().len();
    let mut edges = vec![ComplexPhasor::ZERO; assembled.edge_space.edges().len()];
    for (row, &edge) in assembled.plan.free_edge_indices().iter().enumerate() {
        edges[edge] = ComplexPhasor::new(solution[row], solution[row + free]);
    }
    recovery::recover(&assembled, edges, residual, iterations, tolerance)
}
