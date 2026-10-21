use super::{assembly::MagneticAssembly, MagnetostaticError};
use crate::algebra::{dot_canonical, spmv};
use runmat_meshing_core::StableDigest;

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MagneticFieldSample {
    pub element_identity: StableDigest,
    pub position_m: [f64; 3],
    pub volume_weight_m3: f64,
    pub vector_potential_wb_per_m: [f64; 3],
    pub flux_density_t: [f64; 3],
    pub magnetic_field_a_per_m: [f64; 3],
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MagnetostaticResponse {
    pub edge_node_identities: Vec<[StableDigest; 2]>,
    pub edge_circulations_wb: Vec<f64>,
    pub samples: Vec<MagneticFieldSample>,
    pub magnetic_energy_j: f64,
    /// ∫J·A dV; equals twice magnetic energy for this homogeneous ground policy.
    pub source_work_j: f64,
    pub relative_physical_residual: f64,
    pub relative_gauge_residual: f64,
    pub relative_energy_work_error: f64,
    pub iterations: u32,
}

pub(super) fn solve(
    assembled: MagneticAssembly<'_>,
    tolerance: f64,
) -> Result<MagnetostaticResponse, MagnetostaticError> {
    let n = assembled.saddle.rows();
    // The curl-curl/gauge saddle is indefinite with a material-contrast-dependent spectrum
    // (iron next to air); a sparse LU solves it exactly instead of an unpreconditioned Krylov.
    let (solution, iterations) = if n == 0 {
        (Vec::new(), 0)
    } else {
        let pattern = assembled.saddle.pattern();
        let offsets = pattern.row_offsets();
        let columns = pattern.column_indices();
        let values = assembled.saddle.values();
        let mut entries = Vec::with_capacity(values.len());
        for row in 0..n {
            for k in offsets[row] as usize..offsets[row + 1] as usize {
                entries.push((row, columns[k] as usize, values[k]));
            }
        }
        let solved = crate::solve::sparse_direct::solve_real(n, &entries, &assembled.scaled_rhs)
            .map_err(|e| MagnetostaticError::NotConverged(e.to_string()))?;
        if !solved.relative_residual.is_finite() || solved.relative_residual > tolerance {
            return Err(MagnetostaticError::NotConverged(format!(
                "sparse direct residual={:e}",
                solved.relative_residual
            )));
        }
        (solved.solution, 1)
    };
    let free_count = assembled.boundary.free_edges.len();
    let free = &solution[..free_count];
    let action = spmv(&assembled.physical_stiffness, free)
        .map_err(|e| MagnetostaticError::Operator(e.into()))?;
    let defect: Vec<_> = action
        .iter()
        .zip(&assembled.physical_load)
        .map(|(a, b)| a - b)
        .collect();
    let norm = |x: &[f64]| dot_canonical(x, x).sqrt();
    let load_norm = norm(&assembled.physical_load);
    let physical_residual = if load_norm > 0.0 {
        norm(&defect) / load_norm
    } else {
        norm(&defect)
    };
    let saddle_action =
        spmv(&assembled.saddle, &solution).map_err(|e| MagnetostaticError::Operator(e.into()))?;
    let rhs_norm = norm(&assembled.scaled_rhs);
    let gauge_residual = if rhs_norm > 0.0 {
        norm(&saddle_action[free_count..]) / rhs_norm
    } else {
        norm(&saddle_action[free_count..])
    };
    if !physical_residual.is_finite()
        || !gauge_residual.is_finite()
        || physical_residual > tolerance * 8.0
        || gauge_residual > tolerance * 8.0
    {
        return Err(MagnetostaticError::NotConverged(format!(
            "physical={physical_residual:e}, gauge={gauge_residual:e}"
        )));
    }
    let mut edge_circulations = vec![0.0; assembled.edge_space.edges().len()];
    for (&edge, &value) in assembled.boundary.free_edges.iter().zip(free) {
        edge_circulations[edge] = value;
    }
    let count = assembled
        .projection
        .elements
        .len()
        .checked_mul(15)
        .ok_or_else(|| MagnetostaticError::InvalidInput("magnetic sample count overflow".into()))?;
    let mut samples = Vec::with_capacity(count);
    let mut energy = 0.0;
    let mut work = 0.0;
    for (index, data) in assembled.volumes.iter().enumerate() {
        let indices = assembled.edge_space.element_edge_indices(index).unwrap();
        assembled
            .edge_space
            .visit_element_quadrature(assembled.projection, index, |point| {
                let potential: [f64; 3] = std::array::from_fn(|axis| {
                    (0..6)
                        .map(|i| edge_circulations[indices[i]] * point.values[i][axis])
                        .sum()
                });
                let flux: [f64; 3] = std::array::from_fn(|axis| {
                    (0..6)
                        .map(|i| edge_circulations[indices[i]] * point.curls[i][axis])
                        .sum()
                });
                let field = flux.map(|v| v / data.permeability_h_per_m);
                energy += 0.5 * point.volume_weight_m3 * dot_canonical(&flux, &field);
                work += point.volume_weight_m3
                    * dot_canonical(&potential, &data.current_density_a_per_m2);
                samples.push(MagneticFieldSample {
                    element_identity: data.element_identity,
                    position_m: point.position_m,
                    volume_weight_m3: point.volume_weight_m3,
                    vector_potential_wb_per_m: potential,
                    flux_density_t: flux,
                    magnetic_field_a_per_m: field,
                });
            })
            .map_err(|e| MagnetostaticError::Topology(e.to_string()))?;
    }
    let scale = (2.0 * energy).abs().max(work.abs());
    let closure = if scale > 0.0 {
        (2.0 * energy - work).abs() / scale
    } else {
        0.0
    };
    if !energy.is_finite() || !work.is_finite() || closure > tolerance * 16.0 {
        return Err(MagnetostaticError::NotConverged(format!(
            "energy/work closure={closure:e}"
        )));
    }
    // The solver-equation scaling is not a change of physical field units.
    debug_assert!(assembled.permeability_scale > 0.0);
    Ok(MagnetostaticResponse {
        edge_node_identities: assembled
            .edge_space
            .edges()
            .iter()
            .map(|edge| edge.node_identities)
            .collect(),
        edge_circulations_wb: edge_circulations,
        samples,
        magnetic_energy_j: energy,
        source_work_j: work,
        relative_physical_residual: physical_residual,
        relative_gauge_residual: gauge_residual,
        relative_energy_work_error: closure,
        iterations,
    })
}
