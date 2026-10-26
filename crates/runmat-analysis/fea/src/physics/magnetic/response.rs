use super::{assembly::MagneticAssembly, MagnetostaticError};
use crate::algebra::{dot_canonical, spmv};
use crate::solve::krylov::restarted_gmres;
use runmat_meshing_core::StableDigest;
use runmat_types::physical::{QuantityField, QuantityLayout};

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MagneticFieldSample {
    pub element_identity: StableDigest,
    pub position_m: [f64; 3],
    pub volume_weight_m3: f64,
    #[serde(with = "runmat_types::physical::canonical_storage")]
    pub vector_potential_wb_per_m: runmat_types::physical::Quantity<
        runmat_types::physical::kind::MagneticVectorPotential,
        [f64; 3],
    >,
    #[serde(with = "runmat_types::physical::canonical_storage")]
    pub flux_density_t: runmat_types::physical::Quantity<
        runmat_types::physical::kind::MagneticFluxDensity,
        [f64; 3],
    >,
    #[serde(with = "runmat_types::physical::canonical_storage")]
    pub magnetic_field_a_per_m: runmat_types::physical::Quantity<
        runmat_types::physical::kind::MagneticFieldIntensity,
        [f64; 3],
    >,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MagnetostaticResponse {
    pub edge_node_identities: Vec<[StableDigest; 2]>,
    #[serde(with = "runmat_types::physical::canonical_storage")]
    pub edge_circulations_wb:
        runmat_types::physical::Quantity<runmat_types::physical::kind::MagneticFlux, Vec<f64>>,
    pub samples: Vec<MagneticFieldSample>,
    #[serde(with = "runmat_types::physical::canonical_storage")]
    pub magnetic_energy_j:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Energy, f64>,
    #[serde(with = "runmat_types::physical::canonical_storage")]
    /// ∫J·A dV; equals twice magnetic energy for this homogeneous ground policy.
    pub source_work_j: runmat_types::physical::Quantity<runmat_types::physical::kind::Energy, f64>,
    pub relative_physical_residual: f64,
    pub relative_gauge_residual: f64,
    pub relative_energy_work_error: f64,
    pub iterations: u32,
}

pub(super) fn solve(
    assembled: MagneticAssembly<'_>,
    tolerance: f64,
) -> Result<MagnetostaticResponse, MagnetostaticError> {
    let n = assembled.saddle.input_layout().component_count();
    let (solution, iterations) = if n == 0 {
        (Vec::new(), 0)
    } else {
        let result = assembled
            .saddle
            .with_canonical_storage(|saddle| {
                restarted_gmres(
                    assembled.scaled_rhs.canonical_values(),
                    &vec![0.0; n],
                    super::resources::gmres_policy(n, tolerance)?,
                    |x| spmv(saddle, x).map_err(|e| MagnetostaticError::Operator(e.into())),
                    |a, b| Ok(dot_canonical(a, b)),
                    || false,
                )
            })?
            .ok_or_else(|| MagnetostaticError::NotConverged("invalid Krylov contract".into()))?;
        if !result.converged {
            return Err(MagnetostaticError::NotConverged(format!(
                "GMRES iterations={}, residual={:e}",
                result.iterations, result.residual_norm
            )));
        }
        (result.solution, result.iterations)
    };
    let solution = QuantityField::try_new(assembled.saddle.input_layout().clone(), solution)
        .map_err(|e| {
            MagnetostaticError::Operator(format!("invalid magnetic solution coordinates: {e:?}"))
        })?;
    // Gauge coordinates are normalized Wb, while their actual physical
    // multiplier is A/m. Check reconstruction in the retained basis too.
    solution.with_canonical_storage(|values| -> Result<(), MagnetostaticError> {
        for value in &values[solution.layout().edge_count..] {
            solution
                .layout()
                .physical_gauge(runmat_types::physical::MagneticFlux::new(*value))
                .map_err(|e| MagnetostaticError::Operator(e.to_string()))?;
        }
        Ok(())
    })?;
    solution.with_canonical_storage(|solution| {
        let free_count = assembled.boundary.free_edges.len();
        let free = &solution[..free_count];
        let action = assembled
            .physical_stiffness
            .with_canonical_storage(|matrix| spmv(matrix, free))
            .map_err(|e| MagnetostaticError::Operator(e.into()))?;
        let defect: Vec<_> = action
            .iter()
            .zip(assembled.physical_load.canonical_values())
            .map(|(a, b)| a - b)
            .collect();
        let norm = |x: &[f64]| dot_canonical(x, x).sqrt();
        let load_norm = norm(assembled.physical_load.canonical_values());
        let physical_residual = if load_norm > 0.0 {
            norm(&defect) / load_norm
        } else {
            norm(&defect)
        };
        let saddle_action = assembled
            .saddle
            .with_canonical_storage(|matrix| spmv(matrix, solution))
            .map_err(|e| MagnetostaticError::Operator(e.into()))?;
        let rhs_norm = norm(assembled.scaled_rhs.canonical_values());
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
            .ok_or_else(|| {
                MagnetostaticError::InvalidInput("magnetic sample count overflow".into())
            })?;
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
                    let field = flux.map(|v| v / data.permeability_h_per_m.canonical_value());
                    energy += 0.5 * point.volume_weight_m3 * dot_canonical(&flux, &field);
                    work += point.volume_weight_m3
                        * dot_canonical(
                            &potential,
                            &data.current_density_a_per_m2.canonical_value(),
                        );
                    samples.push(MagneticFieldSample {
                        element_identity: data.element_identity,
                        position_m: point.position_m,
                        volume_weight_m3: point.volume_weight_m3,
                        vector_potential_wb_per_m: runmat_types::physical::Quantity::<
                            runmat_types::physical::kind::MagneticVectorPotential,
                            _,
                        >::new(potential),
                        flux_density_t: runmat_types::physical::Quantity::<
                            runmat_types::physical::kind::MagneticFluxDensity,
                            _,
                        >::new(flux),
                        magnetic_field_a_per_m: runmat_types::physical::Quantity::<
                            runmat_types::physical::kind::MagneticFieldIntensity,
                            _,
                        >::new(field),
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
        debug_assert!(assembled.permeability_scale.canonical_value() > 0.0);
        Ok(MagnetostaticResponse {
            edge_node_identities: assembled
                .edge_space
                .edges()
                .iter()
                .map(|edge| edge.node_identities)
                .collect(),
            edge_circulations_wb: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::MagneticFlux,
                _,
            >::new(edge_circulations),
            samples,
            magnetic_energy_j: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Energy,
                _,
            >::new(energy),
            source_work_j: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Energy,
                _,
            >::new(work),
            relative_physical_residual: physical_residual,
            relative_gauge_residual: gauge_residual,
            relative_energy_work_error: closure,
            iterations,
        })
    })
}
