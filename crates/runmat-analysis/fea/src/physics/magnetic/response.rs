use super::{assembly::MagneticAssembly, MagnetostaticError};
use crate::algebra::{dot_canonical, spmv};
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
/// Fields are declared in canonical (UTF-8 key) order so a streamed encoding of a mesh-sized
/// response holds none of its edge or sample collections back for sorting.
pub struct MagnetostaticResponse {
    #[serde(with = "runmat_types::physical::canonical_storage")]
    pub edge_circulations_wb:
        runmat_types::physical::Quantity<runmat_types::physical::kind::MagneticFlux, Vec<f64>>,
    pub edge_node_identities: Vec<[StableDigest; 2]>,
    pub iterations: u32,
    #[serde(with = "runmat_types::physical::canonical_storage")]
    pub magnetic_energy_j:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Energy, f64>,
    pub relative_energy_work_error: f64,
    pub relative_gauge_residual: f64,
    pub relative_physical_residual: f64,
    pub samples: Vec<MagneticFieldSample>,
    #[serde(with = "runmat_types::physical::canonical_storage")]
    /// ∫J·A dV plus the reaction work of prescribed boundary potentials; equals twice the
    /// magnetic energy.
    pub source_work_j: runmat_types::physical::Quantity<runmat_types::physical::kind::Energy, f64>,
}

pub(super) fn solve(
    assembled: MagneticAssembly<'_>,
    tolerance: f64,
) -> Result<MagnetostaticResponse, MagnetostaticError> {
    let n = assembled.saddle.input_layout().component_count();
    // Weak Coulomb-gauge saddle [curl-curl G; Gᵀ 0]: symmetric indefinite with a
    // material-contrast-dependent spectrum (iron next to air), factored directly
    // (Bunch–Kaufman, signed-regularized LDLᵀ fallback) with refinement.
    let (solution, iterations) = if n == 0 {
        (Vec::new(), 0)
    } else {
        let (solution, _) = assembled
            .saddle
            .with_canonical_storage(|saddle| {
                crate::solve::shared_sparse::solve_direct_real(
                    saddle,
                    assembled.scaled_rhs.canonical_values(),
                    crate::solve::shared_sparse::DirectStructure::SaddlePoint {
                        primal_count: assembled.boundary.free_edges.len(),
                    },
                )
            })
            .map_err(|error| match error {
                runmat_sparse_solvers::SolverError::Cancelled => {
                    MagnetostaticError::NotConverged("cancelled".into())
                }
                other => MagnetostaticError::NotConverged(format!(
                    "sparse direct saddle solve: {other}"
                )),
            })?;
        (solution, 1)
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
        // Gauge rows carry the lifted prescribed potential on their right-hand side.
        let gauge_defect: Vec<f64> = saddle_action[free_count..]
            .iter()
            .zip(&assembled.scaled_rhs.canonical_values()[free_count..])
            .map(|(action, rhs)| action - rhs)
            .collect();
        let gauge_residual = if rhs_norm > 0.0 {
            norm(&gauge_defect) / rhs_norm
        } else {
            norm(&gauge_defect)
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
        let mut edge_circulations = assembled.grounded_circulation.canonical_values().clone();
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
        // Element samples are recovered on the admitted pool; the energy/work sums then run
        // in canonical point order, so totals are identical to a serial recovery.
        let per_element = crate::algebra::element_assembly::ordered_map_range(
            assembled.volumes.len(),
            |index| {
                let data = &assembled.volumes[index];
                let indices = assembled.edge_space.element_edge_indices(index).unwrap();
                let permeability = data.permeability_h_per_m.canonical_value();
                let current_density = data.current_density_a_per_m2.canonical_value();
                let mut element_samples = Vec::with_capacity(15);
                let mut terms = Vec::with_capacity(15);
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
                        let field = flux.map(|v| v / permeability);
                        terms.push((
                            0.5 * point.volume_weight_m3 * dot_canonical(&flux, &field),
                            point.volume_weight_m3 * dot_canonical(&potential, &current_density),
                        ));
                        element_samples.push(MagneticFieldSample {
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
                Ok::<_, MagnetostaticError>((element_samples, terms))
            },
        );
        let mut samples = Vec::with_capacity(count);
        let mut energy = 0.0;
        let mut work = 0.0;
        for element in per_element {
            let (element_samples, terms) = element?;
            for (energy_term, work_term) in terms {
                energy += energy_term;
                work += work_term;
            }
            samples.extend(element_samples);
        }
        // A prescribed boundary potential does work through its reaction:
        // 2 W = integral(J . A) + sum over grounded edges g_e (K a - f)_e.
        let grounded = assembled.grounded_circulation.canonical_values();
        let mut boundary_work = 0.0;
        if grounded.iter().any(|value| *value != 0.0) {
            for (index, data) in assembled.volumes.iter().enumerate() {
                let indices = assembled.edge_space.element_edge_indices(index).unwrap();
                if indices.iter().all(|&edge| grounded[edge] == 0.0) {
                    continue;
                }
                let moments = assembled
                    .edge_space
                    .element_moments(assembled.projection, index)
                    .map_err(|e| MagnetostaticError::Topology(e.to_string()))?;
                let permeability = data.permeability_h_per_m.canonical_value();
                for i in 0..6 {
                    let prescribed = grounded[indices[i]];
                    if prescribed == 0.0 {
                        continue;
                    }
                    let action: f64 = (0..6)
                        .map(|j| {
                            moments.curl_curl[i][j] / permeability * edge_circulations[indices[j]]
                        })
                        .sum();
                    let load: f64 = moments.constant_vector_load[i]
                        .iter()
                        .zip(data.current_density_a_per_m2.canonical_value())
                        .map(|(n, j)| n * j)
                        .sum();
                    boundary_work += prescribed * (action - load);
                }
            }
        }
        let work = work + boundary_work;
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
