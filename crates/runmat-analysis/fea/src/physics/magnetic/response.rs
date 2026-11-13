use super::{assembly::MagneticAssembly, MagnetostaticError};
use crate::algebra::{dot_canonical, spmv};
use runmat_meshing_core::StableDigest;
use runmat_types::physical::{QuantityField, QuantityLayout};

/// One retained sample with its geometry, materialized from a response and the projection
/// it was solved on. Not stored: the response keeps sample values only.
#[derive(Debug, Clone, PartialEq)]
pub struct MagneticFieldSample {
    pub element_identity: StableDigest,
    pub position_m: [f64; 3],
    pub volume_weight_m3: f64,
    pub vector_potential_wb_per_m: runmat_types::physical::Quantity<
        runmat_types::physical::kind::MagneticVectorPotential,
        [f64; 3],
    >,
    pub flux_density_t: runmat_types::physical::Quantity<
        runmat_types::physical::kind::MagneticFluxDensity,
        [f64; 3],
    >,
    pub magnetic_field_a_per_m: runmat_types::physical::Quantity<
        runmat_types::physical::kind::MagneticFieldIntensity,
        [f64; 3],
    >,
}

/// Samples per element: the Keast fifteen-point rule of the sampled layout.
pub const MAGNETIC_SAMPLES_PER_ELEMENT: usize = 15;

/// Retained magnetostatic response. Samples are named by (element, rule point) in canonical
/// element order: per-sample arrays hold three components each, per-element arrays follow the
/// sampled layout's element order, and circulations its canonical oriented-edge order. Sample
/// geometry and identities live in the mesh and the sampled layout, never here.
/// Fields are declared in canonical (UTF-8 key) order so a streamed encoding of a mesh-sized
/// response holds none of its collections back for sorting.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MagnetostaticResponse {
    #[serde(with = "runmat_types::physical::canonical_storage")]
    pub edge_circulations_wb:
        runmat_types::physical::Quantity<runmat_types::physical::kind::MagneticFlux, Vec<f64>>,
    /// Volume-weighted element mean of B over the element's samples, three per element.
    #[serde(with = "runmat_types::physical::canonical_storage")]
    pub element_flux_density_t: runmat_types::physical::Quantity<
        runmat_types::physical::kind::MagneticFluxDensity,
        Vec<f64>,
    >,
    /// Linear isotropic permeability per element; H = B / mu at every sample.
    #[serde(with = "runmat_types::physical::canonical_storage")]
    pub element_permeability_h_per_m: runmat_types::physical::Quantity<
        runmat_types::physical::kind::MagneticPermeability,
        Vec<f64>,
    >,
    #[serde(with = "runmat_types::physical::canonical_storage")]
    pub flux_density_t: runmat_types::physical::Quantity<
        runmat_types::physical::kind::MagneticFluxDensity,
        Vec<f64>,
    >,
    pub iterations: u32,
    #[serde(with = "runmat_types::physical::canonical_storage")]
    pub magnetic_energy_j:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Energy, f64>,
    pub relative_energy_work_error: f64,
    pub relative_gauge_residual: f64,
    pub relative_physical_residual: f64,
    #[serde(with = "runmat_types::physical::canonical_storage")]
    /// ∫J·A dV plus the reaction work of prescribed boundary potentials; equals twice the
    /// magnetic energy.
    pub source_work_j: runmat_types::physical::Quantity<runmat_types::physical::kind::Energy, f64>,
    #[serde(with = "runmat_types::physical::canonical_storage")]
    pub vector_potential_wb_per_m: runmat_types::physical::Quantity<
        runmat_types::physical::kind::MagneticVectorPotential,
        Vec<f64>,
    >,
}

impl MagnetostaticResponse {
    pub fn sample_count(&self) -> usize {
        self.flux_density_t.canonical_values().len() / 3
    }
    pub fn element_count(&self) -> usize {
        self.element_permeability_h_per_m.canonical_values().len()
    }
    pub fn vector_potential(&self, sample: usize) -> [f64; 3] {
        triple(self.vector_potential_wb_per_m.canonical_values(), sample)
    }
    pub fn flux_density(&self, sample: usize) -> [f64; 3] {
        triple(self.flux_density_t.canonical_values(), sample)
    }
    /// H = B / mu with the sample's element permeability.
    pub fn magnetic_field(&self, sample: usize) -> [f64; 3] {
        let mu = self.element_permeability_h_per_m.canonical_values()
            [sample / MAGNETIC_SAMPLES_PER_ELEMENT];
        self.flux_density(sample).map(|b| b / mu)
    }
    pub fn element_flux_density(&self, element: usize) -> [f64; 3] {
        triple(self.element_flux_density_t.canonical_values(), element)
    }
    pub fn element_magnetic_field(&self, element: usize) -> [f64; 3] {
        let mu = self.element_permeability_h_per_m.canonical_values()[element];
        self.element_flux_density(element).map(|b| b / mu)
    }
    /// Every sample with geometry from the projection it was solved on. Mesh-sized: for
    /// small models and tests.
    pub fn samples(
        &self,
        projection: &crate::topology::AnalysisProjection,
    ) -> Result<Vec<MagneticFieldSample>, MagnetostaticError> {
        if projection.elements.len() != self.element_count()
            || self.sample_count() != self.element_count() * MAGNETIC_SAMPLES_PER_ELEMENT
        {
            return Err(MagnetostaticError::Topology(
                "magnetic response differs from its projection".into(),
            ));
        }
        let mut samples = Vec::with_capacity(self.sample_count());
        for element in &projection.elements {
            projection
                .visit_element_quadrature(element, |point| {
                    let index = samples.len();
                    samples.push(MagneticFieldSample {
                        element_identity: element.stable_identity,
                        position_m: point.position_m,
                        volume_weight_m3: point.volume_weight_m3,
                        vector_potential_wb_per_m: runmat_types::physical::Quantity::new(
                            self.vector_potential(index),
                        ),
                        flux_density_t: runmat_types::physical::Quantity::new(
                            self.flux_density(index),
                        ),
                        magnetic_field_a_per_m: runmat_types::physical::Quantity::new(
                            self.magnetic_field(index),
                        ),
                    });
                })
                .map_err(|e| MagnetostaticError::Topology(e.to_string()))?;
        }
        Ok(samples)
    }
}
fn triple(values: &[f64], index: usize) -> [f64; 3] {
    [
        values[3 * index],
        values[3 * index + 1],
        values[3 * index + 2],
    ]
}

/// Volume-weighted mean over one element's samples, in sample order.
pub fn element_mean(weights: &[f64], values: impl Fn(usize) -> [f64; 3]) -> [f64; 3] {
    let volume: f64 = weights.iter().sum();
    let mut mean = [0.0; 3];
    for (sample, weight) in weights.iter().enumerate() {
        let v = values(sample);
        for axis in 0..3 {
            mean[axis] += v[axis] * weight / volume;
        }
    }
    mean
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
                other => {
                    MagnetostaticError::NotConverged(format!("sparse direct saddle solve: {other}"))
                }
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
        // Element samples are recovered on the admitted pool in bounded blocks, straight into
        // the retained arrays; the energy/work sums then run in canonical point order, so
        // totals are identical to a serial recovery.
        const BLOCK: usize = 1 << 16;
        let elements = assembled.volumes.len();
        let mut vector_potential = Vec::with_capacity(count * 3);
        let mut flux_density = Vec::with_capacity(count * 3);
        let mut element_flux = Vec::with_capacity(elements * 3);
        let mut permeabilities = Vec::with_capacity(elements);
        let mut energy = 0.0;
        let mut work = 0.0;
        for start in (0..elements).step_by(BLOCK) {
            let block = crate::algebra::element_assembly::ordered_map_range(
                BLOCK.min(elements - start),
                |offset| {
                    let index = start + offset;
                    let data = &assembled.volumes[index];
                    let indices = assembled.edge_space.element_edge_indices(index).unwrap();
                    let permeability = data.permeability_h_per_m.canonical_value();
                    let current_density = data.current_density_a_per_m2.canonical_value();
                    let mut potentials = [[0.0; 3]; MAGNETIC_SAMPLES_PER_ELEMENT];
                    let mut fluxes = [[0.0; 3]; MAGNETIC_SAMPLES_PER_ELEMENT];
                    let mut weights = [0.0; MAGNETIC_SAMPLES_PER_ELEMENT];
                    let mut terms = [(0.0, 0.0); MAGNETIC_SAMPLES_PER_ELEMENT];
                    let mut ordinal = 0;
                    assembled
                        .edge_space
                        .visit_element_quadrature(assembled.projection, index, |point| {
                            if ordinal >= MAGNETIC_SAMPLES_PER_ELEMENT {
                                ordinal += 1;
                                return;
                            }
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
                            terms[ordinal] = (
                                0.5 * point.volume_weight_m3 * dot_canonical(&flux, &field),
                                point.volume_weight_m3
                                    * dot_canonical(&potential, &current_density),
                            );
                            potentials[ordinal] = potential;
                            fluxes[ordinal] = flux;
                            weights[ordinal] = point.volume_weight_m3;
                            ordinal += 1;
                        })
                        .map_err(|e| MagnetostaticError::Topology(e.to_string()))?;
                    if ordinal != MAGNETIC_SAMPLES_PER_ELEMENT {
                        return Err(MagnetostaticError::Topology(
                            "magnetic element quadrature differs from the sampled rule".into(),
                        ));
                    }
                    let mean = element_mean(&weights, |sample| fluxes[sample]);
                    Ok::<_, MagnetostaticError>((potentials, fluxes, mean, permeability, terms))
                },
            );
            for element in block {
                let (potentials, fluxes, mean, permeability, terms) = element?;
                for (energy_term, work_term) in terms {
                    energy += energy_term;
                    work += work_term;
                }
                vector_potential.extend(potentials.into_iter().flatten());
                flux_density.extend(fluxes.into_iter().flatten());
                element_flux.extend(mean);
                permeabilities.push(permeability);
            }
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
            edge_circulations_wb: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::MagneticFlux,
                _,
            >::new(edge_circulations),
            element_flux_density_t: runmat_types::physical::Quantity::new(element_flux),
            element_permeability_h_per_m: runmat_types::physical::Quantity::new(permeabilities),
            flux_density_t: runmat_types::physical::Quantity::new(flux_density),
            vector_potential_wb_per_m: runmat_types::physical::Quantity::new(vector_potential),
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
