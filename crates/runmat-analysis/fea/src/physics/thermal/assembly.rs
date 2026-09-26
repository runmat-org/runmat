use runmat_analysis_core::{
    AnalysisConstraint, AnalysisModel, BoundaryOperator, ThermalConstraintKind,
};
use thiserror::Error;

use crate::{
    algebra::{
        spmv, AssembledRealSystem, CanonicalRealOperators, ContributionDomain, ContributionId,
        ContributionProvenance, MatrixProperties, NullspaceEliminationEvidence, OperatorRole,
        RealSparseAssembler, RealSparseMatrix,
    },
    excitation::{
        sample_prescribed_heat_flux, sample_prescribed_temperature, ExcitationSamplePoint,
        RealizedExcitationKind, RealizedExcitationTarget, RealizedExcitations,
        ScalarProjectionAccumulator, SpatialIntegrationRule, SpatialProjectionEvidence,
    },
    topology::{AnalysisProjection, ProjectedElement},
};

use super::{constitutive_stats, ThermalConstitutiveStats};

#[cfg(test)]
mod convergence_tests;
mod recovery;
mod temperature_field;

pub(crate) use recovery::ThermalEnergyBalance;

#[derive(Debug, Clone)]
pub(crate) struct ThermalOperatorAssembly {
    pub property_identity: crate::properties::CompiledPropertyMapIdentity,
    pub constitutive_stats: ThermalConstitutiveStats,
    pub projection: AnalysisProjection,
    pub conductivity: RealSparseMatrix,
    pub capacity: RealSparseMatrix,
    pub source: Vec<f64>,
    pub source_evidence: Vec<SpatialProjectionEvidence>,
    pub prescribed_temperature_k: Vec<Option<f64>>,
    pub positive_robin_weight: Vec<f64>,
    pub element_conductivity_w_per_mk: Vec<f64>,
    pub element_volumetric_capacity_j_per_m3k: Vec<f64>,
}

impl ThermalOperatorAssembly {
    pub(crate) fn steady_operator(&self) -> Result<AssembledRealSystem, ThermalAssemblyError> {
        let stiffness = self.conductivity.clone();
        let mut rhs = self.source.clone();
        apply_prescribed_values(&self.prescribed_temperature_k, &mut rhs);
        let constrained = self
            .prescribed_temperature_k
            .iter()
            .map(Option::is_some)
            .collect::<Vec<_>>();
        let mass = self.capacity.clone();
        let damping = RealSparseMatrix::diagonal(
            vec![0.0; self.projection.nodes.len()],
            OperatorRole::Damping,
            MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
        )
        .map_err(|error| ThermalAssemblyError::SparseOperator(format!("{error:?}")))?;
        let operators = CanonicalRealOperators::try_new(stiffness, mass, damping)
            .map_err(|error| ThermalAssemblyError::SparseOperator(format!("{error:?}")))?;
        let evidence = match NullspaceEliminationEvidence::prove_thermal(
            &self.projection,
            operators.stiffness(),
            &constrained,
            &self.positive_robin_weight,
        ) {
            Ok(evidence) => Some(evidence),
            Err(crate::algebra::NullspaceEvidenceError::UnderconstrainedComponent) => None,
            Err(error) => return Err(ThermalAssemblyError::SparseOperator(error.to_string())),
        };
        AssembledRealSystem::try_new_with_nullspace_evidence(operators, constrained, rhs, evidence)
            .map_err(|error| ThermalAssemblyError::SparseOperator(format!("{error:?}")))
    }

    pub(crate) fn transient_operator(
        &self,
        time_step_s: f64,
        previous_temperature_k: &[f64],
    ) -> Result<AssembledRealSystem, ThermalAssemblyError> {
        let inverse_time_step = time_step_s.recip();
        let support = self
            .conductivity
            .pattern()
            .union(self.capacity.pattern())
            .map_err(|error| ThermalAssemblyError::SparseOperator(error.to_string()))?;
        let mut matrix = RealSparseAssembler::with_pattern(
            support,
            OperatorRole::EffectiveTangent,
            MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
        )
        .map_err(|error| ThermalAssemblyError::SparseOperator(error.to_string()))?;
        add_scaled_matrix(&mut matrix, &self.conductivity, 1.0);
        add_scaled_matrix(&mut matrix, &self.capacity, inverse_time_step);
        let stiffness = matrix
            .finish()
            .map_err(|error| ThermalAssemblyError::SparseOperator(format!("{error:?}")))?;
        let capacity_temperature = apply_matrix(&self.capacity, previous_temperature_k);
        let mut rhs = self
            .source
            .iter()
            .zip(capacity_temperature)
            .map(|(source, stored_energy)| source + inverse_time_step * stored_energy)
            .collect::<Vec<_>>();
        apply_prescribed_values(&self.prescribed_temperature_k, &mut rhs);
        let constrained = self
            .prescribed_temperature_k
            .iter()
            .map(Option::is_some)
            .collect::<Vec<_>>();
        let mass = self.capacity.clone();
        let damping = RealSparseMatrix::diagonal(
            vec![0.0; self.projection.nodes.len()],
            OperatorRole::Damping,
            MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
        )
        .map_err(|error| ThermalAssemblyError::SparseOperator(format!("{error:?}")))?;
        let operators = CanonicalRealOperators::try_new(stiffness, mass, damping)
            .map_err(|error| ThermalAssemblyError::SparseOperator(format!("{error:?}")))?;
        AssembledRealSystem::try_new(operators, constrained, rhs)
            .map_err(|error| ThermalAssemblyError::SparseOperator(format!("{error:?}")))
    }
}

#[derive(Debug, Error, Clone, PartialEq)]
pub(crate) enum ThermalAssemblyError {
    #[error("thermal sparse operator construction failed: {0}")]
    SparseOperator(String),
    #[error("thermal analysis projection is invalid: {0}")]
    InvalidProjection(String),
    #[error("volume region {region_id} has no thermal material assignment")]
    MissingMaterial { region_id: String },
    #[error("thermal material coverage is invalid: {0}")]
    InvalidMaterialCoverage(String),
    #[error("thermal excitation {excitation_id} uses an unsupported complex ordinate")]
    ComplexSource { excitation_id: String },
    #[error("thermal excitation {excitation_id} does not resolve any {entity_kind}")]
    UnmappedSource {
        excitation_id: String,
        entity_kind: &'static str,
    },
    #[error("thermal excitation {excitation_id} failed scalar conservation with relative error {relative_error} above tolerance {relative_tolerance}")]
    ProjectionConservation {
        excitation_id: String,
        relative_error: f64,
        relative_tolerance: f64,
    },
    #[error(
        "thermal boundary or constraint {operator_id} signal could not be realized: {message}"
    )]
    SignalRealization {
        operator_id: String,
        message: String,
    },
    #[error("thermal boundary or constraint {operator_id} uses an unsupported complex ordinate")]
    ComplexBoundary { operator_id: String },
    #[error("thermal boundary or constraint {operator_id} does not resolve any boundary faces")]
    UnmappedBoundary { operator_id: String },
    #[error("thermal constraints prescribe conflicting temperatures at solver node {node_id}")]
    ConflictingTemperature { node_id: u64 },
    #[error("temperature-field excitation {excitation_id} uses an unsupported complex ordinate")]
    ComplexTemperatureField { excitation_id: String },
    #[error("temperature-field excitation {excitation_id} does not resolve any solver nodes")]
    UnmappedTemperatureField { excitation_id: String },
    #[error(
        "temperature-field excitations {first_excitation_id} and {second_excitation_id} prescribe conflicting temperatures at {location}"
    )]
    ConflictingTemperatureFields {
        first_excitation_id: String,
        second_excitation_id: String,
        location: String,
    },
}

pub(crate) fn assemble(
    model: &AnalysisModel,
    mesh: &runmat_meshing_core::SolverMeshArtifact,
    support: &crate::algebra::SparsePattern,
    excitations: &RealizedExcitations,
    sample: ExcitationSamplePoint<'_>,
) -> Result<ThermalOperatorAssembly, ThermalAssemblyError> {
    let projection = AnalysisProjection::derive(mesh)
        .map_err(|error| ThermalAssemblyError::InvalidProjection(error.to_string()))?;
    let properties = crate::properties::compile_thermal_property_map(model, Some(mesh))
        .map_err(|error| ThermalAssemblyError::InvalidMaterialCoverage(error.to_string()))?;
    if support.rows() != projection.nodes.len() || support.columns() != projection.nodes.len() {
        return Err(ThermalAssemblyError::SparseOperator(
            "thermal planned support shape mismatch".into(),
        ));
    }
    let mut conductivity = RealSparseAssembler::with_pattern(
        support
            .try_clone()
            .map_err(|error| ThermalAssemblyError::SparseOperator(error.to_string()))?,
        OperatorRole::Conductivity,
        MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
    )
    .map_err(|error| ThermalAssemblyError::SparseOperator(error.to_string()))?;
    let mut capacity = RealSparseAssembler::with_pattern(
        support
            .try_clone()
            .map_err(|error| ThermalAssemblyError::SparseOperator(error.to_string()))?,
        OperatorRole::Capacity,
        MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
    )
    .map_err(|error| ThermalAssemblyError::SparseOperator(error.to_string()))?;
    conductivity.add_contribution(ContributionProvenance {
        domain: ContributionDomain::Thermal,
        contribution_id: ContributionId("conductive_element_and_boundary_terms".to_string()),
    });
    capacity.add_contribution(ContributionProvenance {
        domain: ContributionDomain::Thermal,
        contribution_id: ContributionId("consistent_heat_capacity".to_string()),
    });
    let mut element_conductivity_w_per_mk = Vec::with_capacity(projection.elements.len());
    let mut element_volumetric_capacity_j_per_m3k = Vec::with_capacity(projection.elements.len());
    let mut element_specific_heat_j_per_kgk = Vec::with_capacity(projection.elements.len());
    let mut element_density_kg_per_m3 = Vec::with_capacity(projection.elements.len());

    for element in &projection.elements {
        let material = &properties
            .solid_material(element.stable_identity)
            .ok_or_else(|| ThermalAssemblyError::MissingMaterial {
                region_id: element.region_id.source_topology_id.clone(),
            })?
            .material;
        let thermal = material
            .require_thermal()
            .map_err(|error| ThermalAssemblyError::InvalidMaterialCoverage(error.to_string()))?;
        let mass_density_kg_per_m3 = material
            .require_mass_density()
            .map_err(|error| ThermalAssemblyError::InvalidMaterialCoverage(error.to_string()))?;
        element_conductivity_w_per_mk.push(thermal.conductivity_w_per_mk);
        element_specific_heat_j_per_kgk.push(thermal.specific_heat_j_per_kgk);
        element_density_kg_per_m3.push(mass_density_kg_per_m3);
        element_volumetric_capacity_j_per_m3k
            .push(mass_density_kg_per_m3 * thermal.specific_heat_j_per_kgk);
        projection
            .visit_element_quadrature(element, |point| {
                for (local_row, &row) in element.node_indices.iter().enumerate() {
                    for (local_column, &column) in element.node_indices.iter().enumerate() {
                        let gradient_product = point.physical_shape_gradients()[local_row]
                            .iter()
                            .zip(point.physical_shape_gradients()[local_column])
                            .map(|(left, right)| left * right)
                            .sum::<f64>();
                        conductivity.add(
                            row,
                            column,
                            thermal.conductivity_w_per_mk
                                * gradient_product
                                * point.volume_weight_m3,
                        );
                        capacity.add(
                            row,
                            column,
                            mass_density_kg_per_m3
                                * thermal.specific_heat_j_per_kgk
                                * point.shape_values()[local_row]
                                * point.shape_values()[local_column]
                                * point.volume_weight_m3,
                        );
                    }
                }
            })
            .map_err(|error| ThermalAssemblyError::InvalidProjection(error.to_string()))?;
    }

    let mut source = vec![0.0; projection.nodes.len()];
    let mut source_evidence = Vec::new();
    for excitation in excitations.iter() {
        let (value, rule) = match excitation.kind {
            RealizedExcitationKind::HeatGeneration(value) => {
                (value, SpatialIntegrationRule::VolumeKeastFifteen)
            }
            RealizedExcitationKind::HeatFlux(value) => {
                (value, SpatialIntegrationRule::BoundaryDunavantSeven)
            }
            _ => continue,
        };
        if value.imaginary != 0.0 {
            return Err(ThermalAssemblyError::ComplexSource {
                excitation_id: excitation.excitation_id.0.clone(),
            });
        }
        let entity_count = match excitation.kind {
            RealizedExcitationKind::HeatGeneration(_) => projection
                .elements
                .iter()
                .filter(|element| volume_matches(element, &excitation.target))
                .count(),
            RealizedExcitationKind::HeatFlux(_) => {
                boundary_faces_for_target(&projection, &excitation.target).count()
            }
            _ => 0,
        };
        if entity_count == 0 {
            return Err(ThermalAssemblyError::UnmappedSource {
                excitation_id: excitation.excitation_id.0.clone(),
                entity_kind: match excitation.kind {
                    RealizedExcitationKind::HeatGeneration(_) => "volume elements",
                    _ => "boundary faces",
                },
            });
        }
        let mut evidence =
            ScalarProjectionAccumulator::new(excitation, &projection, rule, entity_count)
                .map_err(|error| ThermalAssemblyError::InvalidProjection(error.to_string()))?;
        match excitation.kind {
            RealizedExcitationKind::HeatGeneration(_) => {
                for element in projection
                    .elements
                    .iter()
                    .filter(|element| volume_matches(element, &excitation.target))
                {
                    projection
                        .visit_element_quadrature(element, |point| {
                            let integrated = value.real * point.volume_weight_m3;
                            evidence.add_requested(integrated, 0.0);
                            for (&node, &shape) in
                                element.node_indices.iter().zip(point.shape_values())
                            {
                                let nodal = integrated * shape;
                                source[node] += nodal;
                                evidence.add_projected(node, nodal, 0.0);
                            }
                        })
                        .map_err(|error| {
                            ThermalAssemblyError::InvalidProjection(error.to_string())
                        })?;
                }
            }
            RealizedExcitationKind::HeatFlux(_) => {
                for face in boundary_faces_for_target(&projection, &excitation.target) {
                    projection
                        .visit_boundary_quadrature(face, |point| {
                            let integrated = value.real * point.area_weight_m2;
                            evidence.add_requested(integrated, 0.0);
                            for (&node, &shape) in
                                face.node_indices.iter().zip(point.shape_values())
                            {
                                let nodal = integrated * shape;
                                source[node] += nodal;
                                evidence.add_projected(node, nodal, 0.0);
                            }
                        })
                        .map_err(|error| {
                            ThermalAssemblyError::InvalidProjection(error.to_string())
                        })?;
                }
            }
            _ => unreachable!(),
        }
        let evidence = evidence.finish();
        evidence
            .validate()
            .map_err(|error| ThermalAssemblyError::InvalidProjection(error.to_string()))?;
        if !evidence.accepted {
            return Err(ThermalAssemblyError::ProjectionConservation {
                excitation_id: evidence.excitation_id.clone(),
                relative_error: evidence.resultant_relative_error(),
                relative_tolerance: evidence.requirements.relative_tolerance,
            });
        }
        source_evidence.push(evidence);
    }

    let mut prescribed_temperature_k =
        temperature_field::prescribed_temperatures(&projection, excitations)?;
    let mut positive_robin_weight = vec![0.0; projection.nodes.len()];
    for constraint in &model.constraints {
        let AnalysisConstraint::Thermal(constraint) = constraint else {
            continue;
        };
        let ThermalConstraintKind::Prescribed { target, value } = &constraint.kind else {
            continue;
        };
        let temperature =
            sample_prescribed_temperature(value, &model.signals, sample).map_err(|error| {
                ThermalAssemblyError::SignalRealization {
                    operator_id: constraint.constraint_id.clone(),
                    message: error.to_string(),
                }
            })?;
        if temperature.imaginary != 0.0 {
            return Err(ThermalAssemblyError::ComplexBoundary {
                operator_id: constraint.constraint_id.clone(),
            });
        }
        let mut faces = projection
            .boundary_faces_matching(&target.region_id)
            .peekable();
        if faces.peek().is_none() {
            return Err(ThermalAssemblyError::UnmappedBoundary {
                operator_id: constraint.constraint_id.clone(),
            });
        }
        for face in faces {
            for &node_index in &face.node_indices {
                if let Some(existing) = prescribed_temperature_k[node_index] {
                    if existing != temperature.real {
                        return Err(ThermalAssemblyError::ConflictingTemperature {
                            node_id: projection.nodes[node_index].node_id,
                        });
                    }
                }
                prescribed_temperature_k[node_index] = Some(temperature.real);
            }
        }
    }

    for boundary in &model.boundaries {
        let mut faces = projection
            .boundary_faces_matching(&boundary.target.region_id)
            .peekable();
        match &boundary.operator {
            BoundaryOperator::ThermalHeatFlux { value } => {
                let heat_flux = sample_prescribed_heat_flux(value, &model.signals, sample)
                    .map_err(|error| ThermalAssemblyError::SignalRealization {
                        operator_id: boundary.boundary_id.clone(),
                        message: error.to_string(),
                    })?;
                if heat_flux.imaginary != 0.0 {
                    return Err(ThermalAssemblyError::ComplexBoundary {
                        operator_id: boundary.boundary_id.clone(),
                    });
                }
                if faces.peek().is_none() {
                    return Err(ThermalAssemblyError::UnmappedBoundary {
                        operator_id: boundary.boundary_id.clone(),
                    });
                }
                for face in faces {
                    projection
                        .visit_boundary_quadrature(face, |point| {
                            let power = heat_flux.real * point.area_weight_m2;
                            for (&node, &shape) in
                                face.node_indices.iter().zip(point.shape_values())
                            {
                                source[node] += power * shape;
                            }
                        })
                        .map_err(|error| {
                            ThermalAssemblyError::InvalidProjection(error.to_string())
                        })?;
                }
            }
            BoundaryOperator::ThermalConvection {
                ambient_temperature,
                coefficient,
            } => {
                let ambient =
                    sample_prescribed_temperature(ambient_temperature, &model.signals, sample)
                        .map_err(|error| ThermalAssemblyError::SignalRealization {
                            operator_id: boundary.boundary_id.clone(),
                            message: error.to_string(),
                        })?;
                if ambient.imaginary != 0.0 {
                    return Err(ThermalAssemblyError::ComplexBoundary {
                        operator_id: boundary.boundary_id.clone(),
                    });
                }
                if faces.peek().is_none() {
                    return Err(ThermalAssemblyError::UnmappedBoundary {
                        operator_id: boundary.boundary_id.clone(),
                    });
                }
                let heat_transfer = coefficient.watts_per_square_meter_kelvin;
                for face in faces {
                    projection
                        .visit_boundary_quadrature(face, |point| {
                            for (local_row, &row) in face.node_indices.iter().enumerate() {
                                let row_shape = point.shape_values()[local_row];
                                if heat_transfer > 0.0 {
                                    positive_robin_weight[row] += heat_transfer
                                        * row_shape
                                        * row_shape
                                        * point.area_weight_m2;
                                }
                                source[row] +=
                                    heat_transfer * ambient.real * row_shape * point.area_weight_m2;
                                for (local_column, &column) in face.node_indices.iter().enumerate()
                                {
                                    conductivity.add(
                                        row,
                                        column,
                                        heat_transfer
                                            * row_shape
                                            * point.shape_values()[local_column]
                                            * point.area_weight_m2,
                                    );
                                }
                            }
                        })
                        .map_err(|error| {
                            ThermalAssemblyError::InvalidProjection(error.to_string())
                        })?;
                }
            }
            _ => {}
        }
    }

    let constitutive_stats = constitutive_stats(
        &element_conductivity_w_per_mk,
        &element_specific_heat_j_per_kgk,
        &element_density_kg_per_m3,
    )
    .ok_or_else(|| {
        ThermalAssemblyError::InvalidMaterialCoverage(
            "thermal property projection produced no complete element coefficients".into(),
        )
    })?;

    Ok(ThermalOperatorAssembly {
        property_identity: properties.identity.clone(),
        constitutive_stats,
        projection,
        conductivity: conductivity
            .finish()
            .map_err(|error| ThermalAssemblyError::SparseOperator(format!("{error:?}")))?,
        capacity: capacity
            .finish()
            .map_err(|error| ThermalAssemblyError::SparseOperator(format!("{error:?}")))?,
        source,
        source_evidence,
        prescribed_temperature_k,
        positive_robin_weight,
        element_conductivity_w_per_mk,
        element_volumetric_capacity_j_per_m3k,
    })
}

fn volume_matches(element: &ProjectedElement, target: &RealizedExcitationTarget) -> bool {
    match target {
        RealizedExcitationTarget::WholeModel => true,
        RealizedExcitationTarget::Region(target) => {
            target.matches_persistent_entity(&element.region_id)
        }
        _ => false,
    }
}

fn boundary_faces_for_target<'a>(
    projection: &'a AnalysisProjection,
    target: &'a RealizedExcitationTarget,
) -> impl Iterator<Item = &'a crate::topology::ProjectedBoundaryFace> + 'a {
    projection.boundary_faces.iter().filter(move |face| {
        let RealizedExcitationTarget::Region(target) = target else {
            return false;
        };
        face.provenance
            .iter()
            .any(|entity| target.matches_persistent_entity(entity))
    })
}

fn add_scaled_matrix(assembler: &mut RealSparseAssembler, matrix: &RealSparseMatrix, scale: f64) {
    for row in 0..matrix.rows() {
        for entry in matrix.pattern().row_range(row) {
            assembler.add(
                row,
                matrix.pattern().column(entry),
                scale * matrix.values()[entry],
            );
        }
    }
}

pub(super) fn apply_matrix(matrix: &RealSparseMatrix, vector: &[f64]) -> Vec<f64> {
    spmv(matrix, vector).expect("validated thermal operator dimensions")
}

fn apply_prescribed_values(prescribed: &[Option<f64>], rhs: &mut [f64]) {
    for (right_hand_side, value) in rhs.iter_mut().zip(prescribed) {
        if let Some(value) = value {
            *right_hand_side = *value;
        }
    }
}

#[cfg(test)]
mod tests;
