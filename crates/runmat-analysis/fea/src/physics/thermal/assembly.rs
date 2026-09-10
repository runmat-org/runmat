use std::collections::BTreeMap;

use runmat_analysis_core::{
    AnalysisConstraint, AnalysisModel, BoundaryOperator, MaterialModel, ThermalConstraintKind,
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

#[cfg(test)]
mod convergence_tests;
mod recovery;
mod temperature_field;

pub(crate) use recovery::ThermalEnergyBalance;

#[derive(Debug, Clone)]
pub(crate) struct ThermalOperatorAssembly {
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
        let mut matrix = RealSparseAssembler::square(
            self.projection.nodes.len(),
            OperatorRole::EffectiveTangent,
            MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
        );
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
    excitations: &RealizedExcitations,
    sample: ExcitationSamplePoint<'_>,
) -> Result<ThermalOperatorAssembly, ThermalAssemblyError> {
    let projection = AnalysisProjection::derive(mesh)
        .map_err(|error| ThermalAssemblyError::InvalidProjection(error.to_string()))?;
    let materials = thermal_materials(model);
    let default_material = (model.materials.len() == 1).then(|| &model.materials[0]);
    let mut conductivity = RealSparseAssembler::square(
        projection.nodes.len(),
        OperatorRole::Conductivity,
        MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
    );
    let mut capacity = RealSparseAssembler::square(
        projection.nodes.len(),
        OperatorRole::Capacity,
        MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
    );
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

    for element in &projection.elements {
        let material = material_for_element(element, default_material, &materials)?;
        element_conductivity_w_per_mk.push(material.thermal.conductivity_w_per_mk);
        element_volumetric_capacity_j_per_m3k
            .push(material.mechanical.density_kg_per_m3 * material.thermal.specific_heat_j_per_kgk);
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
                            material.thermal.conductivity_w_per_mk
                                * gradient_product
                                * point.volume_weight_m3,
                        );
                        capacity.add(
                            row,
                            column,
                            material.mechanical.density_kg_per_m3
                                * material.thermal.specific_heat_j_per_kgk
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
            RealizedExcitationKind::HeatFlux(_) => excitation
                .target
                .solver_region_id()
                .map(|region| projection.boundary_faces_matching(region).count())
                .unwrap_or(0),
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
            ScalarProjectionAccumulator::new(excitation, &projection, rule, entity_count);
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
                let region = excitation.target.solver_region_id().unwrap_or_default();
                for face in projection.boundary_faces_matching(region) {
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
        if !evidence.accepted {
            return Err(ThermalAssemblyError::ProjectionConservation {
                excitation_id: evidence.excitation_id.clone(),
                relative_error: evidence.resultant_relative_error(),
                relative_tolerance: evidence.relative_tolerance,
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
        let faces = projection
            .boundary_faces_matching(&target.region_id)
            .collect::<Vec<_>>();
        if faces.is_empty() {
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
        let faces = projection
            .boundary_faces_matching(&boundary.target.region_id)
            .collect::<Vec<_>>();
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
                if faces.is_empty() {
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
                if faces.is_empty() {
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

    Ok(ThermalOperatorAssembly {
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

fn thermal_materials(model: &AnalysisModel) -> BTreeMap<&str, &MaterialModel> {
    let materials = model
        .materials
        .iter()
        .map(|material| (material.material_id.as_str(), material))
        .collect::<BTreeMap<_, _>>();
    model
        .material_assignments
        .iter()
        .filter_map(|assignment| {
            materials
                .get(assignment.assigned_material_id.as_str())
                .copied()
                .map(|material| (assignment.region_id.as_str(), material))
        })
        .collect()
}

fn material_for_element<'a>(
    element: &ProjectedElement,
    default: Option<&'a MaterialModel>,
    assigned: &BTreeMap<&str, &'a MaterialModel>,
) -> Result<&'a MaterialModel, ThermalAssemblyError> {
    assigned
        .iter()
        .find_map(|(region, material)| {
            element
                .region_id
                .matches_analysis_selection(region)
                .then_some(*material)
        })
        .or(default)
        .ok_or_else(|| ThermalAssemblyError::MissingMaterial {
            region_id: element.region_id.source_topology_id.clone(),
        })
}

fn volume_matches(element: &ProjectedElement, target: &RealizedExcitationTarget) -> bool {
    match target {
        RealizedExcitationTarget::WholeModel => true,
        RealizedExcitationTarget::Region(region) => {
            element.region_id.matches_analysis_selection(region)
        }
        _ => false,
    }
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
mod tests {
    use runmat_analysis_core::{
        AnalysisBoundary, AnalysisConstraint, CoordinateFrame, ExcitationCaseId,
        ExcitationDistribution, ExcitationId, ExcitationKinematics, HeatFlux,
        HeatTransferCoefficient, PhysicalQuantityKind, PhysicalSignal, PrescribedValue,
        ProjectionRequirement, RegionTarget, Signal, SignalApproximationEvidence,
        SignalExtrapolation, SignalId, SignalInterpolation, SignalKnot, SignalOrdinate,
        SignalPrecision, SignalProvenance, SignalRepresentation, SignalShape, SignalValue,
        TabulatedSignal, TabulatedSignalDomain, Temperature, ThermalConstraint,
        ThermalConstraintKind,
    };
    use runmat_execution::Digest;
    use runmat_meshing_core::{ElementOrder, PersistentEntityId, PersistentEntityKind};

    use super::*;
    use crate::{
        excitation::{
            ComplexScalar, RealizedExcitation, RealizedExcitationKind, RealizedExcitationPolicy,
        },
        fixtures::{fixture_model, FixtureId},
        solve::{
            backend::{cpu_reference::CpuReferenceBackend, kind::LinearAlgebraBackendKind},
            linear::solve_operator_system,
            preconditioner::SpdPreconditionerKind,
        },
    };

    fn heat_generation(value: f64) -> RealizedExcitations {
        RealizedExcitations::new(vec![RealizedExcitation {
            excitation_id: ExcitationId("uniform_heat".into()),
            case_id: ExcitationCaseId("uniform_heat_case".into()),
            signal_id: SignalId("uniform_heat_signal".into()),
            quantity: PhysicalQuantityKind::HeatGenerationDensity,
            target: RealizedExcitationTarget::WholeModel,
            kind: RealizedExcitationKind::HeatGeneration(ComplexScalar::real(value)),
            policy: RealizedExcitationPolicy {
                frame: CoordinateFrame::Global,
                distribution: ExcitationDistribution::IntensityPerVolume,
                kinematics: ExcitationKinematics::FixedFrame,
                projection: ProjectionRequirement {
                    preserve_resultant: true,
                    preserve_moment: false,
                    relative_tolerance: 1.0e-12,
                },
            },
        }])
    }

    fn thermal_reference_model() -> (
        AnalysisModel,
        runmat_meshing_core::SolverMeshArtifact,
        usize,
        String,
    ) {
        let mut model = fixture_model(FixtureId::CantileverLinearStatic);
        model.constraints.clear();
        model.boundaries.clear();
        model.excitations.clear();
        let material = &mut model.materials[0];
        material.thermal.conductivity_w_per_mk = 1.0;
        material.thermal.specific_heat_j_per_kgk = 1.0;
        material.mechanical.density_kg_per_m3 = 1.0;

        let mut mesh =
            runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
        let coordinates = mesh
            .topology
            .nodes
            .iter()
            .map(|node| (node.node_id, node.coordinates_m))
            .collect::<BTreeMap<_, _>>();
        let base_face = mesh
            .topology
            .boundary_faces
            .iter_mut()
            .find(|face| {
                face.node_ids
                    .iter()
                    .all(|node| coordinates[node][2].abs() <= 1.0e-12)
            })
            .expect("canonical tetrahedron has a z=0 base face");
        let base_region = PersistentEntityId {
            kind: PersistentEntityKind::Face,
            source_topology_id: "thermal_base".into(),
            assembly_path: vec!["root".into()],
        };
        base_face.provenance.push(base_region.clone());
        base_face.provenance.sort();
        let base_nodes = base_face.node_ids.clone();
        let free_node_id = mesh
            .topology
            .nodes
            .iter()
            .find(|node| !base_nodes.contains(&node.node_id))
            .expect("one tetrahedron node is opposite its base")
            .node_id;
        let free_node_index = mesh
            .topology
            .nodes
            .iter()
            .position(|node| node.node_id == free_node_id)
            .unwrap();
        mesh.seal_canonical_digest().unwrap();
        (model, mesh, free_node_index, base_region.source_topology_id)
    }

    fn prescribe_base(model: &mut AnalysisModel, base_region: String, temperature_k: f64) {
        model
            .constraints
            .push(AnalysisConstraint::Thermal(ThermalConstraint {
                constraint_id: "base_temperature".into(),
                kind: ThermalConstraintKind::Prescribed {
                    target: RegionTarget::boundary(base_region),
                    value: PrescribedValue::Constant(Temperature::new(temperature_k)),
                },
            }));
    }

    #[test]
    fn steady_nullspace_proof_requires_an_anchor_in_every_component() {
        let (mut model, mesh, _, base_region) = thermal_reference_model();
        prescribe_base(&mut model, base_region, 300.0);
        let assembly = assemble(
            &model,
            &mesh,
            &RealizedExcitations::default(),
            ExcitationSamplePoint::Static,
        )
        .unwrap();
        let anchored = assembly.steady_operator().unwrap();
        assert!(anchored.nullspace_evidence().is_some());

        let mut disconnected_projection = assembly.projection.clone();
        disconnected_projection
            .node_neighbors
            .iter_mut()
            .for_each(Vec::clear);
        let constrained = assembly
            .prescribed_temperature_k
            .iter()
            .map(Option::is_some)
            .collect::<Vec<_>>();
        assert_eq!(
            NullspaceEliminationEvidence::prove_thermal(
                &disconnected_projection,
                &assembly.conductivity,
                &constrained,
                &assembly.positive_robin_weight,
            ),
            Err(crate::algebra::NullspaceEvidenceError::NullModeViolation)
        );

        let unanchored = assemble(
            &thermal_reference_model().0,
            &mesh,
            &RealizedExcitations::default(),
            ExcitationSamplePoint::Static,
        )
        .unwrap()
        .steady_operator()
        .unwrap();
        assert!(unanchored.nullspace_evidence().is_none());
    }

    #[test]
    fn thermal_nullspace_proof_rejects_metadata_only_semidefinite_claims() {
        let (mut model, mesh, _, base_region) = thermal_reference_model();
        prescribe_base(&mut model, base_region, 300.0);
        let assembly = assemble(
            &model,
            &mesh,
            &RealizedExcitations::default(),
            ExcitationSamplePoint::Static,
        )
        .unwrap();
        let constrained = assembly
            .prescribed_temperature_k
            .iter()
            .map(Option::is_some)
            .collect::<Vec<_>>();
        let false_semidefinite = RealSparseMatrix::diagonal(
            vec![1.0; assembly.projection.nodes.len()],
            OperatorRole::Conductivity,
            MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
        )
        .unwrap();

        assert_eq!(
            NullspaceEliminationEvidence::prove_thermal(
                &assembly.projection,
                &false_semidefinite,
                &constrained,
                &assembly.positive_robin_weight,
            ),
            Err(crate::algebra::NullspaceEvidenceError::NullModeViolation),
        );
    }

    #[test]
    fn positive_convection_anchors_a_steady_thermal_component() {
        let (mut model, mesh, _, _) = thermal_reference_model();
        let projection = AnalysisProjection::derive(&mesh).unwrap();
        let selector = projection.boundary_faces[0].provenance[0]
            .analysis_selection_id()
            .unwrap();
        model.boundaries.push(AnalysisBoundary {
            boundary_id: "positive_robin_anchor".into(),
            target: RegionTarget::boundary(selector),
            operator: BoundaryOperator::ThermalConvection {
                ambient_temperature: PrescribedValue::Constant(Temperature::new(295.0)),
                coefficient: HeatTransferCoefficient::new(4.0),
            },
        });
        let system = assemble(
            &model,
            &mesh,
            &RealizedExcitations::default(),
            ExcitationSamplePoint::Static,
        )
        .unwrap()
        .steady_operator()
        .unwrap();
        let prepared = system.prepare_linear().unwrap();
        assert!(prepared.admissibility().is_some());
        assert!(crate::algebra::validate_prepared_solver_compatibility(
            &prepared,
            crate::algebra::RealSolver::ConjugateGradient,
            crate::algebra::RealPreconditioner::Jacobi,
        )
        .is_ok());
    }

    #[test]
    fn deserialization_discards_process_local_nullspace_proof() {
        let (mut model, mesh, _, base_region) = thermal_reference_model();
        prescribe_base(&mut model, base_region, 300.0);
        let system = assemble(
            &model,
            &mesh,
            &RealizedExcitations::default(),
            ExcitationSamplePoint::Static,
        )
        .unwrap()
        .steady_operator()
        .unwrap();
        let encoded = serde_json::to_value(system).unwrap();
        assert!(encoded.get("nullspace_evidence").is_none());
        let decoded = serde_json::from_value::<AssembledRealSystem>(encoded).unwrap();
        assert!(decoded.nullspace_evidence().is_none());
    }

    #[test]
    fn deserialization_rejects_caller_asserted_nullspace_proof() {
        let (mut model, mesh, _, base_region) = thermal_reference_model();
        prescribe_base(&mut model, base_region, 300.0);
        let system = assemble(
            &model,
            &mesh,
            &RealizedExcitations::default(),
            ExcitationSamplePoint::Static,
        )
        .unwrap()
        .steady_operator()
        .unwrap();
        let mut encoded = serde_json::to_value(system).unwrap();
        encoded["nullspace_evidence"] = serde_json::json!({
            "proof_identity": Digest::sha256("caller-asserted").to_string()
        });
        assert!(serde_json::from_value::<AssembledRealSystem>(encoded).is_err());
    }

    fn temperature_signal(signal_id: &str, values: [f64; 2]) -> PhysicalSignal {
        PhysicalSignal::Temperature(Signal {
            signal_id: SignalId(signal_id.into()),
            shape: SignalShape::Scalar,
            frame: None,
            representation: SignalRepresentation::Tabulated(TabulatedSignal {
                domain: TabulatedSignalDomain::TimeSeconds,
                knots: [0.0, 1.0]
                    .into_iter()
                    .zip(values)
                    .map(|(abscissa, value)| {
                        SignalKnot::continuous(
                            abscissa,
                            SignalOrdinate::Real(SignalValue::Scalar(Temperature::new(value))),
                        )
                    })
                    .collect(),
                interpolation: SignalInterpolation::Linear,
                extrapolation: SignalExtrapolation::Reject,
            }),
            precision: SignalPrecision::Real64,
            evidence: SignalApproximationEvidence {
                max_absolute_error: Some(0.0),
                max_relative_error: Some(0.0),
                derivative_max_absolute_error: None,
                sample_count: 2,
            },
            provenance: SignalProvenance::Authored,
        })
    }

    fn heat_flux_signal(signal_id: &str, values: [f64; 2]) -> PhysicalSignal {
        PhysicalSignal::HeatFlux(Signal {
            signal_id: SignalId(signal_id.into()),
            shape: SignalShape::Scalar,
            frame: None,
            representation: SignalRepresentation::Tabulated(TabulatedSignal {
                domain: TabulatedSignalDomain::TimeSeconds,
                knots: [0.0, 1.0]
                    .into_iter()
                    .zip(values)
                    .map(|(abscissa, value)| {
                        SignalKnot::continuous(
                            abscissa,
                            SignalOrdinate::Real(SignalValue::Scalar(HeatFlux::new(value))),
                        )
                    })
                    .collect(),
                interpolation: SignalInterpolation::Linear,
                extrapolation: SignalExtrapolation::Reject,
            }),
            precision: SignalPrecision::Real64,
            evidence: SignalApproximationEvidence {
                max_absolute_error: Some(0.0),
                max_relative_error: Some(0.0),
                derivative_max_absolute_error: None,
                sample_count: 2,
            },
            provenance: SignalProvenance::Authored,
        })
    }

    #[test]
    fn time_signals_drive_prescribed_temperature_flux_and_ambient_temperature() {
        let (mut model, mesh, _, base_region) = thermal_reference_model();
        let flux_region = mesh
            .topology
            .boundary_faces
            .iter()
            .flat_map(|face| &face.provenance)
            .find(|region| region.source_topology_id != base_region)
            .expect("reference mesh has a second boundary region")
            .source_topology_id
            .clone();
        model.signals = vec![
            temperature_signal("prescribed_temperature", [300.0, 320.0]),
            heat_flux_signal("boundary_flux", [100.0, 200.0]),
            temperature_signal("ambient_temperature", [280.0, 340.0]),
        ];
        model
            .constraints
            .push(AnalysisConstraint::Thermal(ThermalConstraint {
                constraint_id: "time_prescribed_temperature".into(),
                kind: ThermalConstraintKind::Prescribed {
                    target: RegionTarget::boundary(base_region),
                    value: PrescribedValue::Signal(SignalId("prescribed_temperature".into())),
                },
            }));
        model.boundaries.push(AnalysisBoundary {
            boundary_id: "time_heat_flux".into(),
            target: RegionTarget::boundary(flux_region.clone()),
            operator: BoundaryOperator::ThermalHeatFlux {
                value: PrescribedValue::Signal(SignalId("boundary_flux".into())),
            },
        });

        let at_start = assemble(
            &model,
            &mesh,
            &RealizedExcitations::default(),
            ExcitationSamplePoint::time(0.0),
        )
        .unwrap();
        let at_end = assemble(
            &model,
            &mesh,
            &RealizedExcitations::default(),
            ExcitationSamplePoint::time(1.0),
        )
        .unwrap();
        assert!(at_start
            .prescribed_temperature_k
            .iter()
            .flatten()
            .all(|value| *value == 300.0));
        assert!(at_end
            .prescribed_temperature_k
            .iter()
            .flatten()
            .all(|value| *value == 320.0));
        let start_flux_power = at_start.source.iter().sum::<f64>();
        let end_flux_power = at_end.source.iter().sum::<f64>();
        assert!((end_flux_power - 2.0 * start_flux_power).abs() <= 1.0e-12);

        model.boundaries = vec![AnalysisBoundary {
            boundary_id: "time_convection".into(),
            target: RegionTarget::boundary(flux_region),
            operator: BoundaryOperator::ThermalConvection {
                ambient_temperature: PrescribedValue::Signal(SignalId(
                    "ambient_temperature".into(),
                )),
                coefficient: HeatTransferCoefficient::new(5.0),
            },
        }];
        let cold = assemble(
            &model,
            &mesh,
            &RealizedExcitations::default(),
            ExcitationSamplePoint::time(0.0),
        )
        .unwrap();
        let hot = assemble(
            &model,
            &mesh,
            &RealizedExcitations::default(),
            ExcitationSamplePoint::time(1.0),
        )
        .unwrap();
        let cold_ambient_power = cold.source.iter().sum::<f64>();
        let hot_ambient_power = hot.source.iter().sum::<f64>();
        assert!((hot_ambient_power / cold_ambient_power - 340.0 / 280.0).abs() <= 1.0e-12);
        assert_eq!(cold.conductivity, hot.conductivity);
    }

    #[test]
    fn canonical_tet10_thermal_matrices_pass_constant_patch_and_capacity_integral() {
        let model = fixture_model(FixtureId::CantileverLinearStatic);
        let mesh =
            runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(ElementOrder::Tet10);
        let assembly = assemble(
            &model,
            &mesh,
            &RealizedExcitations::new(Vec::new()),
            ExcitationSamplePoint::Static,
        )
        .unwrap();
        let conductivity_sum = assembly.conductivity.values().iter().sum::<f64>();
        let conductivity_scale = assembly
            .conductivity
            .values()
            .iter()
            .map(|value| value.abs())
            .sum::<f64>();
        assert!(conductivity_sum.abs() <= conductivity_scale * 1.0e-12);

        let material = &model.materials[0];
        let expected_capacity = material.mechanical.density_kg_per_m3
            * material.thermal.specific_heat_j_per_kgk
            * assembly.projection.elements[0].signed_volume_m3;
        let integrated_capacity = assembly.capacity.values().iter().sum::<f64>();
        assert!((integrated_capacity - expected_capacity).abs() <= expected_capacity * 1.0e-12);

        let linear_temperature = assembly
            .projection
            .nodes
            .iter()
            .map(|node| {
                2.0 * node.coordinates_m[0] + 3.0 * node.coordinates_m[1] - node.coordinates_m[2]
            })
            .collect::<Vec<_>>();
        let gradient = assembly
            .recover_element_temperature_gradients(&linear_temperature)
            .unwrap()[0];
        assert!((gradient[0] - 2.0).abs() <= 1.0e-12);
        assert!((gradient[1] - 3.0).abs() <= 1.0e-12);
        assert!((gradient[2] + 1.0).abs() <= 1.0e-12);
        let flux = assembly
            .recover_element_heat_flux(&linear_temperature)
            .unwrap()[0];
        let conductivity = material.thermal.conductivity_w_per_mk;
        assert!((flux[0] + 2.0 * conductivity).abs() <= conductivity * 1.0e-12);
        assert!((flux[1] + 3.0 * conductivity).abs() <= conductivity * 1.0e-12);
        assert!((flux[2] - conductivity).abs() <= conductivity * 1.0e-12);
    }

    #[test]
    fn uniform_heat_generation_is_conserved_and_uniform_temperature_is_preserved() {
        let model = fixture_model(FixtureId::CantileverLinearStatic);
        let mesh =
            runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
        let density = 1200.0;
        let assembly = assemble(
            &model,
            &mesh,
            &heat_generation(density),
            ExcitationSamplePoint::Static,
        )
        .unwrap();
        let expected_source = density * assembly.projection.elements[0].signed_volume_m3;
        let projected_source = assembly.source.iter().sum::<f64>();
        assert!((projected_source - expected_source).abs() <= expected_source * 1.0e-12);
        assert_eq!(assembly.source_evidence.len(), 1);
        assert!(assembly.source_evidence[0].accepted);
        let heated_initial = vec![300.0; assembly.projection.nodes.len()];
        let heated_operator = assembly.transient_operator(0.1, &heated_initial).unwrap();
        let heated = solve_operator_system(
            &heated_operator,
            SpdPreconditionerKind::Jacobi,
            LinearAlgebraBackendKind::CpuReference,
            &CpuReferenceBackend,
            Some(&heated_initial),
        );
        assert!(heated.converged);
        assert!(
            assembly
                .energy_balance(&heated_initial, &heated.solution, 0.1)
                .relative_residual
                <= 1.0e-10
        );

        let no_source = assemble(
            &model,
            &mesh,
            &RealizedExcitations::new(Vec::new()),
            ExcitationSamplePoint::Static,
        )
        .unwrap();
        let initial = vec![315.0; no_source.projection.nodes.len()];
        let operator = no_source.transient_operator(0.25, &initial).unwrap();
        let solved = solve_operator_system(
            &operator,
            SpdPreconditionerKind::Jacobi,
            LinearAlgebraBackendKind::CpuReference,
            &CpuReferenceBackend,
            Some(&initial),
        );
        assert!(solved.converged);
        for temperature in solved.solution {
            assert!((temperature - 315.0).abs() <= 1.0e-9);
        }
    }

    #[test]
    fn exact_face_flux_and_convection_preserve_integrated_power_and_ambient_state() {
        let mut model = fixture_model(FixtureId::CantileverLinearStatic);
        let mesh =
            runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
        let projection = AnalysisProjection::derive(&mesh).unwrap();
        let face = &projection.boundary_faces[0];
        let selector = face.provenance[0].analysis_selection_id().unwrap();
        model.boundaries = vec![AnalysisBoundary {
            boundary_id: "applied_flux".into(),
            target: RegionTarget::boundary(selector.clone()),
            operator: BoundaryOperator::ThermalHeatFlux {
                value: PrescribedValue::Constant(HeatFlux::new(240.0)),
            },
        }];
        let flux = assemble(
            &model,
            &mesh,
            &RealizedExcitations::new(Vec::new()),
            ExcitationSamplePoint::Static,
        )
        .unwrap();
        let expected_power = 240.0 * face.area_m2;
        assert!(
            (flux.source.iter().sum::<f64>() - expected_power).abs() <= expected_power * 1.0e-12
        );

        model.boundaries = vec![AnalysisBoundary {
            boundary_id: "ambient_convection".into(),
            target: RegionTarget::boundary(selector),
            operator: BoundaryOperator::ThermalConvection {
                ambient_temperature: PrescribedValue::Constant(Temperature::new(305.0)),
                coefficient: HeatTransferCoefficient::new(18.0),
            },
        }];
        let convection = assemble(
            &model,
            &mesh,
            &RealizedExcitations::new(Vec::new()),
            ExcitationSamplePoint::Static,
        )
        .unwrap();
        let initial = vec![305.0; convection.projection.nodes.len()];
        let operator = convection.transient_operator(0.25, &initial).unwrap();
        let solved = solve_operator_system(
            &operator,
            SpdPreconditionerKind::Jacobi,
            LinearAlgebraBackendKind::CpuReference,
            &CpuReferenceBackend,
            Some(&initial),
        );
        assert!(solved.converged);
        assert!(solved
            .solution
            .iter()
            .all(|temperature| (temperature - 305.0).abs() <= 1.0e-9));
        assert!(
            convection
                .energy_balance(&initial, &solved.solution, 0.25)
                .relative_residual
                <= 1.0e-10
        );
    }

    #[test]
    fn prescribed_face_temperature_is_enforced_exactly() {
        let mut model = fixture_model(FixtureId::CantileverLinearStatic);
        let mesh =
            runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
        let projection = AnalysisProjection::derive(&mesh).unwrap();
        let face = &projection.boundary_faces[0];
        let selector = face.provenance[0].analysis_selection_id().unwrap();
        model
            .constraints
            .push(AnalysisConstraint::Thermal(ThermalConstraint {
                constraint_id: "fixed_temperature".into(),
                kind: ThermalConstraintKind::Prescribed {
                    target: RegionTarget::boundary(selector),
                    value: PrescribedValue::Constant(Temperature::new(290.0)),
                },
            }));
        let assembly = assemble(
            &model,
            &mesh,
            &RealizedExcitations::new(Vec::new()),
            ExcitationSamplePoint::Static,
        )
        .unwrap();
        let initial = vec![315.0; assembly.projection.nodes.len()];
        let operator = assembly.transient_operator(0.25, &initial).unwrap();
        let solved = solve_operator_system(
            &operator,
            SpdPreconditionerKind::Jacobi,
            LinearAlgebraBackendKind::CpuReference,
            &CpuReferenceBackend,
            Some(&initial),
        );
        assert!(solved.converged);
        for &node in &face.node_indices {
            assert!((solved.solution[node] - 290.0).abs() <= 1.0e-12);
        }
        assert!(
            assembly
                .energy_balance(&initial, &solved.solution, 0.25)
                .relative_residual
                <= 1.0e-10
        );
    }

    #[test]
    fn backward_euler_converges_at_first_order_to_the_closed_form_tetrahedron_mode() {
        let (mut model, mesh, free_node, base_region) = thermal_reference_model();
        prescribe_base(&mut model, base_region, 300.0);
        let assembly = assemble(
            &model,
            &mesh,
            &RealizedExcitations::default(),
            ExcitationSamplePoint::Static,
        )
        .unwrap();
        let final_time_s = 0.2;
        let exact = 300.0 + 100.0 * (-10.0_f64 * final_time_s).exp();
        let mut errors = Vec::new();

        for step_count in [4, 8, 16] {
            let time_step_s = final_time_s / step_count as f64;
            let mut temperature = vec![300.0; assembly.projection.nodes.len()];
            temperature[free_node] = 400.0;
            for _ in 0..step_count {
                let operator = assembly
                    .transient_operator(time_step_s, &temperature)
                    .unwrap();
                let solve = solve_operator_system(
                    &operator,
                    SpdPreconditionerKind::Jacobi,
                    LinearAlgebraBackendKind::CpuReference,
                    &CpuReferenceBackend,
                    Some(&temperature),
                );
                assert!(solve.converged);
                temperature = solve.solution;
            }
            errors.push((temperature[free_node] - exact).abs());
        }

        assert!(errors.windows(2).all(|pair| pair[1] < 0.56 * pair[0]));
        let observed_orders = errors
            .windows(2)
            .map(|pair| (pair[0] / pair[1]).log2())
            .collect::<Vec<_>>();
        assert!(observed_orders.iter().all(|order| *order > 0.8));
    }

    #[test]
    fn steady_convection_matches_the_closed_form_single_free_node_balance() {
        let (mut model, mut mesh, free_node, base_region) = thermal_reference_model();
        prescribe_base(&mut model, base_region, 350.0);
        let convection_region = PersistentEntityId {
            kind: PersistentEntityKind::Face,
            source_topology_id: "convective_sides".into(),
            assembly_path: vec!["root".into()],
        };
        let base_node_ids = mesh.topology.boundary_faces.iter().find_map(|face| {
            face.provenance
                .iter()
                .any(|id| id.source_topology_id == "thermal_base")
                .then(|| face.node_ids.clone())
        });
        let base_node_ids = base_node_ids.unwrap();
        for face in &mut mesh.topology.boundary_faces {
            if !face
                .node_ids
                .iter()
                .all(|node| base_node_ids.contains(node))
            {
                face.provenance.push(convection_region.clone());
                face.provenance.sort();
            }
        }
        mesh.seal_canonical_digest().unwrap();
        let coefficient = 0.2;
        let ambient_temperature_k = 290.0;
        model.boundaries.push(AnalysisBoundary {
            boundary_id: "side_convection".into(),
            target: RegionTarget::boundary(convection_region.source_topology_id),
            operator: BoundaryOperator::ThermalConvection {
                ambient_temperature: PrescribedValue::Constant(Temperature::new(
                    ambient_temperature_k,
                )),
                coefficient: HeatTransferCoefficient::new(coefficient),
            },
        });

        let assembly = assemble(
            &model,
            &mesh,
            &RealizedExcitations::default(),
            ExcitationSamplePoint::Static,
        )
        .unwrap();
        let side_area_m2 = assembly
            .projection
            .boundary_faces
            .iter()
            .filter(|face| {
                face.provenance
                    .iter()
                    .any(|id| id.source_topology_id == "convective_sides")
            })
            .map(|face| face.area_m2)
            .sum::<f64>();
        let conductivity_term = 1.0 / 6.0;
        let convection_diagonal = coefficient * side_area_m2 / 6.0;
        let expected = (conductivity_term * 350.0
            + convection_diagonal * (2.0 * ambient_temperature_k - 350.0))
            / (conductivity_term + convection_diagonal);
        let solve = solve_operator_system(
            &assembly.steady_operator().unwrap(),
            SpdPreconditionerKind::Jacobi,
            LinearAlgebraBackendKind::CpuReference,
            &CpuReferenceBackend,
            None,
        );

        assert!(solve.converged);
        assert!((solve.solution[free_node] - expected).abs() <= 1.0e-10);
    }
}
