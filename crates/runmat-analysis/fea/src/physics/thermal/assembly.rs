use runmat_analysis_core::{
    AnalysisConstraint, AnalysisModel, BoundaryOperator, PrescribedValue, Temperature,
    ThermalConstraintKind,
};

use crate::{
    algebra::{
        element_assembly::{
            assemble_elements, no_visit, ElementAssemblyError, ElementOutputShape, ElementOutputs,
        },
        spmv, AssembledRealSystem, CanonicalRealOperators, ContributionDomain, ContributionId,
        ContributionProvenance, MatrixProperties, OperatorRole, RealSparseAssembler,
        RealSparseMatrix,
    },
    excitation::{
        sample_prescribed_heat_flux, sample_prescribed_temperature,
        sample_prescribed_temperature_difference, ExcitationSamplePoint, RealizedExcitationKind,
        RealizedExcitationTarget, RealizedExcitations, ScalarProjectionAccumulator,
        SpatialIntegrationRule, SpatialProjectionEvidence,
    },
    physics::{
        scalar_constraints::{self, ScalarNodeRelation, ScalarPairing},
        thermal_interface::{self, InterfaceDofs},
    },
    topology::{AnalysisProjection, ProjectedElement},
};

use super::{constitutive_stats, ThermalConstitutiveStats};

#[cfg(test)]
mod convergence_tests;
#[cfg(test)]
mod element_density_tests;
mod energy_recovery;
mod error;
pub(crate) use error::ThermalAssemblyError;
mod nonlinear;
mod recovery;
mod steady;
mod steady_reference;
mod temperature_field;
mod volume_source;

#[derive(Debug, Clone)]
pub(crate) struct ThermalOperatorAssembly {
    pub property_identity: crate::properties::CompiledPropertyMapIdentity,
    pub constitutive_stats: ThermalConstitutiveStats,
    pub projection: AnalysisProjection,
    pub conductivity: RealSparseMatrix,
    pub capacity: RealSparseMatrix,
    /// Applied generation/flux only; excludes the algebraic Robin ambient forcing.
    pub applied_source:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Power, Vec<f64>>,
    pub source_evidence: Vec<SpatialProjectionEvidence>,
    pub prescribed_temperature_k: runmat_types::physical::Quantity<
        runmat_types::physical::kind::Temperature,
        Vec<Option<f64>>,
    >,
    /// Tie, periodic and affine-jump relations: each secondary node follows its primaries.
    pub relations: Vec<ScalarNodeRelation>,
    /// Projection nodes: the mesh nodes, then one side-owned node per conductance-interface node.
    pub dofs: InterfaceDofs,
    pub positive_robin_weight: runmat_types::physical::Quantity<
        runmat_types::physical::kind::ThermalConductance,
        Vec<f64>,
    >,
    pub element_conductivity_w_per_mk: runmat_types::physical::Quantity<
        runmat_types::physical::kind::ThermalConductivity,
        Vec<f64>,
    >,
    pub element_volumetric_capacity_j_per_m3k: runmat_types::physical::Quantity<
        runmat_types::physical::kind::VolumetricHeatCapacity,
        Vec<f64>,
    >,
}

impl ThermalOperatorAssembly {
    pub(crate) fn transient_operator(
        &self,
        model: &AnalysisModel,
        sample: ExcitationSamplePoint<'_>,
        time_step_s: f64,
        previous_temperature_k: &[f64],
        reference_temperature_k: &[f64],
    ) -> Result<
        crate::algebra::PhysicalScalarSystem<runmat_types::physical::kind::TemperatureDifference>,
        ThermalAssemblyError,
    > {
        self.reject_radiation_in_linear_route(model)?;
        if !time_step_s.is_finite()
            || time_step_s <= 0.0
            || previous_temperature_k.len() != self.projection.nodes.len()
            || previous_temperature_k
                .iter()
                .any(|value| !value.is_finite())
        {
            return Err(ThermalAssemblyError::InvalidProjection(
                "transient thermal coordinates require a positive finite time step and one finite departure per node".into(),
            ));
        }
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
        let mut rhs = self.relative_source(model, sample, reference_temperature_k)?;
        for (source, stored_energy) in rhs.iter_mut().zip(capacity_temperature) {
            *source += inverse_time_step * stored_energy;
        }
        for (node, value) in self
            .prescribed_temperature_k
            .canonical_values()
            .iter()
            .enumerate()
        {
            if let Some(value) = value {
                rhs[node] = value - reference_temperature_k[node];
            }
        }
        let constrained = self
            .prescribed_temperature_k
            .canonical_values()
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
            .and_then(|numerical| {
                crate::algebra::PhysicalScalarSystem::try_new(
                    numerical,
                    runmat_types::physical::DynamicDimension::quantity_type(
                        runmat_types::physical::QuantityKind::Power,
                    )
                    .with_semantic_kind(Some(runmat_types::physical::QuantityKind::Power))
                    .expect("thermal equation"),
                )
                .map_err(|error| ThermalAssemblyError::SparseOperator(error.into()))
            })
    }
}

/// The caller owns one property map compiled from its immutable authored model for the run.
/// Constraints/load signals may change per step/sample; material ownership is not reselected.
pub(crate) fn assemble(
    model: &AnalysisModel,
    mesh: &runmat_meshing_core::SolverMeshArtifact,
    properties: &crate::properties::CompiledPropertyMap,
    support: &crate::algebra::SparsePattern,
    excitations: &RealizedExcitations,
    sample: ExcitationSamplePoint<'_>,
) -> Result<ThermalOperatorAssembly, ThermalAssemblyError> {
    let projection = AnalysisProjection::derive(mesh)
        .map_err(|error| ThermalAssemblyError::InvalidProjection(error.to_string()))?;
    // A finite-conductance interface splits its nodes: the second side's elements and faces use
    // their own side-owned nodes, joined to the first side by the interface conductance.
    let (interfaces, dofs) = thermal_interface::solid_conduction(&projection, &model.interfaces)
        .map_err(ThermalAssemblyError::InvalidProjection)?;
    let projection = dofs.split_projection(&projection, &interfaces);
    if properties.identity.mesh_digest != Some(mesh.canonical_digest)
        || properties.identity.geometry_id != model.geometry_id
        || properties.identity.geometry_revision != model.geometry_revision
    {
        return Err(ThermalAssemblyError::InvalidMaterialCoverage(
            "prepared thermal properties do not match model geometry and solver topology".into(),
        ));
    }
    if support.rows() != projection.nodes.len() || support.columns() != projection.nodes.len() {
        return Err(ThermalAssemblyError::SparseOperator(
            "thermal planned support shape mismatch".into(),
        ));
    }
    volume_source::validate_fields(mesh, &projection, excitations)?;
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
            .map_err(|error| ThermalAssemblyError::InvalidMaterialCoverage(error.to_string()))?
            .canonical_value();
        element_conductivity_w_per_mk.push(thermal.conductivity_w_per_mk.canonical_value());
        element_specific_heat_j_per_kgk.push(thermal.specific_heat_j_per_kgk.canonical_value());
        element_density_kg_per_m3.push(mass_density_kg_per_m3);
        element_volumetric_capacity_j_per_m3k
            .push(mass_density_kg_per_m3 * thermal.specific_heat_j_per_kgk.canonical_value());
    }
    // Element conduction/capacity kernels run on the admitted CPU pool and scatter onto the
    // planned support in canonical element order (thread-count independent values).
    let element_map = crate::topology::projection_element_map(&projection)
        .map_err(|error| ThermalAssemblyError::SparseOperator(error.to_string()))?;
    let assembled = assemble_elements(
        &element_map,
        support,
        ElementOutputShape {
            matrices: 2,
            ..Default::default()
        },
        crate::planning::resource::LocalExecutionPlan::serial(),
        |index, _, out: &mut ElementOutputs<'_, f64>| {
            let element = &projection.elements[index];
            let n = element.node_indices.len();
            let conductivity = element_conductivity_w_per_mk[index];
            let capacity = element_volumetric_capacity_j_per_m3k[index];
            let mut local_conductivity = vec![0.0; n * n];
            let mut local_capacity = vec![0.0; n * n];
            projection
                .visit_element_quadrature(element, |point| {
                    let gradients = point.physical_shape_gradients();
                    let shapes = point.shape_values();
                    for row in 0..n {
                        for column in 0..n {
                            let gradient_product = gradients[row]
                                .iter()
                                .zip(gradients[column])
                                .map(|(left, right)| left * right)
                                .sum::<f64>();
                            local_conductivity[row * n + column] +=
                                conductivity * gradient_product * point.volume_weight_m3;
                            local_capacity[row * n + column] +=
                                capacity * shapes[row] * shapes[column] * point.volume_weight_m3;
                        }
                    }
                })
                .map_err(|error| ThermalAssemblyError::InvalidProjection(error.to_string()))?;
            out.matrix(0).copy_from_slice(&local_conductivity);
            out.matrix(1).copy_from_slice(&local_capacity);
            Ok(())
        },
        no_visit,
    )
    .map_err(|error| match error {
        ElementAssemblyError::Element { error, .. } => error,
        other => ThermalAssemblyError::SparseOperator(other.to_string()),
    })?;
    drop(element_map);
    let mut matrices = assembled.matrices.into_iter();
    let mut conductivity = RealSparseAssembler::from_planned_values(
        support
            .try_clone()
            .map_err(|error| ThermalAssemblyError::SparseOperator(error.to_string()))?,
        matrices.next().unwrap_or_default(),
        OperatorRole::Conductivity,
        MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
    )
    .map_err(|error| ThermalAssemblyError::SparseOperator(error.to_string()))?;
    let mut capacity = RealSparseAssembler::from_planned_values(
        support
            .try_clone()
            .map_err(|error| ThermalAssemblyError::SparseOperator(error.to_string()))?,
        matrices.next().unwrap_or_default(),
        OperatorRole::Capacity,
        MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
    )
    .map_err(|error| ThermalAssemblyError::SparseOperator(error.to_string()))?;
    conductivity.add_contribution(ContributionProvenance {
        domain: ContributionDomain::Thermal,
        contribution_id: ContributionId("conductive_element_and_boundary_terms".to_string()),
    });
    thermal_interface::add_conductance(&projection, &interfaces, &dofs, |row, column, value| {
        conductivity.add(row, column, value)
    })
    .map_err(ThermalAssemblyError::InvalidProjection)?;
    capacity.add_contribution(ContributionProvenance {
        domain: ContributionDomain::Thermal,
        contribution_id: ContributionId("consistent_heat_capacity".to_string()),
    });

    let mut applied_source = vec![0.0; projection.nodes.len()];
    let mut source_evidence = Vec::new();
    for excitation in excitations.iter() {
        let (real, imaginary, rule) = match excitation.kind {
            RealizedExcitationKind::HeatGeneration(value) => (
                value.real,
                value.imaginary,
                SpatialIntegrationRule::VolumeKeastFifteen,
            ),
            RealizedExcitationKind::HeatGenerationField(_) => {
                (0.0, 0.0, SpatialIntegrationRule::VolumeKeastFifteen)
            }
            RealizedExcitationKind::HeatFlux(value) => (
                value.real,
                value.imaginary,
                SpatialIntegrationRule::BoundaryDunavantSeven,
            ),
            _ => continue,
        };
        if imaginary != 0.0 {
            return Err(ThermalAssemblyError::ComplexSource {
                excitation_id: excitation.excitation_id.0.clone(),
            });
        }
        let entity_count = match excitation.kind {
            RealizedExcitationKind::HeatGeneration(_)
            | RealizedExcitationKind::HeatGenerationField(_) => projection
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
                    RealizedExcitationKind::HeatGeneration(_)
                    | RealizedExcitationKind::HeatGenerationField(_) => "volume elements",
                    _ => "boundary faces",
                },
            });
        }
        let mut evidence =
            ScalarProjectionAccumulator::new(excitation, &projection, rule, entity_count)
                .map_err(|error| ThermalAssemblyError::InvalidProjection(error.to_string()))?;
        match excitation.kind {
            RealizedExcitationKind::HeatGeneration(_)
            | RealizedExcitationKind::HeatGenerationField(_) => {
                volume_source::integrate(
                    &projection,
                    excitation,
                    real,
                    &mut applied_source,
                    &mut evidence,
                )?;
            }
            RealizedExcitationKind::HeatFlux(_) => {
                for face in boundary_faces_for_target(&projection, &excitation.target) {
                    projection
                        .visit_boundary_quadrature(face, |point| {
                            let integrated = real * point.area_weight_m2;
                            evidence.add_requested(integrated, 0.0);
                            for (&node, &shape) in
                                face.node_indices.iter().zip(point.shape_values())
                            {
                                let nodal = integrated * shape;
                                applied_source[node] += nodal;
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
    let identity = runmat_analysis_core::PeriodicTransform::Identity;
    let mut pairings = Vec::new();
    for constraint in &model.constraints {
        let AnalysisConstraint::Thermal(constraint) = constraint else {
            continue;
        };
        let (target, value) = match &constraint.kind {
            ThermalConstraintKind::Prescribed { target, value } => (target, value),
            ThermalConstraintKind::Tie(pair) => {
                pairings.push(ScalarPairing {
                    constraint_id: &constraint.constraint_id,
                    pair,
                    transform: &identity,
                    offset: 0.0,
                });
                continue;
            }
            ThermalConstraintKind::Periodic(periodic) => {
                pairings.push(ScalarPairing {
                    constraint_id: &constraint.constraint_id,
                    pair: &periodic.pair,
                    transform: &periodic.transform,
                    offset: 0.0,
                });
                continue;
            }
            ThermalConstraintKind::AffineJump {
                periodic,
                temperature_jump,
            } => {
                let jump = sample_prescribed_temperature_difference(
                    temperature_jump,
                    &model.signals,
                    sample,
                )
                .map_err(|error| ThermalAssemblyError::SignalRealization {
                    operator_id: constraint.constraint_id.clone(),
                    message: error.to_string(),
                })?;
                if jump.imaginary != 0.0 || !jump.real.is_finite() {
                    return Err(ThermalAssemblyError::ComplexBoundary {
                        operator_id: constraint.constraint_id.clone(),
                    });
                }
                pairings.push(ScalarPairing {
                    constraint_id: &constraint.constraint_id,
                    pair: &periodic.pair,
                    transform: &periodic.transform,
                    offset: jump.real,
                });
                continue;
            }
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
        let mut faces = projection.boundary_faces_for_target(target).peekable();
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

    let relations = scalar_constraints::relations(mesh, &projection, &pairings)
        .map_err(ThermalAssemblyError::InvalidProjection)?;
    // Ties and periodic pairs resolve mesh nodes; a node split by a conductance interface has
    // two temperatures, and the pairing does not say which side it means.
    if relations.iter().any(|relation| {
        std::iter::once(relation.secondary)
            .chain(relation.primaries.iter().map(|&(node, _)| node))
            .any(|node| {
                dofs.second()
                    .binary_search_by_key(&node, |&(split, _)| split)
                    .is_ok()
            })
    }) {
        return Err(ThermalAssemblyError::InvalidProjection(
            "a tie, periodic or jump relation reaches a node split by a conductance interface; pair faces away from the interface".into(),
        ));
    }
    if let Some(relation) = relations
        .iter()
        .find(|relation| prescribed_temperature_k[relation.secondary].is_some())
    {
        return Err(ThermalAssemblyError::InvalidProjection(format!(
            "solver node {} is both prescribed and the secondary of a tie, periodic or jump relation",
            projection.nodes[relation.secondary].node_id
        )));
    }

    for boundary in &model.boundaries {
        let mut faces = projection
            .boundary_faces_for_target(&boundary.target)
            .peekable();
        match &boundary.operator {
            BoundaryOperator::ThermalRadiation {
                ambient_temperature,
                emissivity,
            } => {
                let ambient = sample_ambient_temperature(
                    model,
                    &boundary.boundary_id,
                    ambient_temperature,
                    sample,
                )?;
                super::radiation::outward_flux_and_tangent(ambient, ambient, *emissivity)
                    .map_err(ThermalAssemblyError::Radiation)?;
                if faces.peek().is_none() {
                    return Err(ThermalAssemblyError::UnmappedBoundary {
                        operator_id: boundary.boundary_id.clone(),
                    });
                }
            }
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
                                applied_source[node] += power * shape;
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
                sample_ambient_temperature(
                    model,
                    &boundary.boundary_id,
                    ambient_temperature,
                    sample,
                )?;
                if faces.peek().is_none() {
                    return Err(ThermalAssemblyError::UnmappedBoundary {
                        operator_id: boundary.boundary_id.clone(),
                    });
                }
                let heat_transfer = coefficient.canonical_value();
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
        applied_source:
            runmat_types::physical::Quantity::<runmat_types::physical::kind::Power, _>::new(
                applied_source,
            ),
        source_evidence,
        prescribed_temperature_k: runmat_types::physical::Quantity::new(prescribed_temperature_k),
        relations,
        dofs,
        positive_robin_weight: runmat_types::physical::Quantity::new(positive_robin_weight),
        element_conductivity_w_per_mk: runmat_types::physical::Quantity::new(
            element_conductivity_w_per_mk,
        ),
        element_volumetric_capacity_j_per_m3k: runmat_types::physical::Quantity::new(
            element_volumetric_capacity_j_per_m3k,
        ),
    })
}

/// Shared by operator assembly and physical recovery, including signal side/approach.
/// An `ambient_temperature` excitation on the boundary's faces supplies the ambient (for
/// example a chamber temperature that varies in time) in place of the authored value.
fn sample_ambient_temperature(
    model: &AnalysisModel,
    boundary_id: &str,
    value: &PrescribedValue<Temperature>,
    sample: ExcitationSamplePoint<'_>,
) -> Result<f64, ThermalAssemblyError> {
    let excited = model
        .boundaries
        .iter()
        .find(|boundary| boundary.boundary_id == boundary_id)
        .and_then(|boundary| ambient_excitation(model, &boundary.target));
    let value = excited.as_ref().unwrap_or(value);
    let ambient =
        sample_prescribed_temperature(value, &model.signals, sample).map_err(|error| {
            ThermalAssemblyError::SignalRealization {
                operator_id: boundary_id.into(),
                message: error.to_string(),
            }
        })?;
    if ambient.imaginary != 0.0 {
        return Err(ThermalAssemblyError::ComplexBoundary {
            operator_id: boundary_id.into(),
        });
    }
    Ok(ambient.real)
}

fn volume_matches(element: &ProjectedElement, target: &RealizedExcitationTarget) -> bool {
    match target {
        RealizedExcitationTarget::WholeModel => true,
        RealizedExcitationTarget::Region(target) => {
            target.selects_volume_element(&element.region_id, &element.provenance)
        }
        _ => false,
    }
}

fn boundary_faces_for_target<'a>(
    projection: &'a AnalysisProjection,
    target: &'a RealizedExcitationTarget,
) -> impl Iterator<Item = &'a crate::topology::ProjectedBoundaryFace> + 'a {
    let region = match target {
        RealizedExcitationTarget::Region(target) => Some(target),
        _ => None,
    };
    region
        .into_iter()
        .flat_map(move |target| projection.boundary_faces_for_target(target))
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

#[cfg(test)]
mod tests;

/// The ambient supplied by an `ambient_temperature` excitation on exactly these faces.
pub(crate) fn ambient_excitation(
    model: &AnalysisModel,
    target: &runmat_analysis_core::RegionTarget,
) -> Option<PrescribedValue<Temperature>> {
    model.excitations.iter().find_map(|action| match action {
        runmat_analysis_core::ExcitationAction::AmbientTemperature(excitation) => {
            match &excitation.target {
                runmat_analysis_core::ExcitationTarget::Region(region)
                    if region.region_id == target.region_id =>
                {
                    Some(PrescribedValue::Signal(excitation.signal_id.clone()))
                }
                _ => None,
            }
        }
        _ => None,
    })
}
