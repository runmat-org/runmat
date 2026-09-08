use std::collections::BTreeMap;

use runmat_analysis_core::{
    AnalysisConstraint, AnalysisModel, BoundaryOperator, MaterialModel, ThermalConstraintKind,
};
use thiserror::Error;

use crate::{
    excitation::{
        RealizedExcitationKind, RealizedExcitationTarget, RealizedExcitations,
        ScalarProjectionAccumulator, SpatialIntegrationRule, SpatialProjectionEvidence,
    },
    operator::{CsrAssembler, CsrMatrix, OperatorSystem},
    topology::{AnalysisProjection, ProjectedElement},
};

#[derive(Debug, Clone)]
pub(crate) struct ThermalOperatorAssembly {
    pub projection: AnalysisProjection,
    pub conductivity: CsrMatrix,
    pub capacity: CsrMatrix,
    pub source: Vec<f64>,
    pub source_evidence: Vec<SpatialProjectionEvidence>,
    pub prescribed_temperature_k: Vec<Option<f64>>,
    pub element_conductivity_w_per_mk: Vec<f64>,
    pub element_volumetric_capacity_j_per_m3k: Vec<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ThermalEnergyBalance {
    pub storage_rate_w: f64,
    pub boundary_loss_w: f64,
    pub applied_power_w: f64,
    pub prescribed_boundary_power_w: f64,
    pub residual_w: f64,
    pub relative_residual: f64,
}

impl ThermalOperatorAssembly {
    pub(crate) fn transient_operator(
        &self,
        time_step_s: f64,
        previous_temperature_k: &[f64],
    ) -> OperatorSystem {
        let inverse_time_step = time_step_s.recip();
        let mut matrix = CsrAssembler::new(self.projection.nodes.len());
        add_scaled_matrix(&mut matrix, &self.conductivity, 1.0);
        add_scaled_matrix(&mut matrix, &self.capacity, inverse_time_step);
        let stiffness_csr = matrix.finish();
        let capacity_temperature = apply_matrix(&self.capacity, previous_temperature_k);
        let mut rhs = self
            .source
            .iter()
            .zip(capacity_temperature)
            .map(|(source, stored_energy)| source + inverse_time_step * stored_energy)
            .collect::<Vec<_>>();
        apply_prescribed_temperatures(&stiffness_csr, &self.prescribed_temperature_k, &mut rhs);
        let stiffness_diag = diagonal(&stiffness_csr, self.projection.nodes.len());
        OperatorSystem {
            dof_count: self.projection.nodes.len(),
            constrained: self
                .prescribed_temperature_k
                .iter()
                .map(Option::is_some)
                .collect(),
            stiffness_dense: None,
            stiffness_csr: Some(stiffness_csr),
            stiffness_diag,
            stiffness_upper: vec![0.0; self.projection.nodes.len().saturating_sub(1)],
            mass_diag: diagonal(&self.capacity, self.projection.nodes.len()),
            damping_diag: vec![0.0; self.projection.nodes.len()],
            rhs,
        }
    }

    pub(crate) fn recover_element_temperature_gradients(
        &self,
        temperature_k: &[f64],
    ) -> Result<Vec<[f64; 3]>, ThermalAssemblyError> {
        self.projection
            .elements
            .iter()
            .map(|element| {
                let mut integrated = [0.0; 3];
                let mut volume = 0.0;
                self.projection
                    .visit_element_quadrature(element, |point| {
                        for (&node, gradient) in element
                            .node_indices
                            .iter()
                            .zip(point.physical_shape_gradients())
                        {
                            for axis in 0..3 {
                                integrated[axis] +=
                                    temperature_k[node] * gradient[axis] * point.volume_weight_m3;
                            }
                        }
                        volume += point.volume_weight_m3;
                    })
                    .map_err(|error| ThermalAssemblyError::InvalidProjection(error.to_string()))?;
                Ok(integrated.map(|component| component / volume))
            })
            .collect()
    }

    pub(crate) fn recover_element_heat_flux(
        &self,
        temperature_k: &[f64],
    ) -> Result<Vec<[f64; 3]>, ThermalAssemblyError> {
        Ok(self
            .recover_element_temperature_gradients(temperature_k)?
            .into_iter()
            .zip(&self.element_conductivity_w_per_mk)
            .map(|(gradient, conductivity)| gradient.map(|value| -conductivity * value))
            .collect())
    }

    pub(crate) fn energy_balance(
        &self,
        previous_temperature_k: &[f64],
        temperature_k: &[f64],
        time_step_s: f64,
    ) -> ThermalEnergyBalance {
        let rate = previous_temperature_k
            .iter()
            .zip(temperature_k)
            .map(|(previous, current)| (current - previous) / time_step_s)
            .collect::<Vec<_>>();
        let storage_rate = apply_matrix(&self.capacity, &rate);
        let conduction = apply_matrix(&self.conductivity, temperature_k);
        let storage_rate_w = storage_rate.iter().sum::<f64>();
        let boundary_loss_w = conduction.iter().sum::<f64>();
        let applied_power_w = self.source.iter().sum::<f64>();
        let prescribed_boundary_power_w = self
            .prescribed_temperature_k
            .iter()
            .enumerate()
            .filter(|(_, value)| value.is_some())
            .map(|(node, _)| -(storage_rate[node] + conduction[node] - self.source[node]))
            .sum::<f64>();
        let residual_w =
            storage_rate_w + boundary_loss_w - applied_power_w + prescribed_boundary_power_w;
        let scale = storage_rate_w
            .abs()
            .max(boundary_loss_w.abs())
            .max(applied_power_w.abs())
            .max(prescribed_boundary_power_w.abs())
            .max(1.0);
        ThermalEnergyBalance {
            storage_rate_w,
            boundary_loss_w,
            applied_power_w,
            prescribed_boundary_power_w,
            residual_w,
            relative_residual: residual_w.abs() / scale,
        }
    }

    pub(crate) fn recover_element_storage_rate(
        &self,
        previous_temperature_k: &[f64],
        temperature_k: &[f64],
        time_step_s: f64,
    ) -> Result<Vec<f64>, ThermalAssemblyError> {
        self.projection
            .elements
            .iter()
            .enumerate()
            .map(|(element_index, element)| {
                let mut integrated_rate = 0.0;
                let mut volume = 0.0;
                self.projection
                    .visit_element_quadrature(element, |point| {
                        let temperature_rate = element
                            .node_indices
                            .iter()
                            .zip(point.shape_values())
                            .map(|(&node, &shape)| {
                                shape * (temperature_k[node] - previous_temperature_k[node])
                                    / time_step_s
                            })
                            .sum::<f64>();
                        integrated_rate += self.element_volumetric_capacity_j_per_m3k
                            [element_index]
                            * temperature_rate
                            * point.volume_weight_m3;
                        volume += point.volume_weight_m3;
                    })
                    .map_err(|error| ThermalAssemblyError::InvalidProjection(error.to_string()))?;
                Ok(integrated_rate / volume)
            })
            .collect()
    }
}

#[derive(Debug, Error, Clone, PartialEq)]
pub(crate) enum ThermalAssemblyError {
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
    #[error("thermal boundary or constraint {operator_id} requires a constant value for this assembly request")]
    NonConstantValue { operator_id: String },
    #[error("thermal boundary or constraint {operator_id} does not resolve any boundary faces")]
    UnmappedBoundary { operator_id: String },
    #[error("thermal constraints prescribe conflicting temperatures at solver node {node_id}")]
    ConflictingTemperature { node_id: u64 },
}

pub(crate) fn assemble(
    model: &AnalysisModel,
    mesh: &runmat_meshing_core::SolverMeshArtifact,
    excitations: &RealizedExcitations,
) -> Result<ThermalOperatorAssembly, ThermalAssemblyError> {
    let projection = AnalysisProjection::derive(mesh)
        .map_err(|error| ThermalAssemblyError::InvalidProjection(error.to_string()))?;
    let materials = thermal_materials(model);
    let default_material = (model.materials.len() == 1).then(|| &model.materials[0]);
    let mut conductivity = CsrAssembler::new(projection.nodes.len());
    let mut capacity = CsrAssembler::new(projection.nodes.len());
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

    let mut prescribed_temperature_k = vec![None; projection.nodes.len()];
    for constraint in &model.constraints {
        let AnalysisConstraint::Thermal(constraint) = constraint else {
            continue;
        };
        let ThermalConstraintKind::Prescribed { target, value } = &constraint.kind else {
            continue;
        };
        let temperature = value.study_constant(&model.signals).ok_or_else(|| {
            ThermalAssemblyError::NonConstantValue {
                operator_id: constraint.constraint_id.clone(),
            }
        })?;
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
                    if existing != temperature.kelvin {
                        return Err(ThermalAssemblyError::ConflictingTemperature {
                            node_id: projection.nodes[node_index].node_id,
                        });
                    }
                }
                prescribed_temperature_k[node_index] = Some(temperature.kelvin);
            }
        }
    }

    for boundary in &model.boundaries {
        let faces = projection
            .boundary_faces_matching(&boundary.target.region_id)
            .collect::<Vec<_>>();
        match &boundary.operator {
            BoundaryOperator::ThermalHeatFlux { value } => {
                let heat_flux = value.study_constant(&model.signals).ok_or_else(|| {
                    ThermalAssemblyError::NonConstantValue {
                        operator_id: boundary.boundary_id.clone(),
                    }
                })?;
                if faces.is_empty() {
                    return Err(ThermalAssemblyError::UnmappedBoundary {
                        operator_id: boundary.boundary_id.clone(),
                    });
                }
                for face in faces {
                    projection
                        .visit_boundary_quadrature(face, |point| {
                            let power = heat_flux.watts_per_square_meter * point.area_weight_m2;
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
                let ambient = ambient_temperature
                    .study_constant(&model.signals)
                    .ok_or_else(|| ThermalAssemblyError::NonConstantValue {
                        operator_id: boundary.boundary_id.clone(),
                    })?;
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
                                source[row] += heat_transfer
                                    * ambient.kelvin
                                    * row_shape
                                    * point.area_weight_m2;
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
        conductivity: conductivity.finish(),
        capacity: capacity.finish(),
        source,
        source_evidence,
        prescribed_temperature_k,
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

fn add_scaled_matrix(assembler: &mut CsrAssembler, matrix: &CsrMatrix, scale: f64) {
    for row in 0..matrix.row_offsets.len().saturating_sub(1) {
        for entry in matrix.row_offsets[row]..matrix.row_offsets[row + 1] {
            assembler.add(
                row,
                matrix.column_indices[entry],
                scale * matrix.values[entry],
            );
        }
    }
}

fn apply_matrix(matrix: &CsrMatrix, vector: &[f64]) -> Vec<f64> {
    (0..matrix.row_offsets.len().saturating_sub(1))
        .map(|row| {
            (matrix.row_offsets[row]..matrix.row_offsets[row + 1])
                .map(|entry| matrix.values[entry] * vector[matrix.column_indices[entry]])
                .sum()
        })
        .collect()
}

fn diagonal(matrix: &CsrMatrix, size: usize) -> Vec<f64> {
    (0..size)
        .map(|row| {
            (matrix.row_offsets[row]..matrix.row_offsets[row + 1])
                .find_map(|entry| {
                    (matrix.column_indices[entry] == row).then_some(matrix.values[entry])
                })
                .unwrap_or(0.0)
        })
        .collect()
}

fn apply_prescribed_temperatures(matrix: &CsrMatrix, prescribed: &[Option<f64>], rhs: &mut [f64]) {
    for row in 0..rhs.len() {
        if let Some(value) = prescribed[row] {
            rhs[row] = value;
            continue;
        }
        for entry in matrix.row_offsets[row]..matrix.row_offsets[row + 1] {
            if let Some(value) = prescribed[matrix.column_indices[entry]] {
                rhs[row] -= matrix.values[entry] * value;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use runmat_analysis_core::{
        AnalysisBoundary, AnalysisConstraint, CoordinateFrame, ExcitationCaseId,
        ExcitationDistribution, ExcitationId, ExcitationKinematics, HeatFlux,
        HeatTransferCoefficient, PhysicalQuantityKind, PrescribedValue, ProjectionRequirement,
        RegionTarget, SignalId, Temperature, ThermalConstraint, ThermalConstraintKind,
    };
    use runmat_meshing_core::ElementOrder;

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

    #[test]
    fn canonical_tet10_thermal_matrices_pass_constant_patch_and_capacity_integral() {
        let model = fixture_model(FixtureId::CantileverLinearStatic);
        let mesh =
            runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(ElementOrder::Tet10);
        let assembly = assemble(&model, &mesh, &RealizedExcitations::new(Vec::new())).unwrap();
        let conductivity_sum = assembly.conductivity.values.iter().sum::<f64>();
        let conductivity_scale = assembly
            .conductivity
            .values
            .iter()
            .map(|value| value.abs())
            .sum::<f64>();
        assert!(conductivity_sum.abs() <= conductivity_scale * 1.0e-12);

        let material = &model.materials[0];
        let expected_capacity = material.mechanical.density_kg_per_m3
            * material.thermal.specific_heat_j_per_kgk
            * assembly.projection.elements[0].signed_volume_m3;
        let integrated_capacity = assembly.capacity.values.iter().sum::<f64>();
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
        let assembly = assemble(&model, &mesh, &heat_generation(density)).unwrap();
        let expected_source = density * assembly.projection.elements[0].signed_volume_m3;
        let projected_source = assembly.source.iter().sum::<f64>();
        assert!((projected_source - expected_source).abs() <= expected_source * 1.0e-12);
        assert_eq!(assembly.source_evidence.len(), 1);
        assert!(assembly.source_evidence[0].accepted);
        let heated_initial = vec![300.0; assembly.projection.nodes.len()];
        let heated_operator = assembly.transient_operator(0.1, &heated_initial);
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

        let no_source = assemble(&model, &mesh, &RealizedExcitations::new(Vec::new())).unwrap();
        let initial = vec![315.0; no_source.projection.nodes.len()];
        let operator = no_source.transient_operator(0.25, &initial);
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
        let flux = assemble(&model, &mesh, &RealizedExcitations::new(Vec::new())).unwrap();
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
        let convection = assemble(&model, &mesh, &RealizedExcitations::new(Vec::new())).unwrap();
        let initial = vec![305.0; convection.projection.nodes.len()];
        let operator = convection.transient_operator(0.25, &initial);
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
        let assembly = assemble(&model, &mesh, &RealizedExcitations::new(Vec::new())).unwrap();
        let initial = vec![315.0; assembly.projection.nodes.len()];
        let operator = assembly.transient_operator(0.25, &initial);
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
}
