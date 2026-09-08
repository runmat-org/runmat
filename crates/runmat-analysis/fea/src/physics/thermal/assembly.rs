use std::collections::BTreeMap;

use runmat_analysis_core::{AnalysisModel, MaterialModel};
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
        let rhs = self
            .source
            .iter()
            .zip(capacity_temperature)
            .map(|(source, stored_energy)| source + inverse_time_step * stored_energy)
            .collect::<Vec<_>>();
        let stiffness_diag = diagonal(&stiffness_csr, self.projection.nodes.len());
        OperatorSystem {
            dof_count: self.projection.nodes.len(),
            constrained: vec![false; self.projection.nodes.len()],
            stiffness_dense: None,
            stiffness_csr: Some(stiffness_csr),
            stiffness_diag,
            stiffness_upper: vec![0.0; self.projection.nodes.len().saturating_sub(1)],
            mass_diag: diagonal(&self.capacity, self.projection.nodes.len()),
            damping_diag: vec![0.0; self.projection.nodes.len()],
            rhs,
        }
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

    for element in &projection.elements {
        let material = material_for_element(element, default_material, &materials)?;
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

    Ok(ThermalOperatorAssembly {
        projection,
        conductivity: conductivity.finish(),
        capacity: capacity.finish(),
        source,
        source_evidence,
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

#[cfg(test)]
mod tests {
    use runmat_analysis_core::{
        CoordinateFrame, ExcitationCaseId, ExcitationDistribution, ExcitationId,
        ExcitationKinematics, PhysicalQuantityKind, ProjectionRequirement, SignalId,
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
}
