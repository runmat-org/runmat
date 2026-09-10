use runmat_meshing_core::SolverMeshArtifact;

use super::{
    dofs::{StructuralDofKind, StructuralDofLayout},
    moment_projection::{project_moment, target_centroid},
    MomentProjectionSummary, SolverMeshRegionMappingError, StructuralLoadKind,
};
use crate::constraints::CanonicalStructuralConstraintPlan;
use crate::excitation::{
    RealizedExcitation, RealizedExcitationKind, RealizedExcitationTarget, RealizedExcitations,
    SpatialIntegrationRule, SpatialProjectionAccumulator, SpatialProjectionEvidence,
};
use crate::topology::AnalysisProjection;

pub(super) fn apply_solver_mesh_structural_loads(
    excitations: &RealizedExcitations,
    mesh: &SolverMeshArtifact,
    constraint_plan: &CanonicalStructuralConstraintPlan,
    layout: &StructuralDofLayout,
    rhs: &mut [f64],
    strict_region_mapping: bool,
) -> Result<StructuralSourceProjection, SolverMeshRegionMappingError> {
    rhs.fill(0.0);
    let projection = AnalysisProjection::derive(mesh).map_err(|error| {
        SolverMeshRegionMappingError::InvalidAnalysisProjection(error.to_string())
    })?;
    let node_coordinates_m = projection
        .nodes
        .iter()
        .map(|node| node.coordinates_m)
        .collect::<Vec<_>>();

    let mut moment_projection = Vec::new();
    let mut evidence = Vec::new();
    for load in excitations.iter() {
        if matches!(load.target, RealizedExcitationTarget::ReferencePoint { .. }) {
            continue;
        }
        if let RealizedExcitationKind::BodyForce(value) = load.kind {
            let (projected_elements, projection_evidence) =
                apply_projected_body_force(&projection, layout, rhs, load, value.real)?;
            if strict_region_mapping && projected_elements == 0 {
                return Err(SolverMeshRegionMappingError::UnmappedLoadRegion {
                    load_id: load.excitation_id.0.clone(),
                    region_id: load
                        .target
                        .solver_region_id()
                        .unwrap_or("<whole_model>")
                        .to_string(),
                    load_kind: StructuralLoadKind::BodyForce,
                });
            }
            if let Some(projection_evidence) = projection_evidence {
                record_spatial_evidence(&mut evidence, projection_evidence)?;
            }
            continue;
        }
        let Some(region_id) = load.target.solver_region_id() else {
            continue;
        };
        let (target_nodes, target_face_count) = projection_region_nodes(&projection, region_id);
        match load.kind {
            RealizedExcitationKind::Force(value) => {
                let [fx, fy, fz] = value.real;
                if target_nodes.is_empty() {
                    if strict_region_mapping {
                        return Err(SolverMeshRegionMappingError::UnmappedLoadRegion {
                            load_id: load.excitation_id.0.clone(),
                            region_id: region_id.to_string(),
                            load_kind: StructuralLoadKind::Force,
                        });
                    }
                    continue;
                }
                let scale = 1.0 / target_nodes.len() as f64;
                let point = target_centroid(&node_coordinates_m, &target_nodes);
                let mut projection_evidence = SpatialProjectionAccumulator::new(
                    load,
                    &projection,
                    SpatialIntegrationRule::UniformResultant,
                    target_face_count,
                    point,
                );
                projection_evidence.add_requested_force(point, value.real);
                for node_index in target_nodes {
                    let nodal_force = [fx * scale, fy * scale, fz * scale];
                    add_rhs(layout, rhs, node_index, StructuralDofKind::Ux, fx * scale);
                    add_rhs(layout, rhs, node_index, StructuralDofKind::Uy, fy * scale);
                    add_rhs(layout, rhs, node_index, StructuralDofKind::Uz, fz * scale);
                    projection_evidence.add_projected_force(node_index, nodal_force);
                }
                record_spatial_evidence(&mut evidence, projection_evidence.finish())?;
            }
            RealizedExcitationKind::Moment(value) => {
                if target_nodes.is_empty() {
                    if strict_region_mapping {
                        return Err(SolverMeshRegionMappingError::UnmappedLoadRegion {
                            load_id: load.excitation_id.0.clone(),
                            region_id: region_id.to_string(),
                            load_kind: StructuralLoadKind::Moment,
                        });
                    }
                    continue;
                }
                let mut summary = add_solver_mesh_moment_projection_rhs(
                    &node_coordinates_m,
                    layout,
                    rhs,
                    &target_nodes,
                    value.real,
                );
                summary.load_id.clone_from(&load.excitation_id.0);
                summary.region_id = region_id.to_string();
                let mut projection_evidence = SpatialProjectionAccumulator::new(
                    load,
                    &projection,
                    SpatialIntegrationRule::UniformResultant,
                    target_face_count,
                    summary.projection_point_m,
                );
                projection_evidence.register_nodes(target_nodes);
                projection_evidence.add_requested_moment(value.real);
                projection_evidence.add_projected_moment(summary.applied_moment);
                record_spatial_evidence(&mut evidence, projection_evidence.finish())?;
                moment_projection.push(summary);
            }
            RealizedExcitationKind::Pressure(value) => {
                if strict_region_mapping
                    && !projection
                        .boundary_faces_matching(region_id)
                        .any(|face| matches!(face.node_indices.len(), 3 | 6))
                {
                    return Err(SolverMeshRegionMappingError::UnmappedLoadRegion {
                        load_id: load.excitation_id.0.clone(),
                        region_id: region_id.to_string(),
                        load_kind: StructuralLoadKind::Pressure,
                    });
                }
                let projection_evidence = apply_projected_pressure(
                    &projection,
                    layout,
                    rhs,
                    load,
                    region_id,
                    value.real,
                )?;
                record_spatial_evidence(&mut evidence, projection_evidence)?;
            }
            RealizedExcitationKind::Traction(value) => {
                if strict_region_mapping
                    && projection
                        .boundary_faces_matching(region_id)
                        .next()
                        .is_none()
                {
                    return Err(SolverMeshRegionMappingError::UnmappedLoadRegion {
                        load_id: load.excitation_id.0.clone(),
                        region_id: region_id.to_string(),
                        load_kind: StructuralLoadKind::Traction,
                    });
                }
                let projection_evidence = apply_projected_traction(
                    &projection,
                    layout,
                    rhs,
                    load,
                    region_id,
                    value.real,
                )?;
                record_spatial_evidence(&mut evidence, projection_evidence)?;
            }
            _ => {}
        }
    }

    for projection_evidence in apply_reference_point_structural_loads(
        excitations,
        constraint_plan,
        &projection,
        layout,
        rhs,
    )? {
        record_spatial_evidence(&mut evidence, projection_evidence)?;
    }

    Ok(StructuralSourceProjection {
        moment_projection,
        evidence,
    })
}

pub(super) struct StructuralSourceProjection {
    pub moment_projection: Vec<MomentProjectionSummary>,
    pub evidence: Vec<SpatialProjectionEvidence>,
}

fn apply_reference_point_structural_loads(
    excitations: &RealizedExcitations,
    plan: &CanonicalStructuralConstraintPlan,
    projection: &AnalysisProjection,
    layout: &StructuralDofLayout,
    rhs: &mut [f64],
) -> Result<Vec<SpatialProjectionEvidence>, SolverMeshRegionMappingError> {
    let mut evidence = Vec::new();
    for load in excitations.iter() {
        let RealizedExcitationTarget::ReferencePoint {
            reference_point_id,
            position,
        } = &load.target
        else {
            continue;
        };
        let point_index = plan
            .reference_points
            .iter()
            .position(|point| point.reference_point_id == *reference_point_id)
            .ok_or_else(|| SolverMeshRegionMappingError::UncoupledReferencePoint {
                load_id: load.excitation_id.0.clone(),
                reference_point_id: reference_point_id.clone(),
            })?;
        if plan.reference_points[point_index].position_meters != position.meters {
            return Err(
                SolverMeshRegionMappingError::ReferencePointPositionMismatch {
                    load_id: load.excitation_id.0.clone(),
                    reference_point_id: reference_point_id.clone(),
                },
            );
        }
        match load.kind {
            RealizedExcitationKind::Force(force) => {
                add_reference_vector(layout, rhs, point_index, force.real, false);
                let mut projection_evidence = SpatialProjectionAccumulator::new(
                    load,
                    projection,
                    SpatialIntegrationRule::ReferencePointGeneralized,
                    1,
                    position.meters,
                );
                projection_evidence.add_requested_force(position.meters, force.real);
                projection_evidence.add_projected_generalized_force(force.real);
                record_spatial_evidence(&mut evidence, projection_evidence.finish())?;
            }
            RealizedExcitationKind::Moment(moment) => {
                add_reference_vector(layout, rhs, point_index, moment.real, true);
                let mut projection_evidence = SpatialProjectionAccumulator::new(
                    load,
                    projection,
                    SpatialIntegrationRule::ReferencePointGeneralized,
                    1,
                    position.meters,
                );
                projection_evidence.add_requested_moment(moment.real);
                projection_evidence.add_projected_moment(moment.real);
                record_spatial_evidence(&mut evidence, projection_evidence.finish())?;
            }
            _ => {
                return Err(
                    SolverMeshRegionMappingError::UnsupportedReferencePointLoad {
                        load_id: load.excitation_id.0.clone(),
                        load_kind: super::structural_inputs::explicit_load_kind(&load.kind),
                    },
                );
            }
        }
    }
    Ok(evidence)
}

fn add_reference_vector(
    layout: &StructuralDofLayout,
    rhs: &mut [f64],
    point_index: usize,
    values: [f64; 3],
    rotational: bool,
) {
    let kinds = if rotational {
        [
            StructuralDofKind::Rx,
            StructuralDofKind::Ry,
            StructuralDofKind::Rz,
        ]
    } else {
        [
            StructuralDofKind::Ux,
            StructuralDofKind::Uy,
            StructuralDofKind::Uz,
        ]
    };
    for (kind, value) in kinds.into_iter().zip(values) {
        if let Some(dof) = layout.reference_point_index(point_index, kind) {
            rhs[dof] += value;
        }
    }
}

fn projection_region_nodes(
    projection: &AnalysisProjection,
    region_id: &str,
) -> (Vec<usize>, usize) {
    let faces = projection
        .boundary_faces_matching(region_id)
        .collect::<Vec<_>>();
    let mut node_indices = faces
        .iter()
        .flat_map(|face| face.node_indices.iter().copied())
        .collect::<Vec<_>>();
    node_indices.sort_unstable();
    node_indices.dedup();
    (node_indices, faces.len())
}

fn apply_projected_pressure(
    projection: &AnalysisProjection,
    layout: &StructuralDofLayout,
    rhs: &mut [f64],
    load: &RealizedExcitation,
    region_id: &str,
    magnitude_pa: f64,
) -> Result<SpatialProjectionEvidence, SolverMeshRegionMappingError> {
    let faces = projection
        .boundary_faces_matching(region_id)
        .collect::<Vec<_>>();
    let reference_point_m = projected_boundary_centroid(&faces);
    let mut evidence = SpatialProjectionAccumulator::new(
        load,
        projection,
        SpatialIntegrationRule::BoundaryDunavantSeven,
        faces.len(),
        reference_point_m,
    );
    for face in faces {
        projection
            .visit_boundary_quadrature(face, |point| {
                let integrated_traction = scale_vec(point.area_vector_weight_m2, magnitude_pa);
                evidence.add_requested_force(point.position_m, integrated_traction);
                for (&node_index, &shape) in face.node_indices.iter().zip(point.shape_values()) {
                    let nodal_force = scale_vec(integrated_traction, shape);
                    add_translational_rhs(layout, rhs, node_index, nodal_force);
                    evidence.add_projected_force(node_index, nodal_force);
                }
            })
            .map_err(|error| {
                SolverMeshRegionMappingError::InvalidAnalysisProjection(error.to_string())
            })?;
    }
    Ok(evidence.finish())
}

fn apply_projected_traction(
    projection: &AnalysisProjection,
    layout: &StructuralDofLayout,
    rhs: &mut [f64],
    load: &RealizedExcitation,
    region_id: &str,
    traction_pa: [f64; 3],
) -> Result<SpatialProjectionEvidence, SolverMeshRegionMappingError> {
    let faces = projection
        .boundary_faces_matching(region_id)
        .collect::<Vec<_>>();
    let reference_point_m = projected_boundary_centroid(&faces);
    let mut evidence = SpatialProjectionAccumulator::new(
        load,
        projection,
        SpatialIntegrationRule::BoundaryDunavantSeven,
        faces.len(),
        reference_point_m,
    );
    for face in faces {
        projection
            .visit_boundary_quadrature(face, |point| {
                let integrated = scale_vec(traction_pa, point.area_weight_m2);
                evidence.add_requested_force(point.position_m, integrated);
                for (&node_index, &shape) in face.node_indices.iter().zip(point.shape_values()) {
                    let nodal_force = scale_vec(integrated, shape);
                    add_translational_rhs(layout, rhs, node_index, nodal_force);
                    evidence.add_projected_force(node_index, nodal_force);
                }
            })
            .map_err(|error| {
                SolverMeshRegionMappingError::InvalidAnalysisProjection(error.to_string())
            })?;
    }
    Ok(evidence.finish())
}

fn apply_projected_body_force(
    projection: &AnalysisProjection,
    layout: &StructuralDofLayout,
    rhs: &mut [f64],
    load: &RealizedExcitation,
    density_n_per_m3: [f64; 3],
) -> Result<(usize, Option<SpatialProjectionEvidence>), SolverMeshRegionMappingError> {
    let elements = projection
        .elements
        .iter()
        .filter(|element| match &load.target {
            RealizedExcitationTarget::WholeModel => true,
            RealizedExcitationTarget::Region(region_id) => element
                .region_id
                .matches_analysis_selection(region_id.as_str()),
            _ => false,
        })
        .collect::<Vec<_>>();
    if elements.is_empty() {
        return Ok((0, None));
    }
    let reference_point_m = projected_volume_centroid(&elements);
    let mut evidence = SpatialProjectionAccumulator::new(
        load,
        projection,
        SpatialIntegrationRule::VolumeKeastFifteen,
        elements.len(),
        reference_point_m,
    );
    for element in &elements {
        projection
            .visit_element_quadrature(element, |point| {
                let integrated = scale_vec(density_n_per_m3, point.volume_weight_m3);
                evidence.add_requested_force(point.position_m, integrated);
                for (&node_index, &shape) in element.node_indices.iter().zip(point.shape_values()) {
                    let nodal_force = scale_vec(integrated, shape);
                    add_translational_rhs(layout, rhs, node_index, nodal_force);
                    evidence.add_projected_force(node_index, nodal_force);
                }
            })
            .map_err(|error| {
                SolverMeshRegionMappingError::InvalidAnalysisProjection(error.to_string())
            })?;
    }
    Ok((elements.len(), Some(evidence.finish())))
}

fn projected_boundary_centroid(faces: &[&crate::topology::ProjectedBoundaryFace]) -> [f64; 3] {
    let mut first_moment = [0.0; 3];
    let mut area = 0.0;
    for face in faces {
        add_scaled(&mut first_moment, face.centroid_m, face.area_m2);
        area += face.area_m2;
    }
    scale_vec(first_moment, 1.0 / area)
}

fn projected_volume_centroid(elements: &[&crate::topology::ProjectedElement]) -> [f64; 3] {
    let mut first_moment = [0.0; 3];
    let mut volume = 0.0;
    for element in elements {
        add_scaled(
            &mut first_moment,
            element.centroid_m,
            element.signed_volume_m3,
        );
        volume += element.signed_volume_m3;
    }
    scale_vec(first_moment, 1.0 / volume)
}

fn add_scaled(target: &mut [f64; 3], value: [f64; 3], scale: f64) {
    for axis in 0..3 {
        target[axis] += value[axis] * scale;
    }
}

fn record_spatial_evidence(
    evidence: &mut Vec<SpatialProjectionEvidence>,
    item: SpatialProjectionEvidence,
) -> Result<(), SolverMeshRegionMappingError> {
    item.validate().map_err(|error| {
        SolverMeshRegionMappingError::InvalidAnalysisProjection(error.to_string())
    })?;
    if !item.accepted {
        let resultant_relative_error = item.resultant_relative_error();
        let moment_relative_error = item.moment_relative_error().unwrap_or(0.0);
        return Err(SolverMeshRegionMappingError::ProjectionConservation {
            load_id: item.excitation_id,
            resultant_relative_error,
            moment_relative_error,
            relative_tolerance: item.requirements.relative_tolerance,
        });
    }
    evidence.push(item);
    Ok(())
}

fn add_solver_mesh_moment_projection_rhs(
    node_coordinates_m: &[[f64; 3]],
    layout: &StructuralDofLayout,
    rhs: &mut [f64],
    target_nodes: &[usize],
    requested_moment: [f64; 3],
) -> MomentProjectionSummary {
    let moment_projection = project_moment(node_coordinates_m, target_nodes, requested_moment);
    for (&node_index, &force) in target_nodes.iter().zip(&moment_projection.nodal_forces) {
        add_translational_rhs(layout, rhs, node_index, force);
    }
    MomentProjectionSummary {
        load_id: String::new(),
        region_id: String::new(),
        target_node_count: target_nodes.len(),
        projection_point_m: moment_projection.point_m,
        applied_force: moment_projection.applied_force,
        applied_moment: moment_projection.applied_moment,
        force_residual: moment_projection.applied_force.map(|value| -value),
        moment_residual: [
            requested_moment[0] - moment_projection.applied_moment[0],
            requested_moment[1] - moment_projection.applied_moment[1],
            requested_moment[2] - moment_projection.applied_moment[2],
        ],
        projection_applied: moment_projection.projection_applied,
    }
}

fn add_rhs(
    layout: &StructuralDofLayout,
    rhs: &mut [f64],
    node_index: usize,
    kind: StructuralDofKind,
    value: f64,
) {
    if let Some(dof) = layout.index(node_index, kind) {
        rhs[dof] += value;
    }
}

fn add_translational_rhs(
    layout: &StructuralDofLayout,
    rhs: &mut [f64],
    node_index: usize,
    force: [f64; 3],
) {
    add_rhs(layout, rhs, node_index, StructuralDofKind::Ux, force[0]);
    add_rhs(layout, rhs, node_index, StructuralDofKind::Uy, force[1]);
    add_rhs(layout, rhs, node_index, StructuralDofKind::Uz, force[2]);
}

fn scale_vec(value: [f64; 3], scale: f64) -> [f64; 3] {
    [value[0] * scale, value[1] * scale, value[2] * scale]
}
