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
use crate::physics::structural::frame::{scale_vector, transform_vector, unit_vector};
use crate::topology::AnalysisProjection;

mod body_force;
mod surface_vector;
pub(super) use body_force::body_load_quadrature_work_upper_bound;
use body_force::{apply_projected_body_force, BodyDensity};
use surface_vector::{apply_projected_force, apply_projected_traction};

#[cfg(test)]
mod tests;

/// Apply structural loads using the caller's admitted analysis projection.
pub(super) fn apply_solver_mesh_structural_loads_on_projection(
    excitations: &RealizedExcitations,
    mesh: &SolverMeshArtifact,
    projection: &AnalysisProjection,
    constraint_plan: &CanonicalStructuralConstraintPlan,
    layout: &StructuralDofLayout,
    rhs: &mut [f64],
    strict_region_mapping: bool,
) -> Result<StructuralSourceProjection, SolverMeshRegionMappingError> {
    rhs.fill(0.0);
    let node_coordinates_m = projection
        .nodes
        .iter()
        .map(|node| node.coordinates_m.canonical_value())
        .collect::<Vec<_>>();

    let mut moment_projection = Vec::new();
    let mut evidence = Vec::new();
    for load in excitations.iter() {
        if matches!(load.target, RealizedExcitationTarget::ReferencePoint { .. }) {
            continue;
        }
        let density = match &load.kind {
            RealizedExcitationKind::BodyForce(value) => Some(BodyDensity::Uniform(value.real)),
            RealizedExcitationKind::BodyForceField(value) => Some(BodyDensity::Nodal(value)),
            _ => None,
        };
        if let Some(density) = density {
            let (projected_elements, projection_evidence) =
                apply_projected_body_force(&projection, layout, rhs, load, density)?;
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
        let RealizedExcitationTarget::Region(target) = &load.target else {
            continue;
        };
        let region_id = &target.region_id;
        let (target_nodes, target_face_count) = projection_region_nodes(&projection, target);
        match load.kind {
            RealizedExcitationKind::Force(value) if target_nodes.is_empty() => {
                // A force on an exact vertex or edge (beam joints and members, shell edges)
                // acts on the mesh nodes carrying that entity: one node per vertex, or a
                // segment-length-consistent line distribution along an edge.
                if let Some(nodal) = entity_nodal_weights(mesh, target) {
                    record_spatial_evidence(
                        &mut evidence,
                        apply_entity_force(&projection, layout, rhs, load, &nodal, value.real)?,
                    )?;
                    continue;
                }
                if strict_region_mapping {
                    return Err(SolverMeshRegionMappingError::UnmappedLoadRegion {
                        load_id: load.excitation_id.0.clone(),
                        region_id: region_id.to_string(),
                        load_kind: StructuralLoadKind::Force,
                    });
                }
            }
            RealizedExcitationKind::Force(value) => {
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
                record_spatial_evidence(
                    &mut evidence,
                    apply_projected_force(&projection, layout, rhs, load, target, value.real)?,
                )?;
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
                let point = target_centroid(&node_coordinates_m, &target_nodes);
                let moment = resolve_vector(
                    load,
                    value.real,
                    point,
                    boundary_normal(&boundary_faces_matching(&projection, target)),
                )?;
                let mut summary = add_solver_mesh_moment_projection_rhs(
                    &node_coordinates_m,
                    layout,
                    rhs,
                    &target_nodes,
                    moment,
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
                projection_evidence.add_requested_moment(moment);
                projection_evidence.add_projected_moment(summary.applied_moment);
                record_spatial_evidence(&mut evidence, projection_evidence.finish())?;
                moment_projection.push(summary);
            }
            RealizedExcitationKind::Pressure(value) => {
                if strict_region_mapping
                    && !boundary_faces_matching(&projection, target)
                        .into_iter()
                        .any(|face| matches!(face.node_indices.len(), 3 | 6))
                {
                    return Err(SolverMeshRegionMappingError::UnmappedLoadRegion {
                        load_id: load.excitation_id.0.clone(),
                        region_id: region_id.to_string(),
                        load_kind: StructuralLoadKind::Pressure,
                    });
                }
                let projection_evidence =
                    apply_projected_pressure(&projection, layout, rhs, load, target, value.real)?;
                record_spatial_evidence(&mut evidence, projection_evidence)?;
            }
            RealizedExcitationKind::Traction(value) => {
                if strict_region_mapping
                    && boundary_faces_matching(&projection, target)
                        .into_iter()
                        .next()
                        .is_none()
                {
                    return Err(SolverMeshRegionMappingError::UnmappedLoadRegion {
                        load_id: load.excitation_id.0.clone(),
                        region_id: region_id.to_string(),
                        load_kind: StructuralLoadKind::Traction,
                    });
                }
                let projection_evidence =
                    apply_projected_traction(&projection, layout, rhs, load, target, value.real)?;
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
        if plan.reference_points[point_index].position_meters != position.canonical_value() {
            return Err(
                SolverMeshRegionMappingError::ReferencePointPositionMismatch {
                    load_id: load.excitation_id.0.clone(),
                    reference_point_id: reference_point_id.clone(),
                },
            );
        }
        match load.kind {
            RealizedExcitationKind::Force(force) => {
                let force = resolve_vector(load, force.real, position.canonical_value(), None)?;
                add_reference_vector(layout, rhs, point_index, force, false);
                let mut projection_evidence = SpatialProjectionAccumulator::new(
                    load,
                    projection,
                    SpatialIntegrationRule::ReferencePointGeneralized,
                    1,
                    position.canonical_value(),
                );
                projection_evidence.add_requested_force(position.canonical_value(), force);
                projection_evidence.add_projected_generalized_force(force);
                record_spatial_evidence(&mut evidence, projection_evidence.finish())?;
            }
            RealizedExcitationKind::Moment(moment) => {
                let moment = resolve_vector(load, moment.real, position.canonical_value(), None)?;
                add_reference_vector(layout, rhs, point_index, moment, true);
                let mut projection_evidence = SpatialProjectionAccumulator::new(
                    load,
                    projection,
                    SpatialIntegrationRule::ReferencePointGeneralized,
                    1,
                    position.canonical_value(),
                );
                projection_evidence.add_requested_moment(moment);
                projection_evidence.add_projected_moment(moment);
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

/// Nodal weights (summing to one) of a target that selects exact vertices or edges, or `None`
/// when no mesh node carries the target's entities.
fn entity_nodal_weights(
    mesh: &SolverMeshArtifact,
    target: &runmat_analysis_core::RegionTarget,
) -> Option<Vec<(usize, f64)>> {
    let index_of = |node_id: u64| {
        mesh.topology
            .nodes
            .binary_search_by_key(&node_id, |node| node.node_id)
            .ok()
    };
    let mut weights = std::collections::BTreeMap::<usize, f64>::new();
    for edge in &mesh.topology.boundary_edges {
        if !edge
            .provenance
            .iter()
            .any(|entity| target.matches_persistent_entity(entity))
        {
            continue;
        }
        let (Some(first), Some(last)) = (index_of(edge.node_ids[0]), index_of(edge.node_ids[1]))
        else {
            continue;
        };
        let a = mesh.topology.nodes[first].coordinates_m.canonical_value();
        let b = mesh.topology.nodes[last].coordinates_m.canonical_value();
        let length = (0..3).map(|k| (b[k] - a[k]).powi(2)).sum::<f64>().sqrt();
        *weights.entry(first).or_default() += 0.5 * length;
        *weights.entry(last).or_default() += 0.5 * length;
    }
    if weights.is_empty() {
        for (index, node) in mesh.topology.nodes.iter().enumerate() {
            if node
                .provenance
                .iter()
                .any(|entity| target.matches_persistent_entity(entity))
            {
                weights.insert(index, 1.0);
            }
        }
    }
    let total = weights.values().sum::<f64>();
    (!weights.is_empty() && total.is_finite() && total > 0.0).then(|| {
        weights
            .into_iter()
            .map(|(node, w)| (node, w / total))
            .collect()
    })
}

fn apply_entity_force(
    projection: &AnalysisProjection,
    layout: &StructuralDofLayout,
    rhs: &mut [f64],
    load: &RealizedExcitation,
    nodal: &[(usize, f64)],
    components: [f64; 3],
) -> Result<SpatialProjectionEvidence, SolverMeshRegionMappingError> {
    let point = nodal.iter().fold([0.0; 3], |mut total, (node, weight)| {
        add_scaled(
            &mut total,
            projection.nodes[*node].coordinates_m.canonical_value(),
            *weight,
        );
        total
    });
    let force = resolve_vector(load, components, point, None)?;
    let mut evidence = SpatialProjectionAccumulator::new(
        load,
        projection,
        SpatialIntegrationRule::UniformResultant,
        nodal.len(),
        point,
    );
    evidence.add_requested_force(point, force);
    for (node, weight) in nodal {
        let nodal_force = scale_vector(force, *weight);
        add_translational_rhs(layout, rhs, *node, nodal_force);
        evidence.add_projected_force(*node, nodal_force);
    }
    Ok(evidence.finish())
}

fn projection_region_nodes(
    projection: &AnalysisProjection,
    target: &runmat_analysis_core::RegionTarget,
) -> (Vec<usize>, usize) {
    let faces = boundary_faces_matching(projection, target);
    let mut node_indices = faces
        .iter()
        .flat_map(|face| face.node_indices.iter().copied())
        .collect::<Vec<_>>();
    node_indices.sort_unstable();
    node_indices.dedup();
    (node_indices, faces.len())
}

fn boundary_faces_matching<'a>(
    projection: &'a AnalysisProjection,
    target: &runmat_analysis_core::RegionTarget,
) -> Vec<&'a crate::topology::ProjectedBoundaryFace> {
    projection
        .boundary_faces
        .iter()
        .filter(|face| {
            face.provenance
                .iter()
                .any(|entity| target.matches_persistent_entity(entity))
        })
        .collect()
}

fn apply_projected_pressure(
    projection: &AnalysisProjection,
    layout: &StructuralDofLayout,
    rhs: &mut [f64],
    load: &RealizedExcitation,
    target: &runmat_analysis_core::RegionTarget,
    magnitude_pa: f64,
) -> Result<SpatialProjectionEvidence, SolverMeshRegionMappingError> {
    let faces = boundary_faces_matching(projection, target);
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
                let integrated_traction = scale_vector(point.area_vector_weight_m2, magnitude_pa);
                evidence.add_requested_force(point.position_m, integrated_traction);
                for (&node_index, &shape) in face.node_indices.iter().zip(point.shape_values()) {
                    let nodal_force = scale_vector(integrated_traction, shape);
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

fn projected_boundary_centroid(faces: &[&crate::topology::ProjectedBoundaryFace]) -> [f64; 3] {
    let mut first_moment = [0.0; 3];
    let mut area = 0.0;
    for face in faces {
        add_scaled(
            &mut first_moment,
            face.centroid_m,
            face.area_m2.canonical_value(),
        );
        area += face.area_m2.canonical_value();
    }
    scale_vector(first_moment, 1.0 / area)
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

fn resolve_vector(
    load: &RealizedExcitation,
    components: [f64; 3],
    point: [f64; 3],
    normal: Option<[f64; 3]>,
) -> Result<[f64; 3], SolverMeshRegionMappingError> {
    transform_vector(&load.policy.frame, components, point, normal).map_err(|error| {
        SolverMeshRegionMappingError::InvalidAnalysisProjection(format!(
            "excitation `{}` coordinate frame is undefined: {error}",
            load.excitation_id.0
        ))
    })
}

fn boundary_normal(faces: &[&crate::topology::ProjectedBoundaryFace]) -> Option<[f64; 3]> {
    unit_vector(faces.iter().fold([0.0; 3], |mut total, face| {
        for (component, value) in total.iter_mut().zip(face.oriented_area_vector_m2) {
            *component += value;
        }
        total
    }))
    .ok()
}
