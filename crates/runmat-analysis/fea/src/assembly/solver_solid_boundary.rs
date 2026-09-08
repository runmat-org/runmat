use runmat_meshing_core::SolverMeshArtifact;

use super::{
    dofs::{StructuralDofKind, StructuralDofLayout},
    MomentProjectionSummary, SolverMeshRegionMappingError, StructuralLoadKind,
};
use crate::constraints::CanonicalStructuralConstraintPlan;
use crate::excitation::{RealizedExcitationKind, RealizedExcitationTarget, RealizedExcitations};

pub(super) fn apply_solver_mesh_structural_loads(
    excitations: &RealizedExcitations,
    mesh: &SolverMeshArtifact,
    layout: &StructuralDofLayout,
    rhs: &mut [f64],
    strict_region_mapping: bool,
) -> Result<Vec<MomentProjectionSummary>, SolverMeshRegionMappingError> {
    rhs.fill(0.0);

    let mut moment_projection = Vec::new();
    for load in excitations.iter() {
        if matches!(load.target, RealizedExcitationTarget::ReferencePoint { .. }) {
            continue;
        }
        let Some(region_id) = load.target.solver_region_id() else {
            continue;
        };
        let target_nodes = solver_mesh_region_node_indices(mesh, region_id);
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
                for node_index in target_nodes {
                    add_rhs(layout, rhs, node_index, StructuralDofKind::Ux, fx * scale);
                    add_rhs(layout, rhs, node_index, StructuralDofKind::Uy, fy * scale);
                    add_rhs(layout, rhs, node_index, StructuralDofKind::Uz, fz * scale);
                }
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
                let summary = add_solver_mesh_moment_projection_rhs(
                    mesh,
                    layout,
                    rhs,
                    &target_nodes,
                    value.real,
                );
                moment_projection.push(MomentProjectionSummary {
                    load_id: load.excitation_id.0.clone(),
                    region_id: region_id.to_string(),
                    ..summary
                });
            }
            RealizedExcitationKind::Pressure(value) => {
                if strict_region_mapping
                    && !mesh.topology.boundary_faces.iter().any(|face| {
                        boundary_face_matches_region(face, region_id)
                            && matches!(face.node_ids.len(), 3 | 6)
                    })
                {
                    return Err(SolverMeshRegionMappingError::UnmappedLoadRegion {
                        load_id: load.excitation_id.0.clone(),
                        region_id: region_id.to_string(),
                        load_kind: StructuralLoadKind::Pressure,
                    });
                }
                apply_solver_mesh_pressure(mesh, layout, rhs, region_id, value.real);
            }
            RealizedExcitationKind::BodyForce(value) => {
                let [gx, gy, gz] = value.real;
                let all_nodes = (0..mesh.topology.nodes.len()).collect::<Vec<_>>();
                let scale = 1.0 / all_nodes.len().max(1) as f64;
                for node_index in all_nodes {
                    add_rhs(layout, rhs, node_index, StructuralDofKind::Ux, gx * scale);
                    add_rhs(layout, rhs, node_index, StructuralDofKind::Uy, gy * scale);
                    add_rhs(layout, rhs, node_index, StructuralDofKind::Uz, gz * scale);
                }
            }
            _ => {}
        }
    }

    Ok(moment_projection)
}

pub(super) fn apply_reference_point_structural_loads(
    excitations: &RealizedExcitations,
    plan: &CanonicalStructuralConstraintPlan,
    layout: &StructuralDofLayout,
    rhs: &mut [f64],
) -> Result<(), SolverMeshRegionMappingError> {
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
                add_reference_vector(layout, rhs, point_index, force.real, false)
            }
            RealizedExcitationKind::Moment(moment) => {
                add_reference_vector(layout, rhs, point_index, moment.real, true)
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
    Ok(())
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

fn solver_mesh_region_node_indices(mesh: &SolverMeshArtifact, region_id: &str) -> Vec<usize> {
    let mut node_ids = Vec::<u64>::new();
    for face in &mesh.topology.boundary_faces {
        if !boundary_face_matches_region(face, region_id) {
            continue;
        }
        for node_id in &face.node_ids {
            if !node_ids.contains(node_id) {
                node_ids.push(*node_id);
            }
        }
    }
    node_ids
        .into_iter()
        .filter_map(|node_id| {
            mesh.topology
                .nodes
                .iter()
                .position(|node| node.node_id == node_id)
        })
        .collect()
}

fn boundary_face_matches_region(
    face: &runmat_meshing_core::SolverBoundaryFace,
    region_id: &str,
) -> bool {
    face.provenance
        .iter()
        .any(|entity| entity.matches_analysis_selection(region_id))
}

fn apply_solver_mesh_pressure(
    mesh: &SolverMeshArtifact,
    layout: &StructuralDofLayout,
    rhs: &mut [f64],
    region_id: &str,
    magnitude_pa: f64,
) {
    for face in &mesh.topology.boundary_faces {
        if !boundary_face_matches_region(face, region_id) || !matches!(face.node_ids.len(), 3 | 6) {
            continue;
        }
        let Some(indices) = solver_mesh_face_node_indices(mesh, &face.node_ids) else {
            continue;
        };
        let coordinates = [
            mesh.topology.nodes[indices[0]].coordinates_m,
            mesh.topology.nodes[indices[1]].coordinates_m,
            mesh.topology.nodes[indices[2]].coordinates_m,
        ];
        let force = scale_vec(triangle_area_normal(coordinates), magnitude_pa);
        let nodal_force = scale_vec(force, 1.0 / 3.0);
        for node_index in indices {
            add_translational_rhs(layout, rhs, node_index, nodal_force);
        }
    }
}

fn add_solver_mesh_moment_projection_rhs(
    mesh: &SolverMeshArtifact,
    layout: &StructuralDofLayout,
    rhs: &mut [f64],
    target_nodes: &[usize],
    requested_moment: [f64; 3],
) -> MomentProjectionSummary {
    if target_nodes.is_empty() {
        return MomentProjectionSummary {
            load_id: String::new(),
            region_id: String::new(),
            target_node_count: 0,
            projection_point_m: [0.0; 3],
            applied_force: [0.0; 3],
            applied_moment: [0.0; 3],
            force_residual: [0.0; 3],
            moment_residual: requested_moment,
            projection_applied: false,
        };
    }

    let centroid = solver_mesh_target_centroid(mesh, target_nodes);
    let mut nodal_forces = vec![[0.0; 3]; target_nodes.len()];
    let mut projection_applied = false;

    if !requested_moment
        .iter()
        .all(|component| component.abs() <= f64::EPSILON)
    {
        let mut coupling = [[0.0_f64; 3]; 3];
        let offsets = target_nodes
            .iter()
            .map(|&node_index| {
                let node = mesh.topology.nodes[node_index].coordinates_m;
                [
                    node[0] - centroid[0],
                    node[1] - centroid[1],
                    node[2] - centroid[2],
                ]
            })
            .collect::<Vec<_>>();
        for offset in &offsets {
            let r2 = dot(*offset, *offset);
            for row in 0..3 {
                coupling[row][row] += r2;
                for col in 0..3 {
                    coupling[row][col] -= offset[row] * offset[col];
                }
            }
        }

        if let Some(inv) = invert_3x3(coupling) {
            let lambda = mat_vec(inv, requested_moment);
            for ((&node_index, offset), nodal_force) in target_nodes
                .iter()
                .zip(offsets.iter())
                .zip(nodal_forces.iter_mut())
            {
                let couple_force = cross(lambda, *offset);
                add_translational_rhs(layout, rhs, node_index, couple_force);
                nodal_force[0] += couple_force[0];
                nodal_force[1] += couple_force[1];
                nodal_force[2] += couple_force[2];
            }
            projection_applied = true;
        }
    }

    let (applied_force, applied_moment) =
        solver_mesh_projected_resultants(mesh, target_nodes, &nodal_forces, centroid);
    MomentProjectionSummary {
        load_id: String::new(),
        region_id: String::new(),
        target_node_count: target_nodes.len(),
        projection_point_m: centroid,
        applied_force,
        applied_moment,
        force_residual: applied_force.map(|value| -value),
        moment_residual: [
            requested_moment[0] - applied_moment[0],
            requested_moment[1] - applied_moment[1],
            requested_moment[2] - applied_moment[2],
        ],
        projection_applied,
    }
}

fn solver_mesh_face_node_indices(
    mesh: &SolverMeshArtifact,
    node_ids: &[u64],
) -> Option<[usize; 3]> {
    if !matches!(node_ids.len(), 3 | 6) {
        return None;
    }
    Some([
        mesh.topology
            .nodes
            .iter()
            .position(|node| node.node_id == node_ids[0])?,
        mesh.topology
            .nodes
            .iter()
            .position(|node| node.node_id == node_ids[1])?,
        mesh.topology
            .nodes
            .iter()
            .position(|node| node.node_id == node_ids[2])?,
    ])
}

fn solver_mesh_target_centroid(mesh: &SolverMeshArtifact, target_nodes: &[usize]) -> [f64; 3] {
    let mut centroid = [0.0_f64; 3];
    for &node_index in target_nodes {
        let node = mesh.topology.nodes[node_index].coordinates_m;
        centroid[0] += node[0];
        centroid[1] += node[1];
        centroid[2] += node[2];
    }
    scale_vec(centroid, 1.0 / target_nodes.len() as f64)
}

fn solver_mesh_projected_resultants(
    mesh: &SolverMeshArtifact,
    target_nodes: &[usize],
    nodal_forces: &[[f64; 3]],
    point_m: [f64; 3],
) -> ([f64; 3], [f64; 3]) {
    let mut applied_force = [0.0_f64; 3];
    let mut applied_moment = [0.0_f64; 3];
    for (&node_index, &force) in target_nodes.iter().zip(nodal_forces.iter()) {
        applied_force[0] += force[0];
        applied_force[1] += force[1];
        applied_force[2] += force[2];
        let node = mesh.topology.nodes[node_index].coordinates_m;
        let arm = [
            node[0] - point_m[0],
            node[1] - point_m[1],
            node[2] - point_m[2],
        ];
        let moment = cross(arm, force);
        applied_moment[0] += moment[0];
        applied_moment[1] += moment[1];
        applied_moment[2] += moment[2];
    }
    (applied_force, applied_moment)
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

fn triangle_area_normal(nodes: [[f64; 3]; 3]) -> [f64; 3] {
    scale_vec(
        cross(
            [
                nodes[1][0] - nodes[0][0],
                nodes[1][1] - nodes[0][1],
                nodes[1][2] - nodes[0][2],
            ],
            [
                nodes[2][0] - nodes[0][0],
                nodes[2][1] - nodes[0][1],
                nodes[2][2] - nodes[0][2],
            ],
        ),
        0.5,
    )
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn scale_vec(value: [f64; 3], scale: f64) -> [f64; 3] {
    [value[0] * scale, value[1] * scale, value[2] * scale]
}

fn mat_vec(matrix: [[f64; 3]; 3], value: [f64; 3]) -> [f64; 3] {
    [
        dot(matrix[0], value),
        dot(matrix[1], value),
        dot(matrix[2], value),
    ]
}

fn invert_3x3(matrix: [[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let m = matrix;
    let c00 = m[1][1] * m[2][2] - m[1][2] * m[2][1];
    let c01 = -(m[1][0] * m[2][2] - m[1][2] * m[2][0]);
    let c02 = m[1][0] * m[2][1] - m[1][1] * m[2][0];
    let c10 = -(m[0][1] * m[2][2] - m[0][2] * m[2][1]);
    let c11 = m[0][0] * m[2][2] - m[0][2] * m[2][0];
    let c12 = -(m[0][0] * m[2][1] - m[0][1] * m[2][0]);
    let c20 = m[0][1] * m[1][2] - m[0][2] * m[1][1];
    let c21 = -(m[0][0] * m[1][2] - m[0][2] * m[1][0]);
    let c22 = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    let det = m[0][0] * c00 + m[0][1] * c01 + m[0][2] * c02;
    if det.abs() <= 1.0e-18 {
        return None;
    }
    let inv_det = 1.0 / det;
    Some([
        [c00 * inv_det, c10 * inv_det, c20 * inv_det],
        [c01 * inv_det, c11 * inv_det, c21 * inv_det],
        [c02 * inv_det, c12 * inv_det, c22 * inv_det],
    ])
}
