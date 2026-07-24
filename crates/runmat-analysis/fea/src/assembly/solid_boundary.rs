use runmat_analysis_core::{AnalysisModel, BoundaryConditionKind, LoadKind};
use runmat_meshing_core::AnalysisMeshArtifact;

use super::{
    dofs::{StructuralDofKind, StructuralDofLayout},
    AnalysisMeshRegionMappingError, WrenchLoweringSummary,
};

pub(super) fn apply_analysis_mesh_structural_regions(
    model: &AnalysisModel,
    mesh: &AnalysisMeshArtifact,
    layout: &StructuralDofLayout,
    constrained: &mut [bool],
    rhs: &mut [f64],
    strict_region_mapping: bool,
) -> Result<Vec<WrenchLoweringSummary>, AnalysisMeshRegionMappingError> {
    rhs.fill(0.0);
    constrained.fill(false);

    let mut wrench_lowering = Vec::new();
    for load in &model.loads {
        let target_nodes = analysis_mesh_region_node_indices(mesh, &load.region_id);
        match load.kind {
            LoadKind::Force { fx, fy, fz } => {
                if target_nodes.is_empty() {
                    if strict_region_mapping {
                        return Err(AnalysisMeshRegionMappingError::UnmappedLoadRegion {
                            load_id: load.load_id.clone(),
                            region_id: load.region_id.clone(),
                            load_kind: "force",
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
            LoadKind::Wrench {
                fx,
                fy,
                fz,
                mx,
                my,
                mz,
                px,
                py,
                pz,
            } => {
                if target_nodes.is_empty() {
                    if strict_region_mapping {
                        return Err(AnalysisMeshRegionMappingError::UnmappedLoadRegion {
                            load_id: load.load_id.clone(),
                            region_id: load.region_id.clone(),
                            load_kind: "wrench",
                        });
                    }
                    continue;
                }
                let summary = add_analysis_mesh_wrench_rhs(
                    mesh,
                    layout,
                    rhs,
                    &target_nodes,
                    [fx, fy, fz],
                    [mx, my, mz],
                    [px, py, pz],
                );
                wrench_lowering.push(WrenchLoweringSummary {
                    load_id: load.load_id.clone(),
                    region_id: load.region_id.clone(),
                    ..summary
                });
            }
            LoadKind::Pressure { magnitude_pa } => {
                if strict_region_mapping
                    && !mesh.boundary_faces.iter().any(|face| {
                        face.region_ids
                            .iter()
                            .any(|region| region == &load.region_id)
                            && face.node_ids.len() == 3
                    })
                {
                    return Err(AnalysisMeshRegionMappingError::UnmappedLoadRegion {
                        load_id: load.load_id.clone(),
                        region_id: load.region_id.clone(),
                        load_kind: "pressure",
                    });
                }
                apply_analysis_mesh_pressure(mesh, layout, rhs, &load.region_id, magnitude_pa);
            }
            LoadKind::BodyForce { gx, gy, gz } => {
                let all_nodes = (0..mesh.nodes.len()).collect::<Vec<_>>();
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

    for bc in &model.boundary_conditions {
        let target_nodes = analysis_mesh_region_node_indices(mesh, &bc.region_id);
        if strict_region_mapping
            && target_nodes.is_empty()
            && matches!(
                bc.kind,
                BoundaryConditionKind::Fixed | BoundaryConditionKind::PrescribedDisplacement
            )
        {
            return Err(
                AnalysisMeshRegionMappingError::UnmappedBoundaryConditionRegion {
                    bc_id: bc.bc_id.clone(),
                    region_id: bc.region_id.clone(),
                    boundary_condition_kind: match bc.kind {
                        BoundaryConditionKind::Fixed => "fixed",
                        BoundaryConditionKind::PrescribedDisplacement => "prescribed_displacement",
                        _ => "unsupported",
                    },
                },
            );
        }
        for node_index in target_nodes {
            match bc.kind {
                BoundaryConditionKind::Fixed | BoundaryConditionKind::PrescribedDisplacement => {
                    for kind in [
                        StructuralDofKind::Ux,
                        StructuralDofKind::Uy,
                        StructuralDofKind::Uz,
                    ] {
                        constrain_dof(layout, constrained, rhs, node_index, kind, 0.0);
                    }
                }
                _ => {}
            }
        }
    }
    Ok(wrench_lowering)
}

fn analysis_mesh_region_node_indices(mesh: &AnalysisMeshArtifact, region_id: &str) -> Vec<usize> {
    let mut node_ids = Vec::<u32>::new();
    for face in &mesh.boundary_faces {
        if !face.region_ids.iter().any(|region| region == region_id) {
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
        .filter_map(|node_id| mesh.nodes.iter().position(|node| node.node_id == node_id))
        .collect()
}

fn apply_analysis_mesh_pressure(
    mesh: &AnalysisMeshArtifact,
    layout: &StructuralDofLayout,
    rhs: &mut [f64],
    region_id: &str,
    magnitude_pa: f64,
) {
    for face in &mesh.boundary_faces {
        if !face.region_ids.iter().any(|region| region == region_id) || face.node_ids.len() != 3 {
            continue;
        }
        let Some(indices) = analysis_mesh_face_node_indices(mesh, &face.node_ids) else {
            continue;
        };
        let coordinates = [
            mesh.nodes[indices[0]].coordinates_m,
            mesh.nodes[indices[1]].coordinates_m,
            mesh.nodes[indices[2]].coordinates_m,
        ];
        let force = scale_vec(triangle_area_normal(coordinates), magnitude_pa);
        let nodal_force = scale_vec(force, 1.0 / 3.0);
        for node_index in indices {
            add_translational_rhs(layout, rhs, node_index, nodal_force);
        }
    }
}

fn add_analysis_mesh_wrench_rhs(
    mesh: &AnalysisMeshArtifact,
    layout: &StructuralDofLayout,
    rhs: &mut [f64],
    target_nodes: &[usize],
    force: [f64; 3],
    moment_at_point: [f64; 3],
    point_m: [f64; 3],
) -> WrenchLoweringSummary {
    if target_nodes.is_empty() {
        return WrenchLoweringSummary {
            load_id: String::new(),
            region_id: String::new(),
            target_node_count: 0,
            applied_force: [0.0; 3],
            applied_moment_at_point: [0.0; 3],
            force_residual: force,
            moment_residual: moment_at_point,
            moment_couple_applied: false,
        };
    }

    let centroid = analysis_mesh_target_centroid(mesh, target_nodes);
    let scale = 1.0 / target_nodes.len() as f64;
    let mut nodal_forces = Vec::with_capacity(target_nodes.len());
    for &node_index in target_nodes {
        let nodal_force = scale_vec(force, scale);
        add_translational_rhs(layout, rhs, node_index, nodal_force);
        nodal_forces.push(nodal_force);
    }

    let force_arm = [
        centroid[0] - point_m[0],
        centroid[1] - point_m[1],
        centroid[2] - point_m[2],
    ];
    let force_moment = cross(force_arm, force);
    let couple = [
        moment_at_point[0] - force_moment[0],
        moment_at_point[1] - force_moment[1],
        moment_at_point[2] - force_moment[2],
    ];
    let mut moment_couple_applied = false;

    if !couple
        .iter()
        .all(|component| component.abs() <= f64::EPSILON)
    {
        let mut coupling = [[0.0_f64; 3]; 3];
        let offsets = target_nodes
            .iter()
            .map(|&node_index| {
                let node = mesh.nodes[node_index].coordinates_m;
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
            let lambda = mat_vec(inv, couple);
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
            moment_couple_applied = true;
        }
    }

    let (applied_force, applied_moment_at_point) =
        analysis_mesh_wrench_resultants(mesh, target_nodes, &nodal_forces, point_m);
    WrenchLoweringSummary {
        load_id: String::new(),
        region_id: String::new(),
        target_node_count: target_nodes.len(),
        applied_force,
        applied_moment_at_point,
        force_residual: [
            force[0] - applied_force[0],
            force[1] - applied_force[1],
            force[2] - applied_force[2],
        ],
        moment_residual: [
            moment_at_point[0] - applied_moment_at_point[0],
            moment_at_point[1] - applied_moment_at_point[1],
            moment_at_point[2] - applied_moment_at_point[2],
        ],
        moment_couple_applied,
    }
}

fn analysis_mesh_face_node_indices(
    mesh: &AnalysisMeshArtifact,
    node_ids: &[u32],
) -> Option<[usize; 3]> {
    if node_ids.len() != 3 {
        return None;
    }
    Some([
        mesh.nodes
            .iter()
            .position(|node| node.node_id == node_ids[0])?,
        mesh.nodes
            .iter()
            .position(|node| node.node_id == node_ids[1])?,
        mesh.nodes
            .iter()
            .position(|node| node.node_id == node_ids[2])?,
    ])
}

fn analysis_mesh_target_centroid(mesh: &AnalysisMeshArtifact, target_nodes: &[usize]) -> [f64; 3] {
    let mut centroid = [0.0_f64; 3];
    for &node_index in target_nodes {
        let node = mesh.nodes[node_index].coordinates_m;
        centroid[0] += node[0];
        centroid[1] += node[1];
        centroid[2] += node[2];
    }
    scale_vec(centroid, 1.0 / target_nodes.len() as f64)
}

fn analysis_mesh_wrench_resultants(
    mesh: &AnalysisMeshArtifact,
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
        let node = mesh.nodes[node_index].coordinates_m;
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

fn constrain_dof(
    layout: &StructuralDofLayout,
    constrained: &mut [bool],
    rhs: &mut [f64],
    node_index: usize,
    kind: StructuralDofKind,
    value: f64,
) {
    if let Some(dof) = layout.index(node_index, kind) {
        constrained[dof] = true;
        rhs[dof] = value;
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
