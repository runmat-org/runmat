use std::collections::{BTreeMap, BTreeSet};

use runmat_meshing_core::{
    contracts::{ProtectedBoundaryComplex, TopologyEntityId},
    quality::predicate::{
        orient_tetrahedron_node_ids, tetrahedron_edge_aspect_ratio, tetrahedron_scaled_jacobian,
    },
};

use crate::{
    cavity::constrained::{ConstrainedCavityRefill, ConstrainedCavityRefillTetrahedron},
    generate::TetrahedronGenerationError,
};

use super::{
    refill::{NestedTetrahedronShellRefill, NestedTetrahedronShellRefillStrategy},
    shell::NestedTetrahedronShell,
};

mod boundary;
mod builder;
mod cells;
mod geometry;
mod shape;

use boundary::{
    partition_boundary_faces, shell_source_faces, sorted_face_vec, triangulated_polygon_faces,
};
use builder::PartitionBuilder;
use cells::{cell_centroid, cell_faces, partition_cells};
use shape::affine_inner_tetrahedron_partition;

const MIN_PARTITION_SCALED_JACOBIAN: f64 = 0.15;

pub(super) fn barycentric_partition_refill(
    plc: &ProtectedBoundaryComplex,
    shell: &NestedTetrahedronShell,
    cavity_id_to_node_id: &BTreeMap<u32, TopologyEntityId>,
    node_id_to_cavity_id: &BTreeMap<TopologyEntityId, u32>,
    target_volume_m3: f64,
) -> Result<Option<NestedTetrahedronShellRefill>, TetrahedronGenerationError> {
    if shell.outer_node_ids.len() != 4 || shell.inner_node_ids.len() != 4 || plc.nodes.len() != 8 {
        return Ok(None);
    }
    let coordinates_by_node_id = plc
        .nodes
        .iter()
        .map(|node| (node.node_id.clone(), node.coordinates_m))
        .collect::<BTreeMap<_, _>>();
    let outer_points = shell
        .outer_node_ids
        .iter()
        .map(|node_id| {
            coordinates_by_node_id.get(node_id).copied().ok_or_else(|| {
                TetrahedronGenerationError::MissingPlcNode {
                    node_id: node_id.id.clone(),
                }
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let partition =
        affine_inner_tetrahedron_partition(shell, &coordinates_by_node_id, &outer_points)?;
    let Some(partition) = partition else {
        return Ok(None);
    };

    let mut builder = PartitionBuilder::new(cavity_id_to_node_id.clone(), outer_points.clone());
    for (index, node_id) in shell.outer_node_ids.iter().enumerate() {
        let cavity_id = cavity_node_id(node_id_to_cavity_id, node_id)?;
        let mut barycentric = [0.0; 4];
        barycentric[index] = 1.0;
        builder.insert_existing_node(barycentric, cavity_id);
    }
    for (index, node_id) in partition.inner_node_ids.iter().enumerate() {
        let cavity_id = cavity_node_id(node_id_to_cavity_id, node_id)?;
        let mut barycentric = partition.inner_lower_bounds;
        barycentric[index] += partition.inner_scale;
        builder.insert_existing_node(barycentric, cavity_id);
    }

    let cells = partition_cells(
        partition.divisions,
        partition.inner_lower_bounds,
        &mut builder,
    );
    if cells.is_empty() {
        return Ok(None);
    }

    let outer_source_faces = shell_source_faces(plc, &shell.outer_node_ids)?;
    let inner_source_faces = shell_source_faces(plc, &partition.inner_node_ids)?;
    let mut face_triangle_cache = BTreeMap::<Vec<u32>, Vec<[u32; 3]>>::new();
    let mut tetrahedra = Vec::<ConstrainedCavityRefillTetrahedron>::new();
    let mut total_volume_m3 = 0.0_f64;
    let mut min_scaled_jacobian = f64::INFINITY;

    for cell in &cells {
        let centroid_id = builder.insert_generated_node(cell_centroid(cell, &builder));
        let mut seen_cell_faces = BTreeSet::<Vec<u32>>::new();
        for face in cell_faces(cell, &builder) {
            let key = sorted_face_vec(&face);
            if !seen_cell_faces.insert(key.clone()) {
                continue;
            }
            let triangles = match face_triangle_cache.get(&key) {
                Some(triangles) => triangles.clone(),
                None => {
                    let triangles = triangulated_polygon_faces(&key, &builder);
                    face_triangle_cache.insert(key, triangles.clone());
                    triangles
                }
            };
            for triangle in triangles {
                let node_ids = [centroid_id, triangle[0], triangle[1], triangle[2]];
                let points = node_ids.map(|node_id| builder.coordinates(node_id));
                let (node_ids, volume_m3) = orient_tetrahedron_node_ids(node_ids, points);
                if volume_m3 <= 1.0e-18 {
                    continue;
                }
                let points = node_ids.map(|node_id| builder.coordinates(node_id));
                let scaled_jacobian = tetrahedron_scaled_jacobian(points);
                if !scaled_jacobian.is_finite() {
                    return Ok(None);
                }
                min_scaled_jacobian = min_scaled_jacobian.min(scaled_jacobian);
                total_volume_m3 += volume_m3;
                tetrahedra.push(ConstrainedCavityRefillTetrahedron {
                    node_ids,
                    volume_m3,
                    aspect_ratio: tetrahedron_edge_aspect_ratio(points),
                    exact_scaled_jacobian: scaled_jacobian,
                });
            }
        }
    }
    if tetrahedra.is_empty()
        || min_scaled_jacobian < MIN_PARTITION_SCALED_JACOBIAN
        || ((total_volume_m3 - target_volume_m3).abs() > target_volume_m3.abs().max(1.0) * 1.0e-9)
    {
        return Ok(None);
    }

    let boundary_faces = partition_boundary_faces(
        &tetrahedra,
        &builder,
        partition.inner_lower_bounds,
        &outer_source_faces,
        &inner_source_faces,
    )?;
    Ok(Some(NestedTetrahedronShellRefill {
        cavity_id_to_node_id: builder.cavity_id_to_node_id,
        generated_nodes: builder.generated_nodes,
        strategy: NestedTetrahedronShellRefillStrategy::BarycentricPartition,
        boundary_centroid_refinement_attempted: false,
        boundary_centroid_refinement_rejected: false,
        refill: ConstrainedCavityRefill {
            tetrahedra,
            boundary_faces,
            inserted_nodes: Vec::new(),
            total_volume_m3,
        },
    }))
}

fn cavity_node_id(
    node_id_to_cavity_id: &BTreeMap<TopologyEntityId, u32>,
    node_id: &TopologyEntityId,
) -> Result<u32, TetrahedronGenerationError> {
    node_id_to_cavity_id.get(node_id).copied().ok_or_else(|| {
        TetrahedronGenerationError::MissingPlcNode {
            node_id: node_id.id.clone(),
        }
    })
}
