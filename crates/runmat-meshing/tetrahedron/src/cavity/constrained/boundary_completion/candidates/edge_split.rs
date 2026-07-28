use std::collections::BTreeMap;

use runmat_meshing_core::{
    quality::predicate::{
        point_in_closed_triangle_surface, tetrahedron_centroid, Point3, PointInClosedSurface,
        Triangle3,
    },
    quality::tolerance::MeshingTolerance,
};

use super::super::super::{
    boundary_splits::{
        boundary_face_edge_split_node_candidates, edge_split_completion_tetrahedra_for_node,
    },
    cavity_boundary_node_ids, split_constrained_cavity_boundary_faces_on_edge,
    topology::{sorted_face, sorted_tetrahedron_nodes},
    validate_constrained_cavity, ConstrainedCavity, ConstrainedCavityNode,
    ConstrainedCavityRefillOptions, ConstrainedCavityRefillTetrahedron,
    ConstrainedCavityValidationError,
};

pub(in super::super::super) fn best_boundary_face_edge_split_completion(
    face: [u32; 3],
    cavity: &ConstrainedCavity,
    boundary_nodes: &BTreeMap<u32, Point3>,
    boundary_triangles: &[Triangle3],
    refill_tetrahedra: &[ConstrainedCavityRefillTetrahedron],
    options: ConstrainedCavityRefillOptions,
) -> Result<
    Option<(
        ConstrainedCavity,
        ConstrainedCavityNode,
        Vec<ConstrainedCavityRefillTetrahedron>,
    )>,
    ConstrainedCavityValidationError,
> {
    let split_candidates = boundary_face_edge_split_node_candidates(face, boundary_nodes);
    let mut best = None::<(
        [u32; 2],
        ConstrainedCavityNode,
        Vec<ConstrainedCavityRefillTetrahedron>,
        f64,
    )>;
    for cap_node_id in cavity_boundary_node_ids(cavity) {
        if face.contains(&cap_node_id) {
            continue;
        }
        for (edge, split_node) in &split_candidates {
            let Some(child_tetrahedra) = edge_split_completion_tetrahedra_for_node(
                face,
                *edge,
                cap_node_id,
                split_node,
                boundary_nodes,
                options,
            ) else {
                continue;
            };
            if child_tetrahedra.iter().any(|tetrahedron| {
                let tetrahedron_points = tetrahedron.node_ids.map(|node_id| {
                    if node_id == split_node.node_id {
                        split_node.coordinates_m
                    } else {
                        boundary_nodes[&node_id]
                    }
                });
                point_in_closed_triangle_surface(
                    tetrahedron_centroid(tetrahedron_points),
                    boundary_triangles,
                    MeshingTolerance::default(),
                ) != PointInClosedSurface::Inside
            }) {
                continue;
            }
            if child_tetrahedra.iter().any(|tetrahedron| {
                refill_tetrahedra.iter().any(|existing| {
                    sorted_tetrahedron_nodes(existing.node_ids)
                        == sorted_tetrahedron_nodes(tetrahedron.node_ids)
                })
            }) {
                continue;
            }
            let min_quality = child_tetrahedra
                .iter()
                .map(|tetrahedron| tetrahedron.exact_scaled_jacobian)
                .fold(f64::INFINITY, f64::min);
            if best
                .as_ref()
                .is_none_or(|(_, _, _, best_quality)| min_quality > *best_quality)
            {
                best = Some((*edge, split_node.clone(), child_tetrahedra, min_quality));
            }
        }
    }
    let Some((edge, split_node, split_tetrahedra, _)) = best else {
        return Ok(None);
    };
    let split_faces = split_constrained_cavity_boundary_faces_on_edge(
        &cavity.boundary_faces,
        face,
        edge,
        split_node.node_id,
    )
    .map_err(|_| ConstrainedCavityValidationError::MissingBoundaryFace {
        node_ids: sorted_face(face),
    })?;
    let mut split_cavity = cavity.clone();
    split_cavity.boundary_faces = split_faces;
    validate_constrained_cavity(&split_cavity)?;
    Ok(Some((split_cavity, split_node, split_tetrahedra)))
}
