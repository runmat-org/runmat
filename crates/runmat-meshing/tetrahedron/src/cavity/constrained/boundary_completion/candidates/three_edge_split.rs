use std::collections::BTreeMap;

use super::super::BoundaryFaceCompletion;

use runmat_meshing_core::{
    quality::predicate::{
        point_in_closed_triangle_surface, tetrahedron_centroid, Point3, PointInClosedSurface,
        Triangle3,
    },
    quality::tolerance::MeshingTolerance,
};

use super::super::super::{
    boundary_splits::{
        boundary_face_mid_edge_split_nodes, three_edge_split_completion_tetrahedra_for_node,
    },
    cavity_boundary_node_ids, split_constrained_cavity_boundary_faces_on_three_edges,
    topology::{face_edges, sorted_edge, sorted_face, sorted_tetrahedron_nodes},
    validate_constrained_cavity, ConstrainedCavity, ConstrainedCavityRefillOptions,
    ConstrainedCavityRefillTetrahedron, ConstrainedCavityValidationError,
};

pub(in super::super::super) fn best_boundary_face_three_edge_split_completion(
    face: [u32; 3],
    cavity: &ConstrainedCavity,
    boundary_nodes: &BTreeMap<u32, Point3>,
    boundary_triangles: &[Triangle3],
    refill_tetrahedra: &[ConstrainedCavityRefillTetrahedron],
    options: ConstrainedCavityRefillOptions,
) -> Result<Option<BoundaryFaceCompletion>, ConstrainedCavityValidationError> {
    let split_nodes = boundary_face_mid_edge_split_nodes(face, boundary_nodes);
    let split_node_by_edge = face_edges(face)
        .into_iter()
        .zip(split_nodes.iter())
        .map(|(edge, node)| (sorted_edge(edge), node.node_id))
        .collect::<BTreeMap<_, _>>();
    let split_node_coordinates = split_nodes
        .iter()
        .map(|node| (node.node_id, node.coordinates_m))
        .collect::<BTreeMap<_, _>>();
    let mut best = None::<(Vec<ConstrainedCavityRefillTetrahedron>, f64)>;
    for cap_node_id in cavity_boundary_node_ids(cavity) {
        if face.contains(&cap_node_id) {
            continue;
        }
        let Some(child_tetrahedra) = three_edge_split_completion_tetrahedra_for_node(
            face,
            cap_node_id,
            &split_node_by_edge,
            &split_node_coordinates,
            boundary_nodes,
            options,
        ) else {
            continue;
        };
        if child_tetrahedra.iter().any(|tetrahedron| {
            let tetrahedron_points = tetrahedron.node_ids.map(|node_id| {
                split_node_coordinates
                    .get(&node_id)
                    .copied()
                    .unwrap_or_else(|| boundary_nodes[&node_id])
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
            .is_none_or(|(_, best_quality)| min_quality > *best_quality)
        {
            best = Some((child_tetrahedra, min_quality));
        }
    }
    let Some((split_tetrahedra, _)) = best else {
        return Ok(None);
    };
    let split_faces = split_constrained_cavity_boundary_faces_on_three_edges(
        &cavity.boundary_faces,
        face,
        split_node_by_edge,
    )
    .map_err(|_| ConstrainedCavityValidationError::MissingBoundaryFace {
        node_ids: sorted_face(face),
    })?;
    let mut split_cavity = cavity.clone();
    split_cavity.boundary_faces = split_faces;
    validate_constrained_cavity(&split_cavity)?;
    Ok(Some((split_cavity, split_nodes, split_tetrahedra)))
}
