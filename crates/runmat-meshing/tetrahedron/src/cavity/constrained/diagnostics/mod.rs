use super::*;

mod boundary_node_completion;
pub(crate) use boundary_node_completion::diagnostic_boundary_node_completion;
mod interior_star;
pub(crate) use interior_star::diagnostic_interior_star_quality;
mod local_cap_quality;
pub(crate) use local_cap_quality::diagnostic_missing_face_local_cap_quality;
mod missing_face_stitch;
pub(crate) use missing_face_stitch::{
    diagnostic_missing_face_edge_subpatch_cap_stitch,
    diagnostic_missing_face_hybrid_subpatch_cap_stitch, diagnostic_missing_face_local_cap_stitch,
    diagnostic_missing_face_shared_patch_cap_stitch,
};

#[cfg(test)]
pub(crate) fn diagnostic_boundary_missing_face_clusters(
    cavity: &ConstrainedCavity,
    boundary_nodes: &[ConstrainedCavityNode],
    options: ConstrainedCavityRefillOptions,
) -> Result<BoundaryMissingFaceClusterDiagnostic, ConstrainedCavityRefillError> {
    validate_refill_options(options)?;
    validate_constrained_cavity(cavity).map_err(ConstrainedCavityRefillError::Validation)?;
    let boundary_node_map = boundary_node_coordinates(cavity, boundary_nodes)?;
    let boundary_triangles = cavity_boundary_triangles(cavity, &boundary_node_map)?;
    let points = cavity_boundary_node_ids(cavity)
        .into_iter()
        .map(|node_id| ConnectivityPoint {
            node_id,
            coordinates_m: boundary_node_map[&node_id],
            is_super: false,
        })
        .collect::<Vec<_>>();
    let mut refill_tetrahedra = Vec::<ConstrainedCavityRefillTetrahedron>::new();
    for tetrahedron in tetrahedralize_points(&points) {
        let node_ids = tetrahedron.vertices.map(|index| points[index].node_id);
        let tetrahedron_points = tetrahedron
            .vertices
            .map(|index| points[index].coordinates_m);
        if point_in_closed_triangle_surface(
            tetrahedron_centroid(tetrahedron_points),
            &boundary_triangles,
            MeshingTolerance::default(),
        ) != PointInClosedSurface::Inside
        {
            continue;
        }
        if let Ok(tetrahedron) =
            raw_refill_tetrahedron_with_rejection_reason(node_ids, tetrahedron_points, options)
        {
            refill_tetrahedra.push(tetrahedron);
        }
    }
    let missing_faces = missing_refill_boundary_faces(cavity, &refill_tetrahedra)
        .map_err(ConstrainedCavityRefillError::Validation)?;
    let edge_component_sizes = missing_face_component_sizes(&missing_faces, MissingFaceLink::Edge);
    let node_components = missing_face_components(&missing_faces, MissingFaceLink::Node);
    let node_component_sizes = node_components.iter().map(Vec::len).collect::<Vec<_>>();
    let mut node_component_common_node_count_histogram = BTreeMap::<usize, usize>::new();
    let mut node_component_common_node_ids = BTreeMap::<u32, usize>::new();
    for component in &node_components {
        let common_node_ids = missing_face_component_common_node_ids(&missing_faces, component);
        *node_component_common_node_count_histogram
            .entry(common_node_ids.len())
            .or_default() += 1;
        for node_id in common_node_ids {
            *node_component_common_node_ids.entry(node_id).or_default() += 1;
        }
    }
    Ok(BoundaryMissingFaceClusterDiagnostic {
        missing_face_count: missing_faces.len(),
        edge_component_count: edge_component_sizes.len(),
        edge_component_size_histogram: component_size_histogram(edge_component_sizes),
        node_component_count: node_component_sizes.len(),
        node_component_size_histogram: component_size_histogram(node_component_sizes),
        node_component_common_node_count_histogram,
        node_component_common_node_ids,
    })
}
