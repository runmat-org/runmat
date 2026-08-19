use std::collections::BTreeMap;

use runmat_meshing_core::quality::predicate::Triangle3;

use super::*;

pub(super) struct MissingFaceStitchSetup {
    pub(super) boundary_node_map: BTreeMap<u32, Point3>,
    pub(super) boundary_triangles: Vec<Triangle3>,
    pub(super) boundary_node_ids: Vec<u32>,
    pub(super) boundary_refill_tetrahedra: Vec<ConstrainedCavityRefillTetrahedron>,
    pub(super) missing_faces: Vec<[u32; 3]>,
    pub(super) missing_face_patches: Vec<Vec<usize>>,
    pub(super) diagnostic: MissingFaceLocalCapStitchDiagnostic,
}

pub(super) fn missing_face_stitch_setup(
    cavity: &ConstrainedCavity,
    boundary_nodes: &[ConstrainedCavityNode],
    options: ConstrainedCavityRefillOptions,
    patch_link: MissingFaceLink,
) -> Result<MissingFaceStitchSetup, ConstrainedCavityRefillError> {
    validate_refill_options(options)?;
    validate_constrained_cavity(cavity).map_err(ConstrainedCavityRefillError::Validation)?;
    let boundary_node_map = boundary_node_coordinates(cavity, boundary_nodes)?;
    let boundary_triangles = cavity_boundary_triangles(cavity, &boundary_node_map)?;
    let boundary_node_ids = cavity_boundary_node_ids(cavity)
        .into_iter()
        .collect::<Vec<_>>();
    let points = boundary_node_ids
        .iter()
        .map(|node_id| ConnectivityPoint {
            node_id: *node_id,
            coordinates_m: boundary_node_map[node_id],
            is_super: false,
        })
        .collect::<Vec<_>>();
    let mut boundary_refill_tetrahedra = Vec::<ConstrainedCavityRefillTetrahedron>::new();
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
            boundary_refill_tetrahedra.push(tetrahedron);
        }
    }
    let missing_faces = missing_refill_boundary_faces(cavity, &boundary_refill_tetrahedra)
        .map_err(ConstrainedCavityRefillError::Validation)?;
    let missing_face_patches = missing_face_components(&missing_faces, patch_link);
    let diagnostic = MissingFaceLocalCapStitchDiagnostic {
        missing_face_count: missing_faces.len(),
        missing_faces: missing_faces.clone(),
        patch_count: missing_face_patches.len(),
        patch_size_histogram: component_size_histogram(
            missing_face_patches
                .iter()
                .map(Vec::len)
                .collect::<Vec<_>>(),
        ),
        patch_capped_face_count_histogram: BTreeMap::new(),
        incomplete_patch_size_histogram: BTreeMap::new(),
        uncapped_faces: Vec::new(),
        capped_face_count: 0,
        inserted_node_count: 0,
        candidate_tetrahedron_count: 0,
        cap_side_face_count: 0,
        zero_mate_cap_side_face_count: 0,
        min_cap_side_face_mate_count: 0,
        max_cap_side_face_mate_count: 0,
        open_interior_face_count: 0,
        open_interior_component_count: 0,
        open_interior_component_size_histogram: BTreeMap::new(),
        candidate_with_orphan_interior_face_count: 0,
        candidate_without_orphan_interior_face_count: 0,
        root_boundary_zero_raw_candidate_face_count: 0,
        root_boundary_zero_addable_candidate_face_count: 0,
        root_boundary_min_raw_candidate_count: 0,
        root_boundary_min_addable_candidate_count: 0,
        root_boundary_max_addable_candidate_count: 0,
        cover_dead_end_reason: "not_evaluated",
        cover_dead_end_depth: 0,
        cover_dead_end_reason_histogram: BTreeMap::new(),
        selected_tetrahedron_count: 0,
        search_attempt_count: 0,
        found_cover: false,
        reason: "not_evaluated",
        max_min_scaled_jacobian: 0.0,
    };
    Ok(MissingFaceStitchSetup {
        boundary_node_map,
        boundary_triangles,
        boundary_node_ids,
        boundary_refill_tetrahedra,
        missing_faces,
        missing_face_patches,
        diagnostic,
    })
}
